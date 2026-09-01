"""Actual CLI and script entry points share temporary Store; all providers fail closed."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from cli.main import app
from cryptotrader.backtest.service import BacktestService
from cryptotrader.backtest.store import BacktestStore
from cryptotrader.db import dispose_engine
from cryptotrader.runtime_config.models import SignalComponentConfig
from cryptotrader.tasks import BackgroundTaskManager
from tests.factories.research import EmptyEngine, ResearchRepository, no_registries
from tests.factories.research_offline import research_offline  # noqa: F401


async def service_for(tmp_path, monkeypatch):
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    url = f"sqlite+aiosqlite:///{tmp_path}/entrypoints.sqlite"
    await migrate_workbench_schema(url)
    repository = ResearchRepository(url)
    document = repository.snapshot.document
    signals = document.signals.model_copy(
        update={
            "components": (
                SignalComponentConfig(component_id="kronos", enabled=False, weight=0.4, parameters={"sample_count": 3}),
            )
        }
    )
    repository.snapshot = replace(repository.snapshot, document=document.model_copy(update={"signals": signals}))
    service = BacktestService(
        repository=repository,
        task_manager=BackgroundTaskManager(),
        engine_factory=EmptyEngine,
        registry_provider=no_registries,
    )
    monkeypatch.setattr("cryptotrader.backtest.service.configured_service", lambda: service)
    # Even if an entry regresses to the retired direct-engine branch it cannot read outside fixtures.
    monkeypatch.setattr(
        "cryptotrader.backtest.engine.BacktestEngine",
        lambda **_: (_ for _ in ()).throw(AssertionError("entrypoint bypassed the persistent service")),
    )
    return service, url


def test_real_cli_persists_unnamed_run_that_a_new_store_can_open(tmp_path, monkeypatch):
    _, url = asyncio.run(service_for(tmp_path, monkeypatch))
    result = CliRunner().invoke(app, ["backtest", "--start", "2024-01-01", "--end", "2024-01-02"])
    assert result.exit_code == 0, result.output

    async def check():
        await dispose_engine(url)
        runs = await BacktestStore(url).list()
        assert len(runs) == 1
        assert runs[0].status == "completed"
        assert runs[0].params.name is None
        assert runs[0].run_id in result.output
        await dispose_engine(url)

    asyncio.run(check())


@pytest.mark.asyncio
async def test_run_script_reads_same_persisted_result(tmp_path, monkeypatch, capsys):
    from scripts import run_backtest

    _, url = await service_for(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_backtest,
        "_arguments",
        lambda: SimpleNamespace(pair="BTC/USDT", start="2024-01-01", end="2024-01-02", interval="1h", capital=1000),
    )
    await run_backtest.main()
    await dispose_engine(url)
    runs = await BacktestStore(url).list()
    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert runs[0].run_id in capsys.readouterr().out
    await dispose_engine(url)


@pytest.mark.asyncio
async def test_ab_script_persists_both_profiles_and_preserves_kronos_parameters(tmp_path, monkeypatch, capsys):
    from scripts import kronos_backtest_ab

    _, url = await service_for(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", ["kronos_backtest_ab", "--start", "2024-01-01", "--end", "2024-01-02"])
    await kronos_backtest_ab.main()
    await dispose_engine(url)
    runs = await BacktestStore(url).list()
    assert len(runs) == 2
    assert all(run.status == "completed" for run in runs)
    output = capsys.readouterr().out
    assert all(run.run_id in output for run in runs)
    assert "不自动排名" in output
    components = [run.config_snapshot["signals"]["components"][0] for run in runs]
    assert all(component["parameters"]["sample_count"] == 3 for component in components)
    assert {component["enabled"] for component in components} == {True, False}
    assert {component["weight"] for component in components} == {1.0, 0.4}
    await dispose_engine(url)
