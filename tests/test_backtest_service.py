"""The real task owner writes every transition; engine doubles never call external services."""

import asyncio

import pytest

from cryptotrader.backtest.models import BacktestParams
from cryptotrader.backtest.result import BacktestResult
from cryptotrader.backtest.service import BacktestService
from cryptotrader.backtest.store import BacktestStore
from cryptotrader.db import dispose_engine
from cryptotrader.tasks import BackgroundTaskManager
from tests.factories.backtest import replay_config

PARAMS = BacktestParams(pair="BTC/USDT", start="2024-01-01", end="2024-01-02")


async def _database_url(path) -> str:
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    database_url = f"sqlite+aiosqlite:///{path}"
    await migrate_workbench_schema(database_url)
    return database_url


async def no_registries(_snapshot):
    return None, None


class Repository:
    def __init__(self, url):
        self.database_url = url

    async def get_existing(self):
        return replay_config()


@pytest.mark.asyncio
async def test_queue_is_durable_before_admission_and_cancel_before_start(tmp_path):
    url = await _database_url(tmp_path / "queued.sqlite")
    manager = BackgroundTaskManager()
    service = BacktestService(repository=Repository(url), task_manager=manager)
    run_id = await service.start(PARAMS)
    manager.interrupt(run_id)
    await manager.drain()
    restored = await BacktestStore(url).get(run_id)
    assert restored.status == "canceled"
    assert restored.progress == 0
    assert restored.result is None
    await dispose_engine(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["failed", "canceled"])
async def test_model_evidence_survives_unsuccessful_run_and_store_recreation(tmp_path, termination):
    from langchain_core.messages import HumanMessage

    from cryptotrader.backtest.evidence import current_model_evidence

    url = await _database_url(tmp_path / "evidence.sqlite")
    manager = BackgroundTaskManager()
    reached = asyncio.Event()

    class Engine:
        def __init__(self, **kwargs):
            pass

        async def run(self):
            callback = current_model_evidence()
            callback.on_chat_model_start(
                {},
                [[HumanMessage(content="private original prompt")]],
                run_id="call-fixture",
                invocation_params={"model": "requested-only"},
            )
            reached.set()
            if termination == "failed":
                callback.on_llm_error(RuntimeError("do not persist transport secret"), run_id="call-fixture")
                raise RuntimeError("engine failed")
            await asyncio.Event().wait()

    service = BacktestService(
        repository=Repository(url), task_manager=manager, engine_factory=Engine, registry_provider=no_registries
    )
    run_id = await service.start(PARAMS)
    await asyncio.wait_for(reached.wait(), 3)
    if termination == "canceled":
        await service.cancel(run_id)
    await manager.drain()
    await dispose_engine(url)
    run = await BacktestStore(url).get(run_id)
    assert run.status == termination
    assert run.result is None
    assert len(run.model_evidence) == 1
    assert run.model_evidence[0]["requested_model"] == "requested-only"
    assert run.model_evidence[0]["actual_model"] is None
    assert len(run.model_evidence[0]["prompt_hash"]) == 64
    assert "private original" not in str(run.model_evidence)
    assert "transport secret" not in str(run.model_evidence)
    await dispose_engine(url)


@pytest.mark.asyncio
async def test_each_progress_and_empty_completion_are_durable(tmp_path):
    url = await _database_url(tmp_path / "completed.sqlite")
    manager = BackgroundTaskManager()
    reached = asyncio.Event()
    resume = asyncio.Event()

    class Engine:
        def __init__(self, **kwargs):
            self.progress = kwargs["progress_callback"]
            assert kwargs["interval"] == "1h"
            assert kwargs["snapshot"].document.execution.connections == ()

        async def run(self):
            await self.progress(0.42)
            reached.set()
            await resume.wait()
            return BacktestResult()

    service = BacktestService(
        repository=Repository(url), task_manager=manager, engine_factory=Engine, registry_provider=no_registries
    )
    run_id = await service.start(PARAMS)
    await asyncio.wait_for(reached.wait(), 3)
    stored = await BacktestStore(url).get(run_id)
    assert stored.status == "running"
    assert stored.progress == 0.42
    resume.set()
    await manager.drain()
    stored = await BacktestStore(url).get(run_id)
    assert stored.status == "completed"
    assert stored.result.win_rate is None
    assert "execution" not in stored.config_snapshot
    assert "base_url" not in stored.config_snapshot["llm"]
    assert stored.config_snapshot["risk"]["position"]["max_single_pct"] == 0.5
    await dispose_engine(url)
