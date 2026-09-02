"""Tests for database-Runtime-backed CLI discovery commands."""

import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import typer
from typer.testing import CliRunner

import cryptotrader.db as database
from cli.main import app
from cryptotrader.decision.models import CycleOutcome, CycleRequest
from cryptotrader.pair import Pair
from cryptotrader.runtime_config.models import RuntimeConfigSnapshot
from cryptotrader.signals.registry import ComponentMetadata
from tests.factories.runtime_config import runtime_document
from tests.runtime_lease import static_cycle_lease


def test_cli_unmounts_legacy_configuration_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("portfolio", "risk", "live-check", "migrate", "sync"):
        assert command not in result.output


def test_schema_migrate_command_creates_the_current_workbench_schema(tmp_path) -> None:
    database_path = tmp_path / "cli-migration.sqlite"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    result = CliRunner().invoke(
        app,
        ["schema", "migrate"],
        env={"DATABASE_URL": database_url},
    )

    assert result.exit_code == 0, result.output
    assert all(key[0] != database_url for key in database._engines)
    with sqlite3.connect(database_path) as connection:
        table_names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {
        "runtime_config",
        "account_snapshots",
        "multi_venue_cycles",
        "book_approvals",
        "component_evaluations",
        "backtest_runs",
        "business_alerts",
        "schedule_rules",
    } <= table_names


def test_run_command_has_no_graph_option() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "--graph" not in result.output
    assert "--mode" not in result.output
    assert "--exchange" not in result.output


@pytest.mark.asyncio
async def test_run_reloads_before_each_pair_and_closes_runtime() -> None:
    import asyncio

    from cli.main import _run

    class Cycle:
        def __init__(self) -> None:
            self.requests = []

        async def run(self, request):
            self.requests.append(request)
            return CycleOutcome("cli-cycle", 6, None, (), "no_change", "not_started", False)

    cycle = Cycle()
    execution_leases: list[str] = []

    def execution_lease(pair: str):
        execution_leases.append(pair)
        return static_cycle_lease(cycle)()

    snapshot = RuntimeConfigSnapshot(6, runtime_document(), datetime(2026, 8, 29, tzinfo=UTC))
    cycle.snapshot = snapshot
    runtime = SimpleNamespace(
        snapshot=snapshot,
        cycle=cycle,
        execution_lease=execution_lease,
        close=AsyncMock(),
    )
    tasks = {}

    async def start_trading(pair, revision, books):
        assert revision == 6
        assert books == ["simulation"]
        tasks[pair.canonical()] = asyncio.create_task(cycle.run(CycleRequest(pair)))
        return pair.canonical()

    runtime.run_service = SimpleNamespace(
        trading_scope=AsyncMock(return_value=SimpleNamespace(saved_revision=6)),
        start_trading=AsyncMock(side_effect=start_trading),
    )
    runtime.task_manager = SimpleNamespace(get=lambda key: SimpleNamespace(task=tasks[key]))

    with patch("cryptotrader.runtime.build_runtime", AsyncMock(return_value=runtime)):
        await _run(["BTC/USDT", "ETH/USDT"], ["simulation"])

    assert cycle.requests == [
        CycleRequest(Pair.parse("BTC/USDT")),
        CycleRequest(Pair.parse("ETH/USDT")),
    ]
    assert runtime.run_service.trading_scope.await_count == 2
    assert execution_leases == []
    runtime.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_run_requires_explicit_books_and_closes_runtime() -> None:
    from cli.main import _run

    runtime = SimpleNamespace(cycle=None, close=AsyncMock())
    with (
        patch("cryptotrader.runtime.build_runtime", AsyncMock(return_value=runtime)),
        pytest.raises(typer.Exit),
    ):
        await _run(["BTC/USDT"])

    runtime.close.assert_awaited_once_with()


def test_agent_list_reads_runtime_signal_registry_and_closes_runtime() -> None:
    from cryptotrader.cycle_events import MultiplexedCycleEventSink, NullCycleEventSink
    from cryptotrader.runtime import Runtime

    signal_registry = SimpleNamespace(
        metadata=lambda: (ComponentMetadata("llm_committee", "LLM committee", "Internal debate"),)
    )
    runtime = Runtime(
        snapshot=RuntimeConfigSnapshot(1, runtime_document(), datetime(2026, 8, 29, tzinfo=UTC)),
        repository=object(),
        cycle=None,
        sessions={},
        signal_registry=signal_registry,
        market_registry=object(),
        venue_registry=object(),
        events=MultiplexedCycleEventSink(NullCycleEventSink()),
    )
    runtime.close = AsyncMock()
    with patch("cryptotrader.runtime.build_runtime", AsyncMock(return_value=runtime)):
        result = CliRunner().invoke(app, ["agent", "list"])

    assert result.exit_code == 0
    assert "llm_committee" in result.output
    assert "enabled" in result.output
    runtime.close.assert_awaited_once_with()
