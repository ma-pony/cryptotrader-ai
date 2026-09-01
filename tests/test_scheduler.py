"""Scheduler 以固定数据库配置驱动可热重载的多资金池周期。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cryptotrader.decision.models import CycleOutcome, CycleRequest, TargetPosition
from cryptotrader.pair import Pair
from cryptotrader.runtime_config.models import SchedulerConfig
from cryptotrader.scheduler import Scheduler
from tests.factories.runtime_config import active_document


class _Cycle:
    def __init__(self, cycle_id: str = "cycle-1") -> None:
        self.cycle_id = cycle_id
        self.requests: list[CycleRequest] = []

    async def run(self, request: CycleRequest) -> CycleOutcome:
        self.requests.append(request)
        return CycleOutcome(
            self.cycle_id,
            4,
            TargetPosition("long", 0.5),
            (),
            "completed",
            "completed",
            False,
        )


class _Runtime:
    def __init__(self, cycle: _Cycle | None, pairs=("BTC/USDT",)) -> None:
        self.cycle = cycle
        self.snapshot = SimpleNamespace(revision=4, document=active_document())
        self.snapshot.document = self.snapshot.document.model_copy(
            update={"execution": self.snapshot.document.execution.model_copy(update={"pairs": pairs})}
        )
        if cycle is not None:
            cycle.snapshot = self.snapshot
        self.repository = SimpleNamespace(database_url=None)
        self.lease_count = 0
        self.execution_lease_count = 0
        self.close = AsyncMock()
        self.tasks = {}
        self.task_manager = SimpleNamespace(get=lambda key: SimpleNamespace(task=self.tasks[key]))
        self.run_service = SimpleNamespace(run_automatic=self.run_automatic)

    async def run_automatic(self, pair, source):
        async def work():
            async with self.execution_lease(pair) as cycle:
                return await cycle.run(CycleRequest(Pair.parse(pair), origin=source))

        self.tasks[pair] = asyncio.create_task(work())
        return pair

    @asynccontextmanager
    async def cycle_lease(self):
        self.lease_count += 1
        if self.cycle is None:
            raise RuntimeError("runtime configuration is not active")
        self.cycle.snapshot = self.snapshot
        yield self.cycle

    @asynccontextmanager
    async def execution_lease(self, _pair: str):
        """Production scheduler admission is pair-scoped, not a batch lease."""
        self.execution_lease_count += 1
        if self.cycle is None:
            raise RuntimeError("runtime configuration is not active")
        self.cycle.snapshot = self.snapshot
        yield self.cycle


def _config() -> SchedulerConfig:
    return SchedulerConfig(
        enabled=True,
        interval_minutes=60,
        daily_summary_hour=2,
    )


def test_scheduler_uses_fixed_config_without_execution_identity_fields() -> None:
    runtime = _Runtime(_Cycle(), ("BTC/USDT:USDT",))

    scheduler = Scheduler(_config(), runtime)

    assert scheduler.pairs == (Pair.parse("BTC/USDT:USDT"),)
    assert scheduler.interval_minutes == 60
    assert scheduler.daily_summary_hour == 2
    assert not hasattr(scheduler, "mode")


@pytest.mark.asyncio
async def test_scheduled_batch_reloads_once_and_uses_one_cycle_for_every_pair() -> None:
    cycle = _Cycle("revision-4-cycle")
    runtime = _Runtime(cycle, ("BTC/USDT", "ETH/USDT"))
    scheduler = Scheduler(_config(), runtime)

    await scheduler.run_once()

    assert runtime.execution_lease_count == 2
    assert [getattr(request, "origin", None) for request in cycle.requests] == ["scheduled", "scheduled"]
    assert cycle.requests == [
        CycleRequest(Pair.parse("BTC/USDT"), origin="scheduled"),
        CycleRequest(Pair.parse("ETH/USDT"), origin="scheduled"),
    ]
    assert scheduler.config_revision == 4
    assert scheduler.status["BTC/USDT"]["cycle_id"] == "revision-4-cycle"
    assert scheduler.status["ETH/USDT"]["cycle_id"] == "revision-4-cycle"


@pytest.mark.asyncio
async def test_scheduler_passes_pair_and_run_metadata_to_shared_runtime_cycle() -> None:
    cycle = _Cycle()
    runtime = _Runtime(cycle, ("BTC/USDT:USDT",))
    scheduler = Scheduler(_config(), runtime)

    await scheduler.run_once()

    assert len(cycle.requests) == 1
    assert [field.name for field in fields(cycle.requests[0])] == [
        "pair",
        "mode",
        "origin",
        "decision_id",
        "confirmed_book_ids",
    ]
    assert cycle.requests[0].mode == "trading"
    assert cycle.requests[0].origin == "scheduled"
    assert scheduler.status["BTC/USDT:USDT"]["last_action"] == "long"


@pytest.mark.asyncio
async def test_scheduler_rejects_reload_without_an_active_cycle() -> None:
    scheduler = Scheduler(_config(), _Runtime(None))

    await scheduler.run_once()
    assert scheduler.status["BTC/USDT"]["last_error"] == "cycle_failed"


@pytest.mark.asyncio
async def test_scheduler_does_not_close_runtime_it_does_not_own() -> None:
    runtime = _Runtime(_Cycle())
    scheduler = Scheduler(_config(), runtime)

    await scheduler.run_once()

    runtime.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_stop_pauses_new_fires_and_waits_for_inflight_batch() -> None:
    scheduler = Scheduler(_config(), _Runtime(_Cycle()))
    entered = asyncio.Event()
    terminal = asyncio.Event()

    async def blocking_batch():
        entered.set()
        await terminal.wait()

    scheduler.run_once = blocking_batch
    loop = asyncio.get_running_loop()
    with patch.object(loop, "add_signal_handler"):
        serving = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0)
        assert scheduler._stop_event is not None
        batch = asyncio.create_task(scheduler._run_cycle())
        await entered.wait()
        scheduler.stop()
        await asyncio.sleep(0)
        assert not serving.done()

        terminal.set()
        await batch
        await serving


@pytest.mark.asyncio
async def test_cancelled_scheduler_start_finally_drains_inflight_batch() -> None:
    scheduler = Scheduler(_config(), _Runtime(_Cycle()))
    entered = asyncio.Event()
    terminal = asyncio.Event()
    batch_cancelled = False

    async def blocking_batch():
        nonlocal batch_cancelled
        entered.set()
        try:
            await terminal.wait()
        except asyncio.CancelledError:
            batch_cancelled = True
            raise

    scheduler.run_once = blocking_batch
    loop = asyncio.get_running_loop()
    with patch.object(loop, "add_signal_handler"):
        serving = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0)
        batch = asyncio.create_task(scheduler._run_cycle())
        await entered.wait()
        serving.cancel()
        await asyncio.sleep(0)
        serving.cancel()
        assert serving.done() is False
        assert batch_cancelled is False

        terminal.set()
        await batch
        with pytest.raises(asyncio.CancelledError):
            await serving
        assert batch_cancelled is False


@pytest.mark.asyncio
async def test_scheduler_status_redacts_cycle_exception_and_keeps_trace_id() -> None:
    class FailingCycle(_Cycle):
        async def run(self, request: CycleRequest) -> CycleOutcome:
            raise RuntimeError("raw adapter account balance and secret")

    scheduler = Scheduler(_config(), _Runtime(FailingCycle()))

    await scheduler.run_once()

    status = scheduler.status["BTC/USDT"]
    assert status["last_error"] == "cycle_failed"
    assert status["trace_id"]
    assert "adapter" not in repr(status)


@pytest.mark.asyncio
async def test_scheduler_setup_failure_is_safe_and_traceable(caplog) -> None:
    runtime = _Runtime(_Cycle())
    scheduler = Scheduler(_config(), runtime)

    @asynccontextmanager
    async def unavailable_execution_lease(_pair: str):
        raise RuntimeError("raw redis credential")
        yield  # pragma: no cover - context-manager shape only

    runtime.execution_lease = unavailable_execution_lease

    await scheduler.run_once()

    status = scheduler.status["BTC/USDT"]
    assert status["last_error"] == "cycle_failed"
    assert status["trace_id"]
    assert "credential" not in repr(status)
    assert "credential" not in caplog.text
