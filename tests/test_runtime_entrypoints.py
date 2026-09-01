"""数据库配置 Runtime 是所有生产入口共享的唯一装配边界。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cryptotrader.runtime_config.models import RuntimeConfigSnapshot, SchedulerConfig, TriggerConfig
from tests.factories.runtime_config import active_document, runtime_document
from tests.runtime_lease import static_cycle_lease


class _Repository:
    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    async def get_or_create(self):
        self.calls += 1
        return self.snapshot


class _Registry:
    def __init__(self, registered=()) -> None:
        self._registered = frozenset(registered)

    def registered_ids(self):
        return self._registered


@pytest.mark.asyncio
async def test_setup_required_runtime_opens_no_venue_sessions_and_has_no_cycle():
    runtime_module = import_module("cryptotrader.runtime")
    snapshot = RuntimeConfigSnapshot(0, runtime_document(), datetime(2026, 8, 29, tzinfo=UTC))
    repository = _Repository(snapshot)

    runtime = await runtime_module.build_runtime(
        repository=repository,
        snapshot=snapshot,
        signal_registry=_Registry({"kronos", "llm_committee"}),
        venue_registry=_Registry({"paper", "okx", "bybit"}),
        market_registry=_Registry({"default"}),
    )

    assert repository.calls == 0
    assert runtime.snapshot is snapshot
    assert runtime.repository is repository
    assert runtime.cycle is None
    assert runtime.sessions == {}
    await runtime.close()
    await runtime.close()


@pytest.mark.asyncio
async def test_api_trigger_callback_reloads_and_runs_platform_independent_cycle() -> None:
    from api import main
    from cryptotrader.decision.models import CycleOutcome, CycleRequest
    from cryptotrader.pair import Pair

    class Cycle:
        def __init__(self) -> None:
            self.requests = []

        async def run(self, request):
            self.requests.append(request)
            return CycleOutcome("trigger-cycle", 9, None, (), "no_change", "not_started", False)

    class Engine:
        instance = None

        def __init__(self, store, redis_state, callback, config) -> None:
            self.callback = callback
            self.start = AsyncMock()
            Engine.instance = self

    cycle = Cycle()

    async def automatic(pair, source):
        return await cycle.run(CycleRequest(Pair.parse(pair), origin=source))

    document = active_document(
        triggers=TriggerConfig(enabled=True),
        scheduler=SchedulerConfig(automation_enabled=True, enabled=True),
    )
    runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(9, document, datetime(2026, 8, 29, tzinfo=UTC)),
        repository=SimpleNamespace(database_url="sqlite+aiosqlite://"),
        cycle=cycle,
        cycle_lease=static_cycle_lease(cycle),
        execution_lease=lambda _pair: static_cycle_lease(cycle)(),
        run_service=SimpleNamespace(run_automatic=automatic),
    )
    application = SimpleNamespace(state=SimpleNamespace(runtime=runtime))

    with (
        patch("cryptotrader.triggers.engine.PriceTriggerEngine", Engine),
        patch("cryptotrader.triggers.store.TriggerRuleStore.ensure_tables", AsyncMock()),
    ):
        await main._init_trigger_engine(application)

    assert Engine.instance is not None
    await Engine.instance.callback("BTC/USDT", {"trigger_event_id": "event-1"})

    assert cycle.requests == [CycleRequest(Pair.parse("BTC/USDT"), origin="trigger")]


@pytest.mark.asyncio
async def test_scheduler_and_api_trigger_wait_for_same_book_execution_lease(tmp_path) -> None:  # noqa: C901
    """Both entrypoints retain application ownership while the cycle serializes its book."""
    from api import main
    from cryptotrader.cycle_events import MultiplexedCycleEventSink, NullCycleEventSink
    from cryptotrader.decision.models import CycleOutcome, CycleRequest
    from cryptotrader.execution_ownership import ExecutionOwnership
    from cryptotrader.migrations.workbench import migrate_workbench_schema
    from cryptotrader.pair import Pair
    from cryptotrader.runtime import Runtime
    from cryptotrader.scheduler import Scheduler

    class Cycle:
        def __init__(self) -> None:
            self.requests: list[CycleRequest] = []
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def run(self, request: CycleRequest) -> CycleOutcome:
            async with ExecutionOwnership("redis://strict-test").book("simulation"):
                self.requests.append(request)
                self.entered.set()
                await self.release.wait()
                return CycleOutcome("shared-cycle", 9, None, (), "no_change", "not_started", False)

    class Engine:
        instance = None

        def __init__(self, _store, _redis_state, callback, _config) -> None:
            self.callback = callback
            self.start = AsyncMock()
            Engine.instance = self

    class StrictRedisState:
        def __init__(self) -> None:
            self.values: dict[str, str] = {}
            self.close_count = 0
            self.contended = asyncio.Event()

        async def try_acquire_strict_lock(self, key: str, owner: str, _ttl: int) -> bool:
            if key in self.values:
                self.contended.set()
                return False
            self.values[key] = owner
            return True

        async def release_strict_lock(self, key: str, owner: str) -> bool:
            if self.values.get(key) != owner:
                return False
            del self.values[key]
            return True

        async def aclose(self) -> None:
            self.close_count += 1

    class Repository:
        async def get_or_create(self):
            return snapshot

    cycle = Cycle()
    document = active_document(
        triggers=TriggerConfig(enabled=True),
        scheduler=SchedulerConfig(automation_enabled=True, enabled=True, interval_minutes=15),
    )
    snapshot = RuntimeConfigSnapshot(9, document, datetime(2026, 8, 29, tzinfo=UTC))
    document = document.model_copy(
        update={"infrastructure": document.infrastructure.model_copy(update={"redis_url": "redis://strict-test"})}
    )
    snapshot = RuntimeConfigSnapshot(9, document, datetime(2026, 8, 29, tzinfo=UTC))
    cycle.snapshot = snapshot
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'runtime-entrypoints.db'}"
    await migrate_workbench_schema(database_url)
    repository = Repository()
    repository.database_url = database_url
    runtime = Runtime(
        snapshot=snapshot,
        repository=repository,
        cycle=cycle,
        sessions={},
        signal_registry=object(),
        market_registry=object(),
        venue_registry=object(),
        events=MultiplexedCycleEventSink(NullCycleEventSink()),
    )
    runtime._reload_for_cycle_locked = AsyncMock(return_value=cycle)
    runtime.run_service._scope = AsyncMock(
        return_value=SimpleNamespace(
            pair="BTC/USDT:USDT", ready=True, books=(SimpleNamespace(book_id="simulation", eligible=True),)
        )
    )
    strict_redis = StrictRedisState()
    application = SimpleNamespace(state=SimpleNamespace(runtime=runtime))
    with (
        patch("cryptotrader.triggers.engine.PriceTriggerEngine", Engine),
        patch("cryptotrader.triggers.store.TriggerRuleStore.ensure_tables", AsyncMock()),
        patch("cryptotrader.cycle_lock.RedisStateManager", return_value=strict_redis),
    ):
        await main._init_trigger_engine(application)

        trigger_task = asyncio.create_task(Engine.instance.callback("BTC/USDT:USDT", {}))
        try:
            await asyncio.wait_for(cycle.entered.wait(), timeout=1)
            scheduler = Scheduler(document.scheduler, runtime)
            scheduler_task = asyncio.create_task(scheduler._run_pair("BTC/USDT:USDT"))
            await asyncio.wait_for(strict_redis.contended.wait(), timeout=1)
            assert len(cycle.requests) == 1
        finally:
            cycle.release.set()
            await asyncio.wait_for(trigger_task, timeout=1)
            await asyncio.wait_for(scheduler_task, timeout=1)
            await runtime.task_manager.drain()

    assert len(cycle.requests) == 2
    assert cycle.requests[0].pair == Pair.parse("BTC/USDT:USDT")
    assert cycle.requests[0].origin == "trigger"
    assert cycle.requests[0].confirmed_book_ids == ("simulation",)
    assert cycle.requests[1].origin == "scheduled"
    assert scheduler.status["BTC/USDT:USDT"]["last_error"] is None
    assert strict_redis.values == {}
    assert strict_redis.close_count == 2


@pytest.mark.asyncio
async def test_api_scheduler_receives_fixed_config_and_runtime_owner() -> None:
    from api import main

    document = active_document(
        scheduler=SchedulerConfig(automation_enabled=True, enabled=True, interval_minutes=15),
    )
    runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(11, document, datetime(2026, 8, 29, tzinfo=UTC)),
    )
    application = SimpleNamespace(state=SimpleNamespace(runtime=runtime))
    scheduler = SimpleNamespace(start=AsyncMock())
    scheduler_type = patch("cryptotrader.scheduler.Scheduler", return_value=scheduler)

    class PendingTask:
        def done(self):
            return False

    def discard_task(coroutine, **_kwargs):
        coroutine.close()
        return PendingTask()

    with scheduler_type as constructor, patch("asyncio.create_task", side_effect=discard_task):
        await main._init_scheduler(application)

    constructor.assert_called_once_with(document.scheduler, runtime)
