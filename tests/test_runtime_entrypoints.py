"""数据库配置 Runtime 是所有生产入口共享的唯一装配边界。"""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cryptotrader.runtime_config.models import RuntimeConfigSnapshot, SchedulerConfig, TriggerConfig
from tests.factories.runtime_config import active_document, runtime_document


class _Repository:
    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    async def get_or_create(self):
        self.calls += 1
        return self.snapshot


class _Registry:
    def __init__(self, installed=()) -> None:
        self._installed = frozenset(installed)

    def installed_ids(self):
        return self._installed


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
    document = active_document(
        triggers=TriggerConfig(enabled=True),
        scheduler=SchedulerConfig(enabled=True, pairs=("BTC/USDT",)),
    )
    runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(9, document, datetime(2026, 8, 29, tzinfo=UTC)),
        repository=SimpleNamespace(database_url="sqlite+aiosqlite://"),
        cycle=cycle,
        reload_for_cycle=AsyncMock(return_value=cycle),
    )
    application = SimpleNamespace(state=SimpleNamespace(runtime=runtime))

    with (
        patch("cryptotrader.triggers.engine.PriceTriggerEngine", Engine),
        patch("cryptotrader.triggers.store.TriggerRuleStore.ensure_tables", AsyncMock()),
    ):
        await main._init_trigger_engine(application)

    assert Engine.instance is not None
    await Engine.instance.callback("BTC/USDT", {"trigger_event_id": "event-1"})

    runtime.reload_for_cycle.assert_awaited_once_with()
    assert cycle.requests == [CycleRequest(Pair.parse("BTC/USDT"))]


@pytest.mark.asyncio
async def test_api_scheduler_receives_fixed_config_and_runtime_owner() -> None:
    from api import main

    document = active_document(
        scheduler=SchedulerConfig(enabled=True, pairs=("BTC/USDT",), interval_minutes=15),
    )
    runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(11, document, datetime(2026, 8, 29, tzinfo=UTC)),
    )
    application = SimpleNamespace(state=SimpleNamespace(runtime=runtime))
    scheduler = SimpleNamespace(start=AsyncMock())
    scheduler_type = patch("cryptotrader.scheduler.Scheduler", return_value=scheduler)

    def discard_task(coroutine, **_kwargs):
        coroutine.close()
        return object()

    with scheduler_type as constructor, patch("asyncio.create_task", side_effect=discard_task):
        await main._init_scheduler(application)

    constructor.assert_called_once_with(document.scheduler, runtime)
