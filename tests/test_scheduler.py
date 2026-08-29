"""Scheduler 只通过 Runtime 和新 CycleRequest 驱动周期。"""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest

from cryptotrader.decision.models import CycleOutcome, CycleRequest, TargetPosition
from cryptotrader.pair import Pair
from cryptotrader.scheduler import Scheduler
from tests.factories.runtime_config import active_document


class _Cycle:
    def __init__(self) -> None:
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        return CycleOutcome("cycle-1", 4, TargetPosition("long", 0.5), (), "completed", "completed", False)


class _Runtime:
    def __init__(self, cycle=None) -> None:
        self.cycle = cycle
        self.snapshot = SimpleNamespace(document=active_document())
        self.repository = SimpleNamespace(database_url=None)
        self.closed = 0

    async def close(self):
        self.closed += 1


def test_scheduler_normalizes_pairs_without_execution_identity_fields():
    scheduler = Scheduler(["BTC/USDT:USDT"], runtime=_Runtime(_Cycle()))
    assert scheduler.pairs == [Pair.parse("BTC/USDT:USDT")]
    assert not hasattr(scheduler, "mode")
    assert not hasattr(scheduler, "exchange_id")


@pytest.mark.asyncio
async def test_scheduler_passes_only_pair_to_shared_runtime_cycle():
    cycle = _Cycle()
    scheduler = Scheduler(["BTC/USDT:USDT"], runtime=_Runtime(cycle))

    await scheduler._run_pair_locked("BTC/USDT:USDT")

    assert len(cycle.requests) == 1
    assert isinstance(cycle.requests[0], CycleRequest)
    assert [field.name for field in fields(cycle.requests[0])] == ["pair"]
    assert scheduler.status["BTC/USDT:USDT"]["last_action"] == "long"


@pytest.mark.asyncio
async def test_scheduler_requires_an_active_runtime_cycle():
    scheduler = Scheduler(["BTC/USDT"], runtime=_Runtime(None))
    with pytest.raises(RuntimeError, match="not active"):
        await scheduler._ensure_trading_cycle()


@pytest.mark.asyncio
async def test_scheduler_shutdown_closes_runtime_once_per_call_boundary():
    runtime = _Runtime(_Cycle())
    scheduler = Scheduler(["BTC/USDT"], runtime=runtime)
    await scheduler._close_live_exchanges()
    assert runtime.closed == 1
