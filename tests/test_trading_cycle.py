"""TradingCycle 的周期级失败、取消与结果查找契约。"""

from __future__ import annotations

import asyncio

import pytest

from cryptotrader.decision.models import CycleRequest
from cryptotrader.signals.runner import ComponentRunError
from tests.test_multi_book_cycle import PAIR, _book, _cycle, _snapshot


class _FailingRunner:
    async def run(self, components, context):
        raise ComponentRunError({"fixture": RuntimeError("unsafe upstream detail")})


class _CancellingRunner:
    async def run(self, components, context):
        raise asyncio.CancelledError


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["manual", "scheduled", "trigger"])
async def test_cycle_persists_request_origin(origin):
    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    cycle, _, _, journal, _ = _cycle(_snapshot(simulation))

    outcome = await cycle.run(CycleRequest(PAIR, origin=origin))

    record = await journal.get(outcome.cycle_id)
    assert record.run.mode == "trading"
    assert record.run.origin == origin


@pytest.mark.asyncio
async def test_component_failure_saves_one_empty_non_executing_cycle():
    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    cycle, _, coordinator, journal, _ = _cycle(_snapshot(simulation))
    cycle.runner = _FailingRunner()
    cycle.analysis.runner = cycle.runner

    outcome = await cycle.run(CycleRequest(PAIR))

    assert outcome.status == "component_failed"
    assert outcome.books == ()
    assert outcome.execution_status == "not_started"
    assert coordinator.proposals == []
    assert len(journal.records) == 1
    assert journal.records[0].component_signals == ()


@pytest.mark.asyncio
async def test_cancellation_is_journaled_then_re_raised_without_execution():
    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    cycle, _, coordinator, journal, _ = _cycle(_snapshot(simulation))
    cycle.runner = _CancellingRunner()
    cycle.analysis.runner = cycle.runner

    with pytest.raises(asyncio.CancelledError):
        await cycle.run(CycleRequest(PAIR))

    assert coordinator.proposals == []
    assert len(journal.records) == 1
    assert journal.records[0].cycle_status == "cancelled"


@pytest.mark.asyncio
async def test_cycle_outcome_book_lookup_is_exact():
    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    cycle, *_ = _cycle(_snapshot(simulation))
    outcome = await cycle.run(CycleRequest(PAIR))

    assert outcome.book("simulation").book_id == "simulation"
    with pytest.raises(LookupError, match="missing"):
        outcome.book("missing")
