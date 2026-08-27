from __future__ import annotations

from dataclasses import replace

import pytest

from cryptotrader.cycle_events import NullCycleEventSink
from cryptotrader.decision.engine import DecisionEngine
from cryptotrader.decision.exit_policy import AtrExitPolicy
from cryptotrader.execution.planner import ExecutionPlanner
from cryptotrader.execution.service import ExecutionResult
from cryptotrader.hitl.store import ApprovalStore
from cryptotrader.journal.store import CycleJournalStore
from cryptotrader.risk.models import RiskDecision
from cryptotrader.signals.fusion import WeightedSignalFusion
from cryptotrader.signals.models import CandleRequirement, DataRequirements
from cryptotrader.signals.registry import SignalComponentRegistry
from cryptotrader.trading_cycle import TradingCycle
from tests.factories.signal_fusion import context, profile, request, signal


class _Component:
    def __init__(self, component_id: str) -> None:
        self.id = component_id
        self.display_name = component_id
        self.description = component_id

    def requirements(self) -> DataRequirements:
        return DataRequirements(candles=(CandleRequirement("1h", 20),))

    async def evaluate(self, _context):
        raise AssertionError("the deterministic runner owns evaluation")


class _Profiles:
    def __init__(self, value) -> None:
        self.value = value

    async def get(self):
        return self.value


class _Contexts:
    async def collect(self, cycle_request, _requirements):
        return replace(context(), mode=cycle_request.mode)

    async def refresh_execution_state(self, stored_context):
        return stored_context


class _Runner:
    def __init__(self, signals) -> None:
        self.signals = signals

    async def run(self, _components, _context):
        return self.signals


class _Risk:
    async def check(self, risk_request, _portfolio):
        return RiskDecision(True, risk_request.plan)


class _Executor:
    async def execute(self, _plan, _context):
        return ExecutionResult(True, (), None, None)


def _cycle() -> TradingCycle:
    active_profile = profile(kronos=0.6, llm=0.4)
    signals = (
        signal("kronos", "long", 0.8),
        signal("llm_committee", "short", 0.3),
    )
    registry = SignalComponentRegistry((_Component("kronos"), _Component("llm_committee")))
    return TradingCycle(
        profiles=_Profiles(active_profile),
        registry=registry,
        contexts=_Contexts(),
        runner=_Runner(signals),
        fusion=WeightedSignalFusion(),
        decisions=DecisionEngine(),
        exits=AtrExitPolicy(),
        approvals=ApprovalStore(),
        risk=_Risk(),
        execution_planner=ExecutionPlanner(0.2),
        executor=_Executor(),
        journal=CycleJournalStore(),
        events=NullCycleEventSink(),
    )


@pytest.mark.asyncio
async def test_live_and_backtest_build_identical_trade_plan_from_same_inputs():
    live = _cycle()
    backtest = _cycle()

    live_outcome = await live.run(request(mode="paper"))
    backtest_outcome = await backtest.run(request(mode="backtest"))

    assert live_outcome.trade_plan == backtest_outcome.trade_plan
    assert live_outcome.fused_signal == backtest_outcome.fused_signal
