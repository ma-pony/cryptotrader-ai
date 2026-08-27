"""唯一 TradingCycle 主链的阶段与终止语义。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from cryptotrader.decision.engine import DecisionEngine
from cryptotrader.decision.exit_policy import AtrExitPolicy
from cryptotrader.decision.models import CycleRequest, TargetPosition
from cryptotrader.execution.planner import ExecutionPlanner
from cryptotrader.hitl.store import ApprovalStore
from cryptotrader.journal.store import CycleJournalStore
from cryptotrader.risk.models import RiskDecision
from cryptotrader.signals.fusion import WeightedSignalFusion
from cryptotrader.signals.models import CandleRequirement, DataRequirements
from cryptotrader.signals.registry import SignalComponentRegistry
from cryptotrader.signals.runner import ComponentRunError
from tests.factories.signal_fusion import context, position, profile, request, signal

if TYPE_CHECKING:
    from cryptotrader.cycle_events import CycleEvent
    from cryptotrader.profiles.models import SignalProfile


class _Component:
    def __init__(self, component_id: str) -> None:
        self.id = component_id
        self.display_name = component_id
        self.description = component_id

    def requirements(self) -> DataRequirements:
        return DataRequirements(candles=(CandleRequirement("1h", 20),))

    async def evaluate(self, signal_context):
        raise AssertionError("fake runner owns component evaluation")


class _Profiles:
    def __init__(self, value: SignalProfile) -> None:
        self.value = value
        self.calls = 0

    async def get(self) -> SignalProfile:
        self.calls += 1
        return self.value


class _Contexts:
    def __init__(self) -> None:
        self.current_position = position()
        self.collect_calls = 0
        self.refresh_calls = 0

    async def collect(self, cycle_request, requirements):
        self.collect_calls += 1
        return context(position=self.current_position)

    async def refresh_execution_state(self, stored_context):
        from dataclasses import replace

        self.refresh_calls += 1
        return replace(stored_context, current_position=self.current_position)


class _Runner:
    def __init__(self, signals, component_error=None, *, cancelled=False) -> None:
        self.signals = signals
        self.component_error = component_error
        self.cancelled = cancelled
        self.calls = 0

    async def run(self, components, signal_context):
        self.calls += 1
        if self.cancelled:
            raise asyncio.CancelledError
        if self.component_error:
            raise ComponentRunError(self.component_error)
        return self.signals


class _Risk:
    def __init__(self, *, passed=True) -> None:
        self.passed = passed
        self.calls = []

    async def check(self, risk_request, portfolio):
        self.calls.append((risk_request, portfolio))
        return RiskDecision(
            passed=self.passed,
            plan=risk_request.plan,
            rejected_by="test" if not self.passed else "",
            reason="blocked" if not self.passed else "",
        )


class _Planner(ExecutionPlanner):
    def __init__(self) -> None:
        super().__init__(max_single_pct=0.2)
        self.last_context = None

    def plan(self, signal_context, trade_plan):
        self.last_context = signal_context
        return super().plan(signal_context, trade_plan)


class _Executor:
    def __init__(self, *, succeeds=True) -> None:
        self.succeeds = succeeds
        self.executed = []

    async def execute(self, execution_plan, signal_context):
        from cryptotrader.execution.service import ExecutionResult

        self.executed.append(execution_plan)
        return ExecutionResult(
            succeeded=self.succeeds,
            orders=(),
            algo_id=None,
            error=None if self.succeeds else "exchange failed",
        )


class _Events:
    def __init__(self) -> None:
        self.events: list[CycleEvent] = []

    async def publish(self, event: CycleEvent) -> None:
        self.events.append(event)


def build_test_cycle(
    *,
    signals=None,
    selected_profile=None,
    component_error=None,
    risk_passed=True,
    execution_succeeds=True,
    cancelled=False,
):
    from cryptotrader.trading_cycle import TradingCycle

    signals = signals or (
        signal("kronos", "long", 0.8),
        signal("llm_committee", "long", 0.6),
    )
    selected_profile = selected_profile or profile(kronos=0.6, llm=0.4)
    registry = SignalComponentRegistry((_Component("kronos"), _Component("llm_committee")))
    return TradingCycle(
        profiles=_Profiles(selected_profile),
        registry=registry,
        contexts=_Contexts(),
        runner=_Runner(signals, component_error, cancelled=cancelled),
        fusion=WeightedSignalFusion(),
        decisions=DecisionEngine(),
        exits=AtrExitPolicy(),
        approvals=ApprovalStore(),
        risk=_Risk(passed=risk_passed),
        execution_planner=_Planner(),
        executor=_Executor(succeeds=execution_succeeds),
        journal=CycleJournalStore(),
        events=_Events(),
    )


@pytest.mark.asyncio
async def test_cycle_runs_components_fusion_risk_execution_and_journal():
    cycle = build_test_cycle()

    outcome = await cycle.run(request())

    assert outcome.status == "completed"
    assert outcome.trade_plan is not None
    assert outcome.trade_plan.target.side == "long"
    assert cycle.executor.executed[0].intents
    assert cycle.journal.records[0].profile_revision == outcome.profile_revision
    assert cycle.journal.records[0].fused_signal["score"] == pytest.approx(0.72)


@pytest.mark.asyncio
async def test_component_failure_skips_fusion_and_execution_but_journals_cycle():
    cycle = build_test_cycle(component_error={"llm_committee": RuntimeError("timeout")})

    outcome = await cycle.run(request())

    assert outcome.status == "component_failed"
    assert cycle.executor.executed == []
    assert cycle.journal.records[0].component_error == {
        "llm_committee": "RuntimeError: timeout",
    }


@pytest.mark.asyncio
async def test_hitl_stores_target_plan_and_approval_replans_from_current_position():
    cycle = build_test_cycle(selected_profile=profile(hitl=True))
    pending = await cycle.run(request())
    assert pending.status == "awaiting_approval"

    cycle.contexts.current_position = position("long", 0.2, 0.2)
    approved = await cycle.resume_approved(pending.approval_id)

    assert approved.status == "completed"
    assert cycle.execution_planner.last_context.current_position.size_ratio == 0.2
    assert cycle.profiles.calls == 1
    assert cycle.runner.calls == 1
    assert len(cycle.journal.records) == 1
    assert cycle.journal.records[0].status == "completed"


@pytest.mark.asyncio
async def test_rejected_approval_never_reaches_risk_or_execution():
    cycle = build_test_cycle(selected_profile=profile(hitl=True))
    pending = await cycle.run(request())

    rejected = await cycle.reject_approval(pending.approval_id, decision_by="web")

    assert rejected.status == "approval_rejected"
    assert cycle.risk.calls == []
    assert cycle.executor.executed == []
    assert cycle.journal.records[0].status == "approval_rejected"


@pytest.mark.asyncio
async def test_risk_rejection_preserves_decision_and_places_no_orders():
    cycle = build_test_cycle(risk_passed=False)

    outcome = await cycle.run(request())

    assert outcome.status == "risk_rejected"
    assert outcome.risk_result.reason == "blocked"
    assert cycle.executor.executed == []


@pytest.mark.asyncio
async def test_risk_gate_exception_fails_closed_as_risk_rejection():
    cycle = build_test_cycle()

    async def broken_check(risk_request, portfolio):
        raise RuntimeError("redis unavailable")

    cycle.risk.check = broken_check
    outcome = await cycle.run(request())

    assert outcome.status == "risk_rejected"
    assert outcome.risk_result.rejected_by == "risk_gate"
    assert outcome.risk_result.reason == "RuntimeError: redis unavailable"
    assert cycle.executor.executed == []


@pytest.mark.asyncio
async def test_execution_failure_does_not_rewrite_passed_risk_decision():
    cycle = build_test_cycle(execution_succeeds=False)

    outcome = await cycle.run(request())

    assert outcome.status == "execution_failed"
    assert outcome.risk_result.passed is True
    assert outcome.execution_result.error == "exchange failed"


@pytest.mark.asyncio
async def test_no_change_finishes_before_hitl_and_risk():
    cycle = build_test_cycle(
        signals=(signal("kronos", "neutral", 0.8), signal("llm_committee", "neutral", 0.6)),
        selected_profile=profile(hitl=True),
    )

    outcome = await cycle.run(request())

    assert outcome.status == "no_change"
    assert await cycle.approvals.list_pending() == []
    assert cycle.risk.calls == []


@pytest.mark.asyncio
async def test_cancelled_component_stage_is_journaled_then_reraised():
    cycle = build_test_cycle(cancelled=True)

    with pytest.raises(asyncio.CancelledError):
        await cycle.run(CycleRequest(request().pair, "paper"))

    assert cycle.journal.records[0].status == "cancelled"
    assert all(event.name != "fusion_completed" for event in cycle.events.events)


def test_target_position_comparison_uses_side_and_ratio():
    from cryptotrader.trading_cycle import target_matches_position

    assert target_matches_position(TargetPosition("flat", 0.0), position())
    assert target_matches_position(TargetPosition("long", 0.2), position("long", 1.0, 0.2))
    assert not target_matches_position(TargetPosition("short", 0.2), position("long", 1.0, 0.2))
