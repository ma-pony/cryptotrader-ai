"""唯一 TradingCycle 主链的阶段与终止语义。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from cryptotrader.decision.engine import DecisionEngine
from cryptotrader.decision.exit_policy import AtrExitPolicy
from cryptotrader.decision.models import CycleRequest, TargetPosition
from cryptotrader.execution.planner import ExecutionPlanner
from cryptotrader.hitl.store import ApprovalStateError, ApprovalStore
from cryptotrader.journal.store import CycleJournalStore
from cryptotrader.risk.models import RiskDecision
from cryptotrader.signals.fusion import WeightedSignalFusion
from cryptotrader.signals.models import CandleRequirement, DataRequirements
from cryptotrader.signals.registry import SignalComponentRegistry
from cryptotrader.signals.runner import ComponentRunError
from tests.factories.signal_fusion import context, position, profile, request, signal, trade_plan

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
    def __init__(self, *, cancel_collect: bool = False) -> None:
        self.current_position = position()
        self.collect_calls = 0
        self.refresh_calls = 0
        self.cancel_collect = cancel_collect
        self.refresh_price = None
        self.refresh_equity = None

    async def collect(self, cycle_request, requirements):
        self.collect_calls += 1
        if self.cancel_collect:
            raise asyncio.CancelledError
        return context(position=self.current_position)

    async def refresh_execution_state(self, stored_context):
        self.refresh_calls += 1
        return replace(
            stored_context,
            current_price=self.refresh_price or stored_context.current_price,
            equity=self.refresh_equity or stored_context.equity,
            current_position=self.current_position,
        )


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
    def __init__(self, *, passed=True, adjusted_target=None) -> None:
        self.passed = passed
        self.adjusted_target = adjusted_target
        self.calls = []

    async def check(self, risk_request, portfolio):
        self.calls.append((risk_request, portfolio))
        adjusted_plan = (
            replace(risk_request.plan, target=self.adjusted_target)
            if self.adjusted_target is not None
            else risk_request.plan
        )
        return RiskDecision(
            passed=self.passed,
            plan=adjusted_plan,
            rejected_by="test" if not self.passed else "",
            reason="blocked" if not self.passed else ("capped by test" if self.adjusted_target else ""),
            cap_source="test_cap" if self.adjusted_target else "",
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
    cancel_context=False,
    adjusted_target=None,
    approval_store=None,
    journal_store=None,
):
    from cryptotrader.trading_cycle import TradingCycle

    signals = signals or (
        signal("kronos", "long", 0.8),
        signal("llm_committee", "long", 0.6),
    )
    selected_profile = selected_profile or profile(kronos=0.6, llm=0.4)
    registry = SignalComponentRegistry((_Component("kronos"), _Component("llm_committee")))
    return TradingCycle(
        mode="paper",
        profiles=_Profiles(selected_profile),
        registry=registry,
        contexts=_Contexts(cancel_collect=cancel_context),
        runner=_Runner(signals, component_error, cancelled=cancelled),
        fusion=WeightedSignalFusion(),
        decisions=DecisionEngine(),
        exits=AtrExitPolicy(),
        approvals=approval_store or ApprovalStore(),
        risk=_Risk(passed=risk_passed, adjusted_target=adjusted_target),
        execution_planner=_Planner(),
        executor=_Executor(succeeds=execution_succeeds),
        journal=journal_store or CycleJournalStore(),
        events=_Events(),
    )


@pytest.mark.asyncio
async def test_cycle_rejects_requests_for_a_different_execution_mode():
    cycle = build_test_cycle()

    with pytest.raises(RuntimeError, match="mode"):
        await cycle.run(request(mode="live"))

    assert cycle.executor.executed == []


@pytest.mark.asyncio
async def test_approval_mode_mismatch_is_rejected_before_claim_or_execution():
    cycle = build_test_cycle(selected_profile=profile(hitl=True))
    approval = await cycle.approvals.create(
        cycle_id="paper-cycle",
        cycle_request=request(mode="live"),
        profile=profile(),
        signal_context=context(),
        plan=trade_plan(TargetPosition("long", 0.4)),
        approval_id="live-approval",
    )

    with pytest.raises(RuntimeError, match="mode"):
        await cycle.resume_approved(approval.approval_id)

    assert (await cycle.approvals.get(approval.approval_id)).status == "pending"
    assert cycle.executor.executed == []


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
async def test_successful_cycle_emits_each_business_stage_in_order():
    cycle = build_test_cycle()

    await cycle.run(request())

    names = [event.name for event in cycle.events.events]
    expected = [
        "cycle_started",
        "context_ready",
        "fusion_completed",
        "decision_created",
        "risk_checked",
        "execution_completed",
        "cycle_completed",
    ]
    assert [name for name in names if name in expected] == expected


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
async def test_profile_failure_is_preflight_and_does_not_open_a_cycle():
    cycle = build_test_cycle()

    async def fail_profile_load():
        raise RuntimeError("profile store unavailable")

    cycle.profiles.get = fail_profile_load

    with pytest.raises(RuntimeError, match="profile store unavailable"):
        await cycle.run(request())

    assert cycle.events.events == []
    assert cycle.journal.records == []
    assert cycle.executor.executed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["context", "fusion", "decision", "exit"])
async def test_started_cycle_orchestration_failure_writes_one_generic_terminal(stage):
    cycle = build_test_cycle()

    if stage == "context":

        async def fail_context(cycle_request, requirements):
            raise RuntimeError("context unavailable")

        cycle.contexts.collect = fail_context
    elif stage == "fusion":
        cycle.fusion.fuse = lambda signals, weights: (_ for _ in ()).throw(RuntimeError("fusion unavailable"))
    elif stage == "decision":
        cycle.decisions.target_for = lambda fused, selected_profile: (_ for _ in ()).throw(
            RuntimeError("decision unavailable")
        )
    else:
        cycle.exits.build_plan = lambda *args: (_ for _ in ()).throw(RuntimeError("exit unavailable"))

    outcome = await cycle.run(request())

    assert outcome.status == "cycle_failed"
    assert outcome.error == f"RuntimeError: {stage} unavailable"
    assert cycle.executor.executed == []
    assert len(cycle.journal.records) == 1
    record = cycle.journal.records[0]
    assert record.status == "cycle_failed"
    assert record.error == f"RuntimeError: {stage} unavailable"
    if stage == "context":
        expected_context = {
            "available": False,
            "pair": "BTC/USDT:USDT",
            "as_of": None,
            "mode": "paper",
            "exchange_id": "okx",
        }
        assert record.context_summary == expected_context

        from api.routes.decisions import _detail

        assert _detail(record).model_dump()["context"] == expected_context
    assert (len(record.component_signals) > 0) is (stage != "context")
    assert (record.fused_signal is not None) is (stage in {"decision", "exit"})
    assert record.trade_plan is None
    event_names = [event.name for event in cycle.events.events]
    assert event_names.count("cycle_started") == 1
    assert event_names.count("cycle_failed") == 1
    assert "execution_completed" not in event_names


@pytest.mark.asyncio
async def test_started_cycle_failure_after_plan_preserves_plan_snapshot():
    cycle = build_test_cycle(selected_profile=profile(hitl=True))

    async def fail_approval_creation(**kwargs):
        raise RuntimeError("approval store unavailable")

    cycle.approvals.create = fail_approval_creation

    outcome = await cycle.run(request())

    assert outcome.status == "cycle_failed"
    assert outcome.trade_plan is not None
    assert cycle.executor.executed == []
    assert len(cycle.journal.records) == 1
    assert cycle.journal.records[0].trade_plan is not None
    assert cycle.journal.records[0].error == "RuntimeError: approval store unavailable"


@pytest.mark.asyncio
async def test_approval_cleanup_failure_cannot_hide_started_cycle_terminal():
    cycle = build_test_cycle(selected_profile=profile(hitl=True))

    async def fail_approval_creation(**kwargs):
        raise RuntimeError("approval store unavailable")

    async def fail_approval_cleanup(approval_id):
        raise RuntimeError("approval cleanup unavailable")

    cycle.approvals.create = fail_approval_creation
    cycle.approvals.cancel_pending = fail_approval_cleanup

    outcome = await cycle.run(request())

    assert outcome.status == "cycle_failed"
    assert outcome.error == (
        "RuntimeError: approval store unavailable; approval cleanup failed: RuntimeError: approval cleanup unavailable"
    )
    assert len(cycle.journal.records) == 1
    assert [event.name for event in cycle.events.events].count("cycle_failed") == 1


@pytest.mark.asyncio
async def test_paper_protection_runs_before_components_and_refreshes_position_on_component_failure():
    cycle = build_test_cycle()
    cycle.contexts.current_position = position("long", 2.0, 0.5)
    call_order = []
    seen_positions = []

    async def process_pending_protection(signal_context):
        call_order.append("protection")
        assert signal_context.current_position.side == "long"
        cycle.contexts.current_position = position()
        return SimpleNamespace(
            algo_id="paper-oco",
            trigger_reason="stop_loss",
            trigger_price=90.0,
            order_id="paper-close",
        )

    async def fail_components(components, signal_context):
        call_order.append("components")
        seen_positions.append(signal_context.current_position)
        raise ComponentRunError({"kronos": RuntimeError("component failed")})

    cycle.executor.process_pending_protection = process_pending_protection
    cycle.runner.run = fail_components

    outcome = await cycle.run(request())

    assert outcome.status == "component_failed"
    assert call_order == ["protection", "components"]
    assert cycle.contexts.refresh_calls == 1
    assert seen_positions == [position()]
    assert cycle.journal.records[0].execution_result["protection_trigger"] == {
        "algo_id": "paper-oco",
        "trigger_reason": "stop_loss",
        "trigger_price": 90.0,
        "order_id": "paper-close",
    }


@pytest.mark.asyncio
async def test_paper_protection_refresh_failure_writes_execution_failed_terminal_audit():
    cycle = build_test_cycle()
    cycle.contexts.current_position = position("long", 2.0, 0.5)

    async def process_pending_protection(signal_context):
        cycle.contexts.current_position = position()
        return SimpleNamespace(
            algo_id="paper-oco",
            trigger_reason="take_profit",
            trigger_price=120.0,
            order_id="paper-close",
        )

    async def fail_refresh(stored_context):
        cycle.contexts.refresh_calls += 1
        raise RuntimeError("portfolio refresh down")

    cycle.executor.process_pending_protection = process_pending_protection
    cycle.contexts.refresh_execution_state = fail_refresh

    outcome = await cycle.run(request())

    assert outcome.status == "execution_failed"
    assert outcome.execution_result.succeeded is False
    assert "portfolio refresh down" in outcome.execution_result.error
    assert cycle.runner.calls == 0
    assert len(cycle.journal.records) == 1
    record = cycle.journal.records[0]
    assert record.status == "execution_failed"
    assert record.execution_result["protection_trigger"] == {
        "algo_id": "paper-oco",
        "trigger_reason": "take_profit",
        "trigger_price": 120.0,
        "order_id": "paper-close",
    }


_HITL_PROTECTION_TRIGGER = {
    "algo_id": "paper-hitl-oco",
    "trigger_reason": "stop_loss",
    "trigger_price": 90.0,
    "order_id": "paper-hitl-close",
}


async def _run_to_hitl_after_paper_protection(cycle):
    cycle.contexts.current_position = position("long", 2.0, 0.5)

    async def process_pending_protection(signal_context):
        cycle.contexts.current_position = position()
        return SimpleNamespace(**_HITL_PROTECTION_TRIGGER)

    cycle.executor.process_pending_protection = process_pending_protection
    pending = await cycle.run(request())
    assert pending.status == "awaiting_approval"
    assert cycle.journal.records[0].execution_result["protection_trigger"] == _HITL_PROTECTION_TRIGGER
    return pending


def _assert_hitl_terminal_retains_protection_trigger(cycle, expected_status):
    record = cycle.journal.records[0]
    assert record.status == expected_status
    assert record.execution_result is not None
    assert record.execution_result["protection_trigger"] == _HITL_PROTECTION_TRIGGER


@pytest.mark.asyncio
async def test_hitl_approval_completion_retains_prior_paper_protection_trigger():
    cycle = build_test_cycle(selected_profile=profile(hitl=True))
    pending = await _run_to_hitl_after_paper_protection(cycle)

    outcome = await cycle.resume_approved(pending.approval_id)

    assert outcome.status == "completed"
    _assert_hitl_terminal_retains_protection_trigger(cycle, "completed")


@pytest.mark.asyncio
async def test_hitl_rejection_retains_prior_paper_protection_trigger():
    cycle = build_test_cycle(selected_profile=profile(hitl=True))
    pending = await _run_to_hitl_after_paper_protection(cycle)

    outcome = await cycle.reject_approval(pending.approval_id, decision_by="web")

    assert outcome.status == "approval_rejected"
    _assert_hitl_terminal_retains_protection_trigger(cycle, "approval_rejected")


@pytest.mark.asyncio
async def test_hitl_approval_refresh_failure_retains_prior_paper_protection_trigger():
    cycle = build_test_cycle(selected_profile=profile(hitl=True))
    pending = await _run_to_hitl_after_paper_protection(cycle)

    async def fail_refresh(stored_context):
        raise RuntimeError("approval refresh unavailable")

    cycle.contexts.refresh_execution_state = fail_refresh
    outcome = await cycle.resume_approved(pending.approval_id)

    assert outcome.status == "risk_rejected"
    _assert_hitl_terminal_retains_protection_trigger(cycle, "risk_rejected")


@pytest.mark.asyncio
async def test_hitl_approval_cancellation_retains_prior_paper_protection_trigger():
    cycle = build_test_cycle(selected_profile=profile(hitl=True))
    pending = await _run_to_hitl_after_paper_protection(cycle)

    async def cancel_execution(execution_plan, signal_context):
        raise asyncio.CancelledError

    cycle.executor.execute = cancel_execution
    with pytest.raises(asyncio.CancelledError):
        await cycle.resume_approved(pending.approval_id)

    _assert_hitl_terminal_retains_protection_trigger(cycle, "cancelled")


@pytest.mark.asyncio
async def test_hitl_stores_target_plan_and_approval_replans_from_current_position():
    cycle = build_test_cycle(selected_profile=profile(hitl=True))
    pending = await cycle.run(request())
    assert pending.status == "awaiting_approval"
    assert any(event.name == "approval_required" for event in cycle.events.events)

    cycle.contexts.current_position = position("long", 0.2, 0.2)
    approved = await cycle.resume_approved(pending.approval_id)

    assert approved.status == "completed"
    assert cycle.execution_planner.last_context.current_position.size_ratio == 0.2
    assert cycle.profiles.calls == 1
    assert cycle.runner.calls == 1
    assert len(cycle.journal.records) == 1
    assert cycle.journal.records[0].status == "completed"


@pytest.mark.asyncio
async def test_hitl_approval_refreshes_price_equity_and_position_without_recomputing_analysis():
    cycle = build_test_cycle(selected_profile=profile(hitl=True))
    pending = await cycle.run(request())
    frozen_plan = pending.trade_plan
    frozen_signals = pending.component_signals
    frozen_fusion = pending.fused_signal
    cycle.contexts.current_position = position("long", 0.2, 0.2)
    cycle.contexts.refresh_price = 125.0
    cycle.contexts.refresh_equity = 12_500.0

    approved = await cycle.resume_approved(pending.approval_id)

    assert approved.trade_plan == frozen_plan
    assert approved.component_signals == frozen_signals
    assert approved.fused_signal == frozen_fusion
    assert cycle.contexts.collect_calls == 1
    assert cycle.runner.calls == 1
    assert cycle.execution_planner.last_context.current_price == 125.0
    assert cycle.execution_planner.last_context.equity == 12_500.0
    assert cycle.execution_planner.last_context.current_position.size_ratio == 0.2


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
async def test_failed_execution_journals_and_emits_retained_protection_ids():
    from cryptotrader.execution.service import ExecutionResult

    cycle = build_test_cycle()

    async def fail_with_retained_protection(execution_plan, signal_context):
        return ExecutionResult(
            succeeded=False,
            orders=(),
            algo_id=None,
            error="replacement failed; original position restored",
            retained_algo_ids=("old-oco",),
        )

    cycle.executor.execute = fail_with_retained_protection

    outcome = await cycle.run(request())

    assert outcome.status == "execution_failed"
    assert cycle.journal.records[0].execution_result["retained_algo_ids"] == ["old-oco"]
    execution_event = next(event for event in cycle.events.events if event.name == "execution_completed")
    assert execution_event.data["execution_result"]["retained_algo_ids"] == ["old-oco"]


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


@pytest.mark.asyncio
async def test_context_collection_cancellation_writes_one_empty_terminal_record_and_reraises():
    cycle = build_test_cycle(cancel_context=True)

    with pytest.raises(asyncio.CancelledError):
        await cycle.run(request())

    assert len(cycle.journal.records) == 1
    record = cycle.journal.records[0]
    assert record.status == "cancelled"
    assert record.component_signals == ()
    assert record.fused_signal is None
    assert record.target_position is None
    assert record.trade_plan is None
    assert [event.name for event in cycle.events.events].count("cycle_cancelled") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_sqlite", [False, True])
async def test_post_commit_approval_creation_cancellation_is_irreversible(tmp_path, use_sqlite):
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'cancelled-approval.db'}" if use_sqlite else None

    class _CancelAfterCommitApprovalStore(ApprovalStore):
        def __init__(self):
            super().__init__(database_url)
            self.created_approval_id = None

        async def create(self, **kwargs):
            record = await super().create(**kwargs)
            self.created_approval_id = record.approval_id
            raise asyncio.CancelledError

    approvals = _CancelAfterCommitApprovalStore()
    journal = CycleJournalStore(database_url)
    cycle = build_test_cycle(
        selected_profile=profile(hitl=True),
        approval_store=approvals,
        journal_store=journal,
    )

    with pytest.raises(asyncio.CancelledError):
        await cycle.run(request())

    approval_id = approvals.created_approval_id
    assert approval_id is not None
    approval = await approvals.get(approval_id)
    assert approval is not None
    assert approval.status == "cancelled"
    assert await approvals.list_pending() == []
    with pytest.raises(ApprovalStateError, match="not pending"):
        await approvals.approve(approval_id, decision_by="web")
    with pytest.raises(ApprovalStateError, match="not pending"):
        await approvals.reject(approval_id, decision_by="web")
    with pytest.raises(ApprovalStateError, match="not pending"):
        await cycle.resume_approved(approval_id)
    records = await journal.list(limit=10)
    assert len(records) == 1
    assert records[0].status == "cancelled"
    assert records[0].component_signals == ()
    assert records[0].fused_signal is None
    assert records[0].target_position is None
    assert records[0].trade_plan is None
    assert [event.name for event in cycle.events.events].count("cycle_cancelled") == 1


@pytest.mark.asyncio
async def test_risk_adjustment_keeps_original_plan_in_journal_and_executes_adjusted_target():
    cycle = build_test_cycle(adjusted_target=TargetPosition("long", 0.25))

    outcome = await cycle.run(request())

    record = cycle.journal.records[0]
    assert outcome.trade_plan.target.size_ratio == pytest.approx(0.65)
    assert record.target_position == {"side": "long", "size_ratio": pytest.approx(0.65)}
    assert record.trade_plan["target"] == {"side": "long", "size_ratio": pytest.approx(0.65)}
    assert record.risk_result["target"] == {"side": "long", "size_ratio": 0.25}
    assert record.risk_result["cap_source"] == "test_cap"
    assert record.risk_result["reason"] == "capped by test"
    assert cycle.executor.executed[0].intents[0].amount == pytest.approx(5.0)


def test_target_position_comparison_uses_side_and_ratio():
    from cryptotrader.trading_cycle import target_matches_position

    assert target_matches_position(TargetPosition("flat", 0.0), position())
    assert target_matches_position(TargetPosition("long", 0.2), position("long", 1.0, 0.2))
    assert not target_matches_position(TargetPosition("short", 0.2), position("long", 1.0, 0.2))
