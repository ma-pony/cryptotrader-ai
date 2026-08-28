"""组件、融合、目标仓位、审批、风控与执行的唯一顶层主链。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from cryptotrader.cycle_events import CycleEvent
from cryptotrader.cycle_serialization import (
    component_signal_payload,
    fused_signal_payload,
    signal_context_payload,
    signal_profile_payload,
    target_payload,
    trade_plan_payload,
)
from cryptotrader.decision.models import CycleOutcome
from cryptotrader.execution.planner import ExecutionPlanningError
from cryptotrader.execution.service import ExecutionResult
from cryptotrader.hitl.gate import requires_approval
from cryptotrader.journal.models import TradingCycleRecord
from cryptotrader.profiles.models import validate_signal_profile
from cryptotrader.risk.models import RiskDecision, RiskRequest
from cryptotrader.signals.models import DataRequirements
from cryptotrader.signals.runner import ComponentRunError

if TYPE_CHECKING:
    from cryptotrader.decision.models import CycleRequest, CycleStatus, TargetPosition, TradePlan
    from cryptotrader.signals.fusion import FusedSignal
    from cryptotrader.signals.models import ComponentSignal, PositionSnapshot, SignalContext, TradingMode


def target_matches_position(target: TargetPosition, position: PositionSnapshot) -> bool:
    return target.side == position.side and abs(target.size_ratio - position.size_ratio) < 1e-9


def _risk_payload(result: RiskDecision | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "passed": result.passed,
        "rejected_by": result.rejected_by,
        "reason": result.reason,
        "cap_source": result.cap_source,
        "target": target_payload(result.plan.target),
    }


def _execution_payload(result: ExecutionResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "succeeded": result.succeeded,
        "algo_id": result.algo_id,
        "error": result.error,
        "retained_algo_ids": list(result.retained_algo_ids),
        "protection_trigger": (
            {
                "algo_id": result.protection_trigger.algo_id,
                "trigger_reason": result.protection_trigger.trigger_reason,
                "trigger_price": result.protection_trigger.trigger_price,
                "order_id": result.protection_trigger.order_id,
            }
            if result.protection_trigger is not None
            else None
        ),
        "orders": [
            {
                "intent": {
                    "pair": item.intent.pair,
                    "side": item.intent.side,
                    "amount": item.intent.amount,
                    "reduce_only": item.intent.reduce_only,
                },
                "status": item.status,
                "exchange_id": item.exchange_id,
                "raw": dict(item.raw),
                "filled_amount": item.filled_amount,
            }
            for item in result.orders
        ],
    }


class TradingCycle:
    def __init__(
        self,
        *,
        mode: TradingMode,
        profiles,
        registry,
        contexts,
        runner,
        fusion,
        decisions,
        exits,
        approvals,
        risk,
        execution_planner,
        executor,
        journal,
        events,
        exit_requirement: DataRequirements | None = None,
    ) -> None:
        self.mode = mode
        self.profiles = profiles
        self.registry = registry
        self.contexts = contexts
        self.runner = runner
        self.fusion = fusion
        self.decisions = decisions
        self.exits = exits
        self.approvals = approvals
        self.risk = risk
        self.execution_planner = execution_planner
        self.executor = executor
        self.journal = journal
        self.events = events
        self.exit_requirement = exit_requirement or DataRequirements()

    async def run(self, request: CycleRequest) -> CycleOutcome:
        self._require_mode(request.mode)
        cycle_id = str(uuid4())
        created_at = datetime.now(UTC)
        await self.events.publish(
            CycleEvent(
                "cycle_started",
                {"cycle_id": cycle_id, "pair": request.pair.canonical(), "mode": request.mode},
            )
        )
        profile = await self.profiles.get()
        if profile is None:
            raise RuntimeError("global signal profile is not initialized")
        validate_signal_profile(profile, self.registry.ids())
        components = self.registry.enabled(profile)
        requirements = DataRequirements.merge(
            *(component.requirements() for component in components),
            self.exit_requirement,
        )
        context = None
        approval_id = None
        paper_execution_result = None
        try:
            context = await self.contexts.collect(request, requirements)
            protection_processor = getattr(self.executor, "process_pending_protection", None)
            if self.mode == "paper" and protection_processor is not None:
                protection_trigger = await protection_processor(context)
                if protection_trigger is not None:
                    paper_execution_result = ExecutionResult(
                        succeeded=True,
                        orders=(),
                        algo_id=None,
                        error=None,
                        protection_trigger=protection_trigger,
                    )
                    await self.events.publish(
                        CycleEvent(
                            "paper_protection_triggered",
                            {
                                "cycle_id": cycle_id,
                                "execution_result": _execution_payload(paper_execution_result),
                            },
                        )
                    )
                    try:
                        context = await self.contexts.refresh_execution_state(context)
                    except Exception as error:
                        reason = f"execution state refresh failed: {type(error).__name__}: {error}"
                        failed_result = replace(paper_execution_result, succeeded=False, error=reason)
                        return await self._finish(
                            cycle_id=cycle_id,
                            created_at=created_at,
                            status="execution_failed",
                            profile=profile,
                            context=context,
                            request=request,
                            execution_result=failed_result,
                            error=reason,
                        )
            await self.events.publish(
                CycleEvent(
                    "context_ready",
                    {"cycle_id": cycle_id, "context": signal_context_payload(context)},
                )
            )
            signals = await self.runner.run(components, context)
            fused = self.fusion.fuse(signals, profile.components)
            await self.events.publish(
                CycleEvent(
                    "fusion_completed",
                    {"cycle_id": cycle_id, "fusion": fused_signal_payload(fused)},
                )
            )
            target = self.decisions.target_for(fused, profile)
            plan = self.exits.build_plan(context, target, signals, fused, profile)
            await self.events.publish(
                CycleEvent(
                    "decision_created",
                    {"cycle_id": cycle_id, "trade_plan": trade_plan_payload(plan)},
                )
            )
            if target_matches_position(plan.target, context.current_position):
                return await self._finish(
                    cycle_id=cycle_id,
                    created_at=created_at,
                    status="no_change",
                    profile=profile,
                    context=context,
                    signals=signals,
                    fused=fused,
                    plan=plan,
                    execution_result=paper_execution_result,
                )
            if requires_approval(profile, request.mode):
                approval_id = str(uuid4())
                approval = await self.approvals.create(
                    cycle_id=cycle_id,
                    cycle_request=request,
                    profile=profile,
                    signal_context=context,
                    plan=plan,
                    approval_id=approval_id,
                    created_at=created_at,
                )
                await self.events.publish(
                    CycleEvent(
                        "approval_required",
                        {
                            "cycle_id": cycle_id,
                            "approval_id": approval.approval_id,
                            "trade_plan": trade_plan_payload(plan),
                        },
                    )
                )
                return await self._finish(
                    cycle_id=cycle_id,
                    created_at=created_at,
                    status="awaiting_approval",
                    profile=profile,
                    context=context,
                    signals=signals,
                    fused=fused,
                    plan=plan,
                    hitl_result={"approval_id": approval.approval_id, "status": "pending"},
                    execution_result=paper_execution_result,
                    approval_id=approval.approval_id,
                )
            return await self._risk_plan_execute(
                cycle_id=cycle_id,
                created_at=created_at,
                profile=profile,
                context=context,
                signals=signals,
                fused=fused,
                plan=plan,
                prior_execution_result=paper_execution_result,
            )
        except asyncio.CancelledError:
            if approval_id is not None:
                await self.approvals.cancel_pending(approval_id)
            existing = await self.journal.get(cycle_id)
            await self._finish(
                cycle_id=cycle_id,
                created_at=created_at,
                status="cancelled",
                profile=profile,
                context=context,
                request=request,
                execution_result=paper_execution_result,
                replace_journal=existing is not None,
                error="cycle cancelled",
            )
            raise
        except ComponentRunError as error:
            errors = {component_id: f"{type(cause).__name__}: {cause}" for component_id, cause in error.errors.items()}
            return await self._finish(
                cycle_id=cycle_id,
                created_at=created_at,
                status="component_failed",
                profile=profile,
                context=context,
                request=request,
                component_error=errors,
                execution_result=paper_execution_result,
                error=str(error),
            )

    async def resume_approved(self, approval_id: str, *, decision_by: str = "web") -> CycleOutcome:
        pending = await self.approvals.get(approval_id)
        if pending is None:
            raise LookupError(f"approval {approval_id!r} does not exist")
        self._require_mode(pending.cycle_request.mode)
        approval = await self.approvals.approve(approval_id, decision_by=decision_by)
        context = None
        try:
            try:
                context = await self.contexts.refresh_execution_state(approval.signal_context)
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                risk_result = RiskDecision(
                    passed=False,
                    plan=approval.plan,
                    rejected_by="execution_state_refresh",
                    reason=reason,
                )
                return await self._finish(
                    cycle_id=approval.cycle_id,
                    created_at=approval.created_at,
                    status="risk_rejected",
                    profile=approval.profile,
                    context=approval.signal_context,
                    signals=approval.plan.component_signals,
                    fused=approval.plan.fused_signal,
                    plan=approval.plan,
                    hitl_result={
                        "approval_id": approval.approval_id,
                        "status": "approved",
                        "decision_by": approval.decision_by,
                        "refresh_status": "failed",
                        "error": reason,
                    },
                    risk_result=risk_result,
                    replace_journal=True,
                    error=reason,
                )
            return await self._risk_plan_execute(
                cycle_id=approval.cycle_id,
                created_at=approval.created_at,
                profile=approval.profile,
                context=context,
                signals=approval.plan.component_signals,
                fused=approval.plan.fused_signal,
                plan=approval.plan,
                hitl_result={
                    "approval_id": approval.approval_id,
                    "status": "approved",
                    "decision_by": approval.decision_by,
                },
                replace_journal=True,
            )
        except asyncio.CancelledError:
            await self._finish(
                cycle_id=approval.cycle_id,
                created_at=approval.created_at,
                status="cancelled",
                profile=approval.profile,
                context=context or approval.signal_context,
                replace_journal=True,
                error="cycle cancelled",
            )
            raise

    async def reject_approval(self, approval_id: str, *, decision_by: str = "web") -> CycleOutcome:
        pending = await self.approvals.get(approval_id)
        if pending is None:
            raise LookupError(f"approval {approval_id!r} does not exist")
        self._require_mode(pending.cycle_request.mode)
        approval = await self.approvals.reject(approval_id, decision_by=decision_by)
        return await self._finish(
            cycle_id=approval.cycle_id,
            created_at=approval.created_at,
            status="approval_rejected",
            profile=approval.profile,
            context=approval.signal_context,
            signals=approval.plan.component_signals,
            fused=approval.plan.fused_signal,
            plan=approval.plan,
            hitl_result={
                "approval_id": approval.approval_id,
                "status": "rejected",
                "decision_by": approval.decision_by,
            },
            approval_id=approval.approval_id,
            replace_journal=True,
        )

    def _require_mode(self, requested_mode: TradingMode) -> None:
        if requested_mode != self.mode:
            raise RuntimeError(
                f"trading cycle mode {self.mode!r} cannot handle {requested_mode!r} request",
            )

    async def _risk_plan_execute(
        self,
        *,
        cycle_id: str,
        created_at: datetime,
        profile,
        context: SignalContext,
        signals: tuple[ComponentSignal, ...],
        fused: FusedSignal,
        plan: TradePlan,
        hitl_result: dict[str, Any] | None = None,
        replace_journal: bool = False,
        prior_execution_result: ExecutionResult | None = None,
    ) -> CycleOutcome:
        try:
            risk_result = await self.risk.check(RiskRequest(context=context, plan=plan), dict(context.portfolio))
        except Exception as error:
            risk_result = RiskDecision(
                passed=False,
                plan=plan,
                rejected_by="risk_gate",
                reason=f"{type(error).__name__}: {error}",
            )
        await self.events.publish(
            CycleEvent(
                "risk_checked",
                {"cycle_id": cycle_id, "risk_result": _risk_payload(risk_result)},
            )
        )
        if not risk_result.passed:
            return await self._finish(
                cycle_id=cycle_id,
                created_at=created_at,
                status="risk_rejected",
                profile=profile,
                context=context,
                signals=signals,
                fused=fused,
                plan=plan,
                hitl_result=hitl_result,
                risk_result=risk_result,
                execution_result=prior_execution_result,
                replace_journal=replace_journal,
                error=risk_result.reason,
            )

        final_plan = risk_result.plan
        try:
            execution_plan = self.execution_planner.plan(context, final_plan)
            execution_result = await self.executor.execute(execution_plan, context)
        except ExecutionPlanningError as error:
            execution_result = ExecutionResult(False, (), None, f"{type(error).__name__}: {error}")
        except Exception as error:
            execution_result = ExecutionResult(False, (), None, f"{type(error).__name__}: {error}")
        if prior_execution_result is not None:
            execution_result = replace(
                execution_result,
                protection_trigger=prior_execution_result.protection_trigger,
            )

        await self.events.publish(
            CycleEvent(
                "execution_completed",
                {"cycle_id": cycle_id, "execution_result": _execution_payload(execution_result)},
            )
        )
        status: CycleStatus = "completed" if execution_result.succeeded else "execution_failed"
        return await self._finish(
            cycle_id=cycle_id,
            created_at=created_at,
            status=status,
            profile=profile,
            context=context,
            signals=signals,
            fused=fused,
            plan=plan,
            hitl_result=hitl_result,
            risk_result=risk_result,
            execution_result=execution_result,
            replace_journal=replace_journal,
            error=execution_result.error,
        )

    async def _finish(
        self,
        *,
        cycle_id: str,
        created_at: datetime,
        status: CycleStatus,
        profile,
        context: SignalContext | None,
        request: CycleRequest | None = None,
        signals: tuple[ComponentSignal, ...] = (),
        fused: FusedSignal | None = None,
        plan: TradePlan | None = None,
        component_error: dict[str, str] | None = None,
        hitl_result: dict[str, Any] | None = None,
        risk_result: RiskDecision | None = None,
        execution_result: ExecutionResult | None = None,
        approval_id: str | None = None,
        replace_journal: bool = False,
        error: str | None = None,
    ) -> CycleOutcome:
        if context is None and request is None:
            raise ValueError("context or request is required to finish a trading cycle")
        context_summary = (
            signal_context_payload(context)
            if context is not None
            else {
                "pair": request.pair.canonical(),
                "as_of": request.as_of.isoformat() if request.as_of is not None else None,
                "mode": request.mode,
                "exchange_id": request.exchange_id,
            }
        )
        pair = context.pair.canonical() if context is not None else request.pair.canonical()
        record = TradingCycleRecord(
            cycle_id=cycle_id,
            created_at=created_at,
            pair=pair,
            status=status,
            profile_revision=profile.revision,
            profile_snapshot=signal_profile_payload(profile),
            context_summary=context_summary,
            component_signals=tuple(component_signal_payload(item) for item in signals),
            component_error=component_error,
            fused_signal=fused_signal_payload(fused) if fused is not None else None,
            target_position=target_payload(plan.target) if plan is not None else None,
            trade_plan=trade_plan_payload(plan) if plan is not None else None,
            hitl_result=hitl_result,
            risk_result=_risk_payload(risk_result),
            execution_result=_execution_payload(execution_result),
        )
        if replace_journal:
            await self.journal.replace(record)
        else:
            await self.journal.append(record)
        event_name = {
            "completed": "cycle_completed",
            "no_change": "cycle_completed",
            "awaiting_approval": "cycle_awaiting_approval",
            "cancelled": "cycle_cancelled",
        }.get(status, "cycle_failed")
        await self.events.publish(CycleEvent(event_name, {"cycle_id": cycle_id, "status": status, "error": error}))
        return CycleOutcome(
            cycle_id=cycle_id,
            status=status,
            profile_revision=profile.revision,
            component_signals=signals,
            fused_signal=fused,
            trade_plan=plan,
            risk_result=risk_result,
            execution_result=execution_result,
            approval_id=approval_id or (hitl_result or {}).get("approval_id"),
            error=error,
        )
