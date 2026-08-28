"""Legacy single-position planning and new execution-book proposal planning."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import TYPE_CHECKING

from cryptotrader.decision.models import ExecutionPlan
from cryptotrader.decision.models import OrderIntent as LegacyOrderIntent
from cryptotrader.execution.models import BookExecutionProposal, ConnectionExecutionPlan
from cryptotrader.risk.models import ConnectionRiskDecision, ConnectionRiskRequest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cryptotrader.decision.models import TradePlan
    from cryptotrader.pair import Pair
    from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot
    from cryptotrader.risk.gate import BookRiskGate, ConnectionRiskGate
    from cryptotrader.risk.models import BookRiskDecision, BookRiskRequest
    from cryptotrader.signals.models import SignalContext
    from cryptotrader.venues.protocol import VenueSession


class ExecutionPlanningError(ValueError):
    pass


@dataclass(frozen=True)
class _PreflightResult:
    decision: ConnectionRiskDecision
    plan: ConnectionExecutionPlan | None
    unavailable: bool
    error: str


class ExecutionPlanner:
    """Build plans from normalized session contracts without adapter conditionals."""

    def __init__(
        self,
        max_single_pct: float | None = None,
        *,
        book_risk_gate: BookRiskGate | None = None,
        connection_risk_gate: ConnectionRiskGate | None = None,
    ) -> None:
        self.max_single_pct = max_single_pct
        self._book_risk_gate = book_risk_gate
        self._connection_risk_gate = connection_risk_gate

    def plan(self, context: SignalContext, trade_plan: TradePlan) -> ExecutionPlan:
        """Staged legacy entry retained only for callers migrated atomically in Task 11."""
        if self.max_single_pct is None:
            raise ExecutionPlanningError("legacy planning requires max_single_pct")
        target = trade_plan.target
        if context.market_type == "spot" and target.side == "short":
            raise ExecutionPlanningError("spot markets do not support short target positions")

        target_amount = context.equity * self.max_single_pct * target.size_ratio / context.current_price
        current_signed = context.current_position.signed_amount
        target_signed = {
            "long": target_amount,
            "short": -target_amount,
            "flat": 0.0,
        }[target.side]

        if current_signed * target_signed < 0.0:
            close_intent = LegacyOrderIntent(
                pair=context.pair.canonical(),
                side="sell" if current_signed > 0.0 else "buy",
                amount=abs(current_signed),
                reduce_only=True,
            )
            enter_intent = LegacyOrderIntent(
                pair=context.pair.canonical(),
                side="buy" if target_signed > 0.0 else "sell",
                amount=abs(target_signed),
                reduce_only=False,
            )
            return ExecutionPlan(
                intents=(close_intent, enter_intent),
                stop_loss=trade_plan.stop_loss,
                take_profit=trade_plan.take_profit,
            )

        delta = target_signed - current_signed
        if abs(delta) < 1e-12:
            intents: tuple[LegacyOrderIntent, ...] = ()
        else:
            intents = (
                LegacyOrderIntent(
                    pair=context.pair.canonical(),
                    side="buy" if delta > 0.0 else "sell",
                    amount=abs(delta),
                    reduce_only=abs(target_signed) < abs(current_signed),
                ),
            )
        return ExecutionPlan(
            intents=intents,
            stop_loss=trade_plan.stop_loss,
            take_profit=trade_plan.take_profit,
        )

    async def propose(
        self,
        request: BookRiskRequest,
        sessions: Mapping[str, VenueSession],
        *,
        pair: Pair,
        stop_loss: Decimal | None,
        take_profit: Decimal | None,
        config_revision: int,
    ) -> BookExecutionProposal:
        """Preflight every enabled target and return no hidden partial increase plan."""
        if self._book_risk_gate is None or self._connection_risk_gate is None:
            raise ExecutionPlanningError("book proposal planning requires both risk gates")
        self._validate_proposal_inputs(request, sessions, pair, stop_loss, take_profit, config_revision)
        book_risk = self._book_risk_gate.evaluate(request)
        if not book_risk.passed:
            return self._proposal(request, pair, config_revision, book_risk, (), (), (), (), False)

        portfolios = {item.connection_id: item for item in request.portfolio.connections}
        results = await asyncio.gather(
            *(
                self._preflight(
                    target,
                    portfolios[target.connection_id],
                    sessions.get(target.connection_id),
                    pair,
                    stop_loss,
                    take_profit,
                )
                for target in book_risk.connection_targets
            )
        )
        connection_risks = tuple(result.decision for result in results)
        unavailable = tuple(result.decision.connection_id for result in results if result.unavailable)
        errors = tuple(result.error for result in results if result.error)
        requires_all = any(result.decision.risk_increase for result in results)
        failed = any(not result.decision.passed or result.unavailable for result in results)
        if requires_all and failed:
            rejected_risk = replace(
                book_risk,
                passed=False,
                rejected_by="connection_preflight",
                reason="all enabled connections must pass before risk increase",
            )
            return self._proposal(
                request,
                pair,
                config_revision,
                rejected_risk,
                connection_risks,
                (),
                unavailable,
                errors,
                False,
            )

        plans = tuple(result.plan for result in results if result.plan is not None and result.decision.passed)
        required_deltas = any(
            target.target_signed_notional != portfolios[target.connection_id].position.signed_notional
            for target in book_risk.connection_targets
        )
        ready = bool(plans) or not required_deltas
        return self._proposal(
            request,
            pair,
            config_revision,
            book_risk,
            connection_risks,
            plans,
            unavailable,
            errors,
            ready,
        )

    async def _preflight(
        self,
        target,
        portfolio: ConnectionPortfolioSnapshot,
        session: VenueSession | None,
        pair: Pair,
        stop_loss: Decimal | None,
        take_profit: Decimal | None,
    ) -> _PreflightResult:
        if session is None or getattr(session, "connection_id", None) != target.connection_id:
            return self._unavailable(target, portfolio)
        try:
            quote_result, state_result = await asyncio.gather(
                session.fetch_quote(pair),
                session.list_open_state(pair),
                return_exceptions=True,
            )
            if isinstance(quote_result, BaseException) or isinstance(state_result, BaseException):
                return self._unavailable(target, portfolio)
            request = ConnectionRiskRequest(
                target,
                portfolio,
                True,
                quote_result,
                state_result,
                session.capabilities,
            )
            decision = self._connection_risk_gate.evaluate(request)
            if not decision.passed:
                return _PreflightResult(
                    decision,
                    None,
                    False,
                    f"connection {target.connection_id}: {decision.reason}",
                )
            delta = target.target_signed_notional - portfolio.position.signed_notional
            if delta == 0:
                return _PreflightResult(decision, None, False, "")
            execution_price = quote_result.ask if delta > 0 else quote_result.bid
            base_amount = abs(delta) / execution_price
            amount = await session.normalize_amount(pair, base_amount)
            reduce_only = self._reduces_position(portfolio.position.signed_notional, target.target_signed_notional)
            target_is_flat = target.target_signed_notional == 0
            protection_ids = tuple(
                dict.fromkeys(
                    protection_id
                    for protection in state_result.protections
                    if protection.active
                    for protection_id in protection.protection_ids
                )
            )
            plan = ConnectionExecutionPlan(
                book_id=target.book_id,
                connection_id=target.connection_id,
                pair=pair,
                current_signed_notional=portfolio.position.signed_notional,
                target_signed_notional=target.target_signed_notional,
                delta_signed_notional=delta,
                quote=quote_result,
                execution_price=execution_price,
                amount=amount,
                side="buy" if delta > 0 else "sell",
                reduce_only=reduce_only,
                market_type=pair.market_type,
                stop_loss=None if target_is_flat else stop_loss,
                take_profit=None if target_is_flat else take_profit,
                old_protection_ids=protection_ids,
                capabilities=session.capabilities,
            )
            return _PreflightResult(decision, plan, False, "")
        except Exception:
            return self._unavailable(target, portfolio)

    @staticmethod
    def _unavailable(target, portfolio: ConnectionPortfolioSnapshot) -> _PreflightResult:
        current = portfolio.position.signed_notional
        desired = target.target_signed_notional
        increase = desired != 0 and (current == 0 or current * desired < 0 or abs(desired) > abs(current))
        decision = ConnectionRiskDecision(target.connection_id, False, increase, "connection unavailable")
        return _PreflightResult(
            decision,
            None,
            True,
            f"connection {target.connection_id}: preflight unavailable",
        )

    @staticmethod
    def _reduces_position(current: Decimal, target: Decimal) -> bool:
        return target == 0 or (current * target > 0 and abs(target) < abs(current))

    @staticmethod
    def _validate_proposal_inputs(request, sessions, pair, stop_loss, take_profit, config_revision) -> None:
        from collections.abc import Mapping

        from cryptotrader.pair import Pair
        from cryptotrader.risk.models import BookRiskRequest

        if not isinstance(request, BookRiskRequest):
            raise ExecutionPlanningError("request must be a BookRiskRequest")
        if not isinstance(sessions, Mapping):
            raise ExecutionPlanningError("sessions must be a mapping")
        if not isinstance(pair, Pair):
            raise ExecutionPlanningError("pair must be a Pair")
        if any(item.position.pair != pair for item in request.portfolio.connections):
            raise ExecutionPlanningError("portfolio positions must match the proposal pair")
        for field_name, value in (("stop_loss", stop_loss), ("take_profit", take_profit)):
            if value is not None and (not isinstance(value, Decimal) or not value.is_finite() or value <= 0):
                raise ExecutionPlanningError(f"{field_name} must be a positive finite Decimal or None")
        if type(config_revision) is not int or config_revision < 0:
            raise ExecutionPlanningError("config_revision must be a non-negative integer")

    @staticmethod
    def _proposal(
        request,
        pair,
        config_revision,
        risk: BookRiskDecision,
        connection_risks,
        connection_plans,
        unavailable_connections,
        errors,
        ready,
    ) -> BookExecutionProposal:
        return BookExecutionProposal(
            book_id=request.book.id,
            capital_scope=request.book.capital_scope,
            config_revision=config_revision,
            pair=pair,
            requested_target_exposure=risk.requested_target_exposure,
            target_exposure=risk.capped_target_exposure,
            risk=risk,
            connection_risks=connection_risks,
            connection_plans=connection_plans,
            unavailable_connections=unavailable_connections,
            errors=errors,
            ready=ready,
        )
