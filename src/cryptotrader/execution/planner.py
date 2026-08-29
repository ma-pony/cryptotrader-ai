"""基于执行资金池和标准化 venue 的预检提案。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import TYPE_CHECKING

from cryptotrader.execution.models import BookExecutionProposal, ConnectionExecutionPlan
from cryptotrader.risk.models import ConnectionRiskDecision, ConnectionRiskRequest
from cryptotrader.venues.models import OpenVenueState, ProtectionSpec, VenueCapabilities, VenueQuote
from cryptotrader.venues.protocol import VenueOperationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cryptotrader.pair import Pair
    from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot
    from cryptotrader.risk.gate import BookRiskGate, ConnectionRiskGate
    from cryptotrader.risk.models import BookRiskDecision, BookRiskRequest
    from cryptotrader.venues.protocol import VenueSession


class ExecutionPlanningError(ValueError):
    pass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreflightResult:
    decision: ConnectionRiskDecision
    plan: ConnectionExecutionPlan | None
    unavailable: bool
    error: str


@dataclass(frozen=True)
class _PreflightInputs:
    quote: VenueQuote
    state: OpenVenueState
    capabilities: VenueCapabilities


class ExecutionPlanner:
    """Build plans from normalized session contracts without adapter conditionals."""

    def __init__(
        self,
        *,
        book_risk_gate: BookRiskGate | None = None,
        connection_risk_gate: ConnectionRiskGate | None = None,
    ) -> None:
        self._book_risk_gate = book_risk_gate
        self._connection_risk_gate = connection_risk_gate

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
        failed = any(not result.decision.passed for result in results)
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
        inputs = await self._read_preflight_inputs(target, portfolio, session, pair)
        if isinstance(inputs, _PreflightResult):
            return inputs
        decision = self._evaluate_connection_risk(target, portfolio, inputs)
        if isinstance(decision, _PreflightResult):
            return decision

        current_notional = portfolio.position.signed_notional
        target_notional = target.target_signed_notional
        if target_notional == current_notional:
            return _PreflightResult(decision, None, False, "")

        side = self._order_side(current_notional, target_notional)
        execution_price = inputs.quote.ask if side == "buy" else inputs.quote.bid
        current_amount = portfolio.position.signed_amount
        target_amount = Decimal("0") if target_notional == 0 else target_notional / execution_price
        delta_amount = target_amount - current_amount
        if not self._valid_protection(
            pair,
            target_amount,
            execution_price,
            inputs.capabilities,
            stop_loss,
            take_profit,
        ):
            return self._failure(target, portfolio, "validate_protection", unavailable=False)

        try:
            amount = await session.normalize_amount(pair, abs(delta_amount))
        except asyncio.CancelledError:
            raise
        except VenueOperationError:
            return self._failure(target, portfolio, "normalize_amount", unavailable=True)

        signed_fill = amount if side == "buy" else -amount
        post_fill_amount = current_amount + signed_fill
        reduce_only = self._reduces_position(current_amount, target_amount)
        target_is_flat = target_amount == 0
        plan = self._build_connection_plan(
            target=target,
            portfolio=portfolio,
            pair=pair,
            inputs=inputs,
            execution_price=execution_price,
            amount=amount,
            current_amount=current_amount,
            target_amount=target_amount,
            delta_amount=delta_amount,
            post_fill_amount=post_fill_amount,
            side=side,
            reduce_only=reduce_only,
            stop_loss=None if target_is_flat else stop_loss,
            take_profit=None if target_is_flat else take_profit,
        )
        return _PreflightResult(decision, plan, False, "")

    async def _read_preflight_inputs(
        self,
        target,
        portfolio: ConnectionPortfolioSnapshot,
        session: VenueSession | None,
        pair: Pair,
    ) -> _PreflightInputs | _PreflightResult:
        if session is None:
            return self._failure(target, portfolio, "session", unavailable=True)
        if getattr(session, "connection_id", None) != target.connection_id:
            raise ExecutionPlanningError("session connection_id must match the connection target")
        quote, state = await asyncio.gather(
            session.fetch_quote(pair),
            session.list_open_state(pair),
            return_exceptions=True,
        )
        if isinstance(quote, asyncio.CancelledError):
            raise quote
        if isinstance(state, asyncio.CancelledError):
            raise state
        if isinstance(quote, BaseException) and not isinstance(quote, VenueOperationError):
            raise quote
        if isinstance(state, BaseException) and not isinstance(state, VenueOperationError):
            raise state
        if isinstance(quote, VenueOperationError):
            return self._failure(target, portfolio, "fetch_quote", unavailable=True)
        if isinstance(state, VenueOperationError):
            return self._failure(target, portfolio, "list_open_state", unavailable=True)
        return _PreflightInputs(quote, state, session.capabilities)

    def _evaluate_connection_risk(
        self,
        target,
        portfolio: ConnectionPortfolioSnapshot,
        inputs: _PreflightInputs,
    ) -> ConnectionRiskDecision | _PreflightResult:
        request = ConnectionRiskRequest(
            target,
            portfolio,
            True,
            inputs.quote,
            inputs.state,
            inputs.capabilities,
        )
        decision = self._connection_risk_gate.evaluate(request)
        return decision if decision.passed else self._rejected(decision)

    @staticmethod
    def _build_connection_plan(
        *,
        target,
        portfolio: ConnectionPortfolioSnapshot,
        pair: Pair,
        inputs: _PreflightInputs,
        execution_price: Decimal,
        amount: Decimal,
        current_amount: Decimal,
        target_amount: Decimal,
        delta_amount: Decimal,
        post_fill_amount: Decimal,
        side: str,
        reduce_only: bool,
        stop_loss: Decimal | None,
        take_profit: Decimal | None,
    ) -> ConnectionExecutionPlan:
        protection_ids = tuple(
            dict.fromkeys(
                protection_id
                for protection in inputs.state.protections
                if protection.active and protection.pair == pair
                for protection_id in protection.protection_ids
            )
        )
        return ConnectionExecutionPlan(
            book_id=target.book_id,
            connection_id=target.connection_id,
            pair=pair,
            current_signed_notional=portfolio.position.signed_notional,
            target_signed_notional=target.target_signed_notional,
            delta_signed_notional=target.target_signed_notional - portfolio.position.signed_notional,
            current_signed_amount=current_amount,
            target_signed_amount=target_amount,
            delta_signed_amount=delta_amount,
            post_fill_signed_amount=post_fill_amount,
            quote=inputs.quote,
            execution_price=execution_price,
            amount=amount,
            side=side,
            reduce_only=reduce_only,
            market_type=pair.market_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            old_protection_ids=protection_ids,
            capabilities=inputs.capabilities,
        )

    @staticmethod
    def _failure(
        target,
        portfolio: ConnectionPortfolioSnapshot,
        operation: str,
        *,
        unavailable: bool,
    ) -> _PreflightResult:
        current = portfolio.position.signed_notional
        desired = target.target_signed_notional
        increase = desired != 0 and (current == 0 or current * desired < 0 or abs(desired) > abs(current))
        logger.warning(
            "connection preflight failed",
            extra={"connection_id": target.connection_id, "operation": operation},
        )
        decision = ConnectionRiskDecision(
            target.connection_id,
            False,
            increase,
            f"{operation} failed",
            operation,
        )
        return _PreflightResult(
            decision,
            None,
            unavailable,
            f"connection {target.connection_id}: {operation} failed",
        )

    @staticmethod
    def _rejected(decision: ConnectionRiskDecision) -> _PreflightResult:
        logger.warning(
            "connection preflight rejected",
            extra={"connection_id": decision.connection_id, "operation": decision.operation},
        )
        return _PreflightResult(
            decision,
            None,
            False,
            f"connection {decision.connection_id}: {decision.operation} failed",
        )

    @staticmethod
    def _order_side(current_notional: Decimal, target_notional: Decimal) -> str:
        if target_notional == 0:
            return "sell" if current_notional > 0 else "buy"
        if current_notional == 0 or current_notional * target_notional < 0:
            return "buy" if target_notional > 0 else "sell"
        if abs(target_notional) > abs(current_notional):
            return "buy" if target_notional > 0 else "sell"
        return "sell" if current_notional > 0 else "buy"

    @staticmethod
    def _valid_protection(
        pair: Pair,
        target_amount: Decimal,
        execution_price: Decimal,
        capabilities,
        stop_loss: Decimal | None,
        take_profit: Decimal | None,
    ) -> bool:
        if target_amount == 0:
            return True
        has_protection = stop_loss is not None or take_profit is not None
        if pair.market_type != "spot" and not has_protection:
            return False
        if not has_protection:
            return True
        if not capabilities.native_protection:
            return False
        spec = ProtectionSpec(
            pair,
            "long" if target_amount > 0 else "short",
            abs(target_amount),
            stop_loss,
            take_profit,
        )
        return spec.has_valid_geometry(execution_price)

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
