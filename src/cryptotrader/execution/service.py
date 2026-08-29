"""顺序执行目标仓位差值并维护唯一交易所保护单。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from cryptotrader.decision.models import OrderIntent
from cryptotrader.execution.models import (
    NO_COMPENSATION,
    CompensationResult,
    ConnectionExecutionPlan,
    ConnectionExecutionResult,
    ExecutionFinalPosition,
)
from cryptotrader.models import Order, OrderStatus
from cryptotrader.venues.models import (
    NormalizedOrder,
    OpenVenueState,
    ProtectionSpec,
    ProtectionState,
    VenueQuote,
)
from cryptotrader.venues.models import (
    OrderIntent as VenueOrderIntent,
)
from cryptotrader.venues.protocol import VenueOperationError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from cryptotrader.decision.models import ExecutionPlan
    from cryptotrader.execution.order import OrderManager
    from cryptotrader.signals.models import SignalContext
    from cryptotrader.venues.protocol import VenueSession


@dataclass(frozen=True)
class ExecutionOrderResult:
    intent: OrderIntent
    status: str
    venue_order_id: str | None
    raw: Mapping[str, Any] = field(default_factory=dict)
    filled_amount: float = 0.0


@dataclass(frozen=True)
class ProtectionTriggerResult:
    algo_id: str
    trigger_reason: str
    trigger_price: float
    order_id: str


@dataclass(frozen=True)
class ExecutionResult:
    succeeded: bool
    orders: tuple[ExecutionOrderResult, ...]
    algo_id: str | None
    error: str | None
    retained_algo_ids: tuple[str, ...] = ()
    protection_trigger: ProtectionTriggerResult | None = None


class ExecutionService:
    def __init__(
        self,
        orders: OrderManager,
        venue: Any,
        *,
        manage_protection: bool = True,
    ) -> None:
        self.orders = orders
        self.venue = venue
        self.manage_protection = manage_protection

    async def execute(self, plan: ExecutionPlan, context: SignalContext) -> ExecutionResult:
        results: list[ExecutionOrderResult] = []
        final_signed_amount = self._final_signed_amount(plan, context)

        if not self.manage_protection:
            return await self._execute_intents(plan.intents, context, results)

        pair = context.pair.canonical()
        prepared = await self._prepare_protection(plan, context.current_price, final_signed_amount, pair)
        if isinstance(prepared, ExecutionResult):
            return prepared
        old_algo_ids = prepared
        filled_intents, execution_failure = await self._execute_protected_intents(
            plan.intents,
            context.current_price,
            results,
            old_algo_ids,
        )
        if execution_failure is not None:
            return execution_failure
        if abs(final_signed_amount) < 1e-12:
            return await self._finish_flat(
                pair,
                old_algo_ids,
                filled_intents,
                context.current_price,
                results,
            )
        return await self._replace_protection(
            plan,
            pair,
            final_signed_amount,
            old_algo_ids,
            filled_intents,
            context.current_price,
            context.as_of,
            results,
        )

    async def process_pending_protection(self, context: SignalContext) -> ProtectionTriggerResult | None:
        processor = getattr(self.venue, "process_pending_protection", None)
        if processor is None:
            return None
        return await processor(context)

    @staticmethod
    def _final_signed_amount(plan: ExecutionPlan, context: SignalContext) -> float:
        final_signed_amount = context.current_position.signed_amount
        for intent in plan.intents:
            final_signed_amount += intent.amount if intent.side == "buy" else -intent.amount
        return final_signed_amount

    async def _prepare_protection(
        self,
        plan: ExecutionPlan,
        current_price: float,
        final_signed_amount: float,
        pair: str,
    ) -> tuple[str, ...] | ExecutionResult:
        supports_protection = self.venue.capabilities.native_protection
        if abs(final_signed_amount) >= 1e-12:
            if not supports_protection:
                return ExecutionResult(False, (), None, "protection orders are unsupported by this exchange")
            validation_error = self._validate_protection(plan, current_price, final_signed_amount)
            if validation_error is not None:
                return ExecutionResult(False, (), None, validation_error)
        if not supports_protection:
            return ()
        try:
            return await self._pending_algo_ids(pair)
        except Exception as error:
            return ExecutionResult(
                False,
                (),
                None,
                f"cannot inspect existing protection: {type(error).__name__}: {error}",
            )

    async def _execute_protected_intents(
        self,
        intents: tuple[OrderIntent, ...],
        price: float,
        results: list[ExecutionOrderResult],
        old_algo_ids: tuple[str, ...],
    ) -> tuple[list[OrderIntent], ExecutionResult | None]:
        filled_intents: list[OrderIntent] = []
        for intent in intents:
            try:
                placed = await self._place_intent(intent, price, results)
            except Exception as error:
                reason = f"order placement failed: {type(error).__name__}: {error}"
                if filled_intents:
                    reason = self._failure_with_compensation(
                        reason,
                        await self._compensate(filled_intents, price, results),
                    )
                return filled_intents, ExecutionResult(False, tuple(results), None, reason, old_algo_ids)

            actual_fill = results[-1].filled_amount
            if actual_fill > 0.0:
                filled_intents.append(
                    OrderIntent(
                        pair=intent.pair,
                        side=intent.side,
                        amount=actual_fill,
                        reduce_only=intent.reduce_only,
                    )
                )
            if placed.status == OrderStatus.FILLED and math.isclose(actual_fill, intent.amount, rel_tol=1e-9):
                continue
            raw = results[-1].raw
            reason = str(raw.get("reason") or raw.get("error_msg") or placed.status.value)
            if filled_intents:
                compensation_error = await self._compensate(filled_intents, price, results)
                reason = self._failure_with_compensation(
                    f"incomplete target-delta execution: {reason}",
                    compensation_error,
                )
            return filled_intents, ExecutionResult(False, tuple(results), None, reason, old_algo_ids)
        return filled_intents, None

    async def _finish_flat(
        self,
        pair: str,
        old_algo_ids: tuple[str, ...],
        filled_intents: list[OrderIntent],
        price: float,
        results: list[ExecutionOrderResult],
    ) -> ExecutionResult:
        retained_algo_ids, error = await self._cancel_algos(old_algo_ids, pair)
        if error is not None:
            compensation_error = await self._compensate(filled_intents, price, results)
            return ExecutionResult(
                False,
                tuple(results),
                None,
                self._failure_with_compensation(
                    f"cannot cancel existing protection: {type(error).__name__}: {error}",
                    compensation_error,
                ),
                retained_algo_ids,
            )
        return ExecutionResult(True, tuple(results), None, None)

    async def _replace_protection(
        self,
        plan: ExecutionPlan,
        pair: str,
        final_signed_amount: float,
        old_algo_ids: tuple[str, ...],
        filled_intents: list[OrderIntent],
        price: float,
        created_as_of: datetime,
        results: list[ExecutionOrderResult],
    ) -> ExecutionResult:
        pos_side = "long" if final_signed_amount > 0.0 else "short"
        try:
            algo_id = await self.venue.place_algo_oco(
                pair,
                side="sell" if pos_side == "long" else "buy",
                amount=abs(final_signed_amount),
                sl_trigger_px=float(plan.stop_loss),
                tp_trigger_px=float(plan.take_profit),
                pos_side=pos_side,
                created_as_of=created_as_of,
            )
        except Exception as error:
            compensation_error = await self._compensate(filled_intents, price, results)
            return ExecutionResult(
                False,
                tuple(results),
                None,
                self._failure_with_compensation(
                    f"cannot install replacement protection: {type(error).__name__}: {error}",
                    compensation_error,
                ),
                old_algo_ids,
            )
        return await self._retire_old_protection(
            algo_id,
            pair,
            old_algo_ids,
            filled_intents,
            price,
            results,
        )

    async def _retire_old_protection(
        self,
        algo_id: str,
        pair: str,
        old_algo_ids: tuple[str, ...],
        filled_intents: list[OrderIntent],
        price: float,
        results: list[ExecutionOrderResult],
    ) -> ExecutionResult:
        retained_old_ids, error = await self._cancel_algos(old_algo_ids, pair)
        if error is not None:
            cleanup_errors: list[str] = []
            retained_new_ids, cleanup_error = await self._cancel_algos((algo_id,), pair)
            if cleanup_error is not None:
                cleanup_errors.append(f"replacement cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}")
            compensation_error = await self._compensate(filled_intents, price, results)
            if compensation_error is not None:
                cleanup_errors.append(compensation_error)
            detail = f"cannot cancel existing protection: {type(error).__name__}: {error}"
            if cleanup_errors:
                detail = f"{detail}; {'; '.join(cleanup_errors)}"
            return ExecutionResult(
                False,
                tuple(results),
                None,
                detail,
                retained_old_ids + retained_new_ids,
            )
        return ExecutionResult(True, tuple(results), algo_id, None)

    async def _execute_intents(
        self,
        intents: tuple[OrderIntent, ...],
        context: SignalContext,
        results: list[ExecutionOrderResult],
    ) -> ExecutionResult:
        for intent in intents:
            placed = await self._place_intent(intent, context.current_price, results)
            if placed.status != OrderStatus.FILLED:
                raw = results[-1].raw
                reason = str(raw.get("reason") or raw.get("error_msg") or placed.status.value)
                return ExecutionResult(False, tuple(results), None, reason)
        return ExecutionResult(True, tuple(results), None, None)

    async def _place_intent(
        self,
        intent: OrderIntent,
        price: float,
        results: list[ExecutionOrderResult],
    ) -> Order:
        order = Order(
            pair=intent.pair,
            side=intent.side,
            amount=intent.amount,
            price=price,
            reduce_only=intent.reduce_only,
        )
        placed, raw = await self.orders.place(order, self.venue)
        filled_amount = self._filled_amount(intent, placed, raw)
        results.append(
            ExecutionOrderResult(
                intent=intent,
                status=placed.status.value,
                venue_order_id=placed.venue_order_id,
                raw=raw,
                filled_amount=filled_amount,
            )
        )
        return placed

    async def _compensate(
        self,
        filled_intents: list[OrderIntent],
        price: float,
        results: list[ExecutionOrderResult],
    ) -> str | None:
        for intent in reversed(filled_intents):
            compensation = OrderIntent(
                pair=intent.pair,
                side="sell" if intent.side == "buy" else "buy",
                amount=intent.amount,
                reduce_only=not intent.reduce_only,
            )
            try:
                placed = await self._place_intent(compensation, price, results)
            except Exception as error:
                return f"position compensation failed: {type(error).__name__}: {error}"
            actual_fill = results[-1].filled_amount
            if placed.status != OrderStatus.FILLED or not math.isclose(
                actual_fill,
                compensation.amount,
                rel_tol=1e-9,
            ):
                raw = results[-1].raw
                reason = str(raw.get("reason") or raw.get("error_msg") or placed.status.value)
                return f"position compensation failed: {reason}"
        return None

    @staticmethod
    def _filled_amount(intent: OrderIntent, placed: Order, raw: Mapping[str, Any]) -> float:
        reported = raw.get("filled")
        if reported is None:
            return intent.amount if placed.status == OrderStatus.FILLED else 0.0
        try:
            filled = float(reported)
        except (TypeError, ValueError):
            return 0.0
        return filled if math.isfinite(filled) and filled > 0.0 else 0.0

    @staticmethod
    def _failure_with_compensation(reason: str, compensation_error: str | None) -> str:
        if compensation_error is None:
            return f"{reason}; original position restored"
        return f"{reason}; {compensation_error}"

    @staticmethod
    def _validate_protection(plan: ExecutionPlan, current_price: float, final_signed_amount: float) -> str | None:
        try:
            stop_loss = float(plan.stop_loss)
            take_profit = float(plan.take_profit)
        except (TypeError, ValueError):
            return "invalid protection: non-flat execution requires stop loss and take profit"
        if not all(math.isfinite(value) and value > 0.0 for value in (current_price, stop_loss, take_profit)):
            return "invalid protection: current, stop-loss, and take-profit prices must be positive and finite"
        if final_signed_amount > 0.0 and not stop_loss < current_price < take_profit:
            return "invalid protection: long exits must satisfy stop-loss < current price < take-profit"
        if final_signed_amount < 0.0 and not take_profit < current_price < stop_loss:
            return "invalid protection: short exits must satisfy take-profit < current price < stop-loss"
        return None

    async def _pending_algo_ids(self, pair: str) -> tuple[str, ...]:
        pending = await self.venue.list_pending_algos(pair=pair)
        return tuple(str(algo_id) for item in pending for algo_id in (item.get("algoId") or item.get("id"),) if algo_id)

    async def _cancel_algos(
        self,
        algo_ids: tuple[str, ...],
        pair: str,
    ) -> tuple[tuple[str, ...], Exception | None]:
        for index, algo_id in enumerate(algo_ids):
            try:
                await self.venue.cancel_algo(algo_id, pair)
            except Exception as error:
                return algo_ids[index:], error
        return (), None


# The venue-bound execution path is staged independently until Task 11.  It
# deliberately does not call, wrap, or fall back to ExecutionService above.


@dataclass(frozen=True)
class _FreshTransition:
    initial_state: OpenVenueState
    quote: VenueQuote
    target_amount: Decimal
    delta_amount: Decimal
    side: str
    normalized_amount: Decimal
    expected_amount: Decimal
    risk_increase: bool
    sign_flip: bool


class VenueExecutionService:
    """Execute one immutable plan against one bound normalized venue session."""

    def __init__(self, session: VenueSession) -> None:
        self.session = session
        self.connection_id = session.connection_id

    async def execute(self, plan: ConnectionExecutionPlan) -> ConnectionExecutionResult:
        if not isinstance(plan, ConnectionExecutionPlan):
            raise TypeError("plan must be a ConnectionExecutionPlan")
        if plan.connection_id != self.connection_id:
            raise ValueError("plan connection_id must match the bound venue session")
        if self.session.capabilities != plan.capabilities:
            raise ValueError("session capabilities changed after proposal creation")

        trace: list[str] = []
        try:
            initial = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            return self._failed(plan, "pre_read", trace=("pre_read",))
        self._validate_fresh_state(plan, initial)
        if plan.pair.market_type == "spot" and self._active_protection_ids(initial):
            return self._failed(
                plan,
                "precondition",
                requires_attention=True,
                trace=("pre_read", "precondition"),
            )
        try:
            quote = await self.session.fetch_quote(plan.pair)
        except VenueOperationError:
            return self._failed(plan, "fetch_quote", trace=("pre_read", "fetch_quote"))
        trace.append("pre_read")
        self._validate_fresh_inputs(plan, initial, quote)

        transition = await self._fresh_transition(plan, initial, quote, trace)
        if isinstance(transition, ConnectionExecutionResult):
            return transition
        if transition.sign_flip:
            return await self._execute_flip(plan, transition, trace)
        return await self._execute_direct(plan, transition, trace)

    async def _fresh_transition(
        self,
        plan: ConnectionExecutionPlan,
        initial: OpenVenueState,
        quote: VenueQuote,
        trace: list[str],
    ) -> _FreshTransition | ConnectionExecutionResult:
        current = initial.position.signed_amount
        target, delta, side = self._target_transition(plan, current, quote)
        if (
            plan.pair.market_type != "spot"
            and target != 0
            and (plan.stop_loss is not None or plan.take_profit is not None)
        ):
            protection = ProtectionSpec(
                plan.pair,
                "long" if target > 0 else "short",
                abs(target),
                plan.stop_loss,
                plan.take_profit,
            )
            protection.validate_geometry(quote.ask if side == "buy" else quote.bid)
        sign_flip = current != 0 and target != 0 and current * target < 0
        risk_increase = target != 0 and (current == 0 or sign_flip or abs(target) > abs(current))
        if delta == 0:
            normalized = Decimal("0")
            expected = current
        else:
            try:
                normalized = await self.session.normalize_amount(plan.pair, abs(delta))
            except VenueOperationError:
                return self._failed(
                    plan,
                    "normalize_amount",
                    trace=tuple(trace),
                    target_amount=target,
                    execution_quote=quote,
                )
            self._validate_normalized_amount(normalized, abs(delta))
            expected = current + (normalized if side == "buy" else -normalized)
            if not sign_flip and not self._notional_within_spread(
                expected,
                plan.target_signed_notional,
                quote,
            ):
                final = self._summarize(initial)
                return self._failed(
                    plan,
                    "incomplete_fill",
                    final_position=final,
                    requires_attention=(
                        plan.pair.market_type != "spot" and initial.position.signed_amount != 0 and not final.protected
                    ),
                    trace=tuple(trace),
                    target_amount=target,
                    execution_quote=quote,
                )
        return _FreshTransition(initial, quote, target, delta, side, normalized, expected, risk_increase, sign_flip)

    async def _execute_direct(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        trace: list[str],
    ) -> ConnectionExecutionResult:
        orders: list[NormalizedOrder] = []
        if transition.normalized_amount != 0:
            intent = VenueOrderIntent(
                plan.pair,
                transition.side,
                transition.normalized_amount,
                "market",
                None,
                plan.pair.market_type != "spot"
                and self._is_reduction(transition.initial_state.position.signed_amount, transition.target_amount),
            )
            trace.append("place_order")
            try:
                order = await self.session.place_order(intent)
            except VenueOperationError:
                return await self._failed_after_order_error(plan, transition, orders, trace, "place_order")
            self._validate_order(order, intent)
            orders.append(order)
            if not self._fully_filled(order):
                return await self._handle_incomplete_order(plan, transition, orders, trace, order)

        if transition.target_amount == 0:
            return await self._finish_flat(plan, transition, orders, trace)
        if plan.pair.market_type == "spot":
            return await self._finish_spot_nonflat(plan, transition, orders, trace)
        return await self._finish_nonflat(plan, transition, orders, trace)

    async def _execute_flip(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        trace: list[str],
    ) -> ConnectionExecutionResult:
        orders: list[NormalizedOrder] = []
        current = transition.initial_state.position.signed_amount
        try:
            close_amount = await self.session.normalize_amount(plan.pair, abs(current))
        except VenueOperationError:
            return self._failed(
                plan,
                "normalize_amount",
                trace=tuple(trace),
                target_amount=transition.target_amount,
                execution_quote=transition.quote,
            )
        self._validate_normalized_amount(close_amount, abs(current), exact=True)
        close_intent = VenueOrderIntent(
            plan.pair,
            "sell" if current > 0 else "buy",
            close_amount,
            "market",
            None,
            True,
        )
        trace.append("close_old_side")
        try:
            close_order = await self.session.place_order(close_intent)
        except VenueOperationError:
            return await self._failed_after_flip_close(
                plan,
                transition,
                orders,
                trace,
                "close_old_side",
            )
        self._validate_order(close_order, close_intent)
        orders.append(close_order)
        if not self._fully_filled(close_order):
            return await self._failed_after_flip_close(
                plan,
                transition,
                orders,
                trace,
                "incomplete_fill",
            )

        trace.append("reconcile_flat")
        try:
            flat_state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            return self._failed(
                plan,
                "reconcile_flat",
                orders=tuple(orders),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=transition.target_amount,
                execution_quote=transition.quote,
            )
        if flat_state.position.signed_amount != 0:
            return self._failed(
                plan,
                "position_mismatch",
                orders=tuple(orders),
                final_position=self._summarize(flat_state),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=transition.target_amount,
                execution_quote=transition.quote,
            )

        old_ids = self._active_protection_ids(transition.initial_state)
        trace.append("cancel_old_protection")
        try:
            await self.session.cancel_protection(old_ids)
        except VenueOperationError:
            final = self._summarize(flat_state)
            return self._failed(
                plan,
                "cancel_old_protection",
                orders=tuple(orders),
                final_position=final,
                requires_attention=False,
                trace=tuple(trace),
                target_amount=transition.target_amount,
                execution_quote=transition.quote,
            )

        open_side = "buy" if transition.target_amount > 0 else "sell"
        prepared_open = await self._prepare_flip_open_amount(plan, transition, orders, trace, open_side)
        if isinstance(prepared_open, ConnectionExecutionResult):
            return prepared_open
        open_amount = prepared_open
        open_intent = VenueOrderIntent(plan.pair, open_side, open_amount, "market", None, False)
        trace.append("open_target_side")
        try:
            open_order = await self.session.place_order(open_intent)
        except VenueOperationError:
            return await self._failed_after_flip_open_error(plan, transition, orders, trace)
        self._validate_order(open_order, open_intent)
        orders.append(open_order)
        expected = open_order.filled_amount if open_side == "buy" else -open_order.filled_amount
        flip_transition = _FreshTransition(
            transition.initial_state,
            transition.quote,
            transition.target_amount,
            transition.target_amount,
            open_side,
            open_amount,
            expected,
            True,
            True,
        )
        if not self._fully_filled(open_order):
            return await self._handle_incomplete_order(
                plan,
                flip_transition,
                orders,
                trace,
                open_order,
                safe_amount=Decimal("0"),
            )
        return await self._finish_nonflat(plan, flip_transition, orders, trace, safe_amount=Decimal("0"))

    async def _prepare_flip_open_amount(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        orders: list[NormalizedOrder],
        trace: list[str],
        open_side: str,
    ) -> Decimal | ConnectionExecutionResult:
        execution_price = transition.quote.ask if open_side == "buy" else transition.quote.bid
        requested = abs(plan.target_signed_notional / execution_price)
        try:
            amount = await self.session.normalize_amount(plan.pair, requested)
        except VenueOperationError:
            return await self._failed_while_flat(
                plan,
                orders,
                trace,
                "normalize_amount",
                transition.target_amount,
                transition.quote,
            )
        self._validate_normalized_amount(amount, requested)
        signed_amount = amount if open_side == "buy" else -amount
        if not self._notional_within_spread(signed_amount, plan.target_signed_notional, transition.quote):
            return await self._failed_while_flat(
                plan,
                orders,
                trace,
                "incomplete_fill",
                transition.target_amount,
                transition.quote,
            )
        return amount

    async def _finish_flat(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        orders: list[NormalizedOrder],
        trace: list[str],
    ) -> ConnectionExecutionResult:
        try:
            state_while_flat = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            return self._failed(
                plan,
                "reconcile",
                orders=tuple(orders),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=Decimal("0"),
                execution_quote=transition.quote,
            )
        if state_while_flat.position.signed_amount != 0:
            return self._failed(
                plan,
                "position_mismatch",
                orders=tuple(orders),
                final_position=self._summarize(state_while_flat),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=Decimal("0"),
                execution_quote=transition.quote,
            )
        trace.append("cancel_old_protection")
        try:
            await self.session.cancel_protection(self._active_protection_ids(transition.initial_state))
        except VenueOperationError:
            return self._failed(
                plan,
                "cancel_old_protection",
                orders=tuple(orders),
                final_position=self._summarize(state_while_flat),
                requires_attention=False,
                trace=tuple(trace),
                target_amount=Decimal("0"),
                execution_quote=transition.quote,
            )
        trace.append("reconcile")
        try:
            final_state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            return self._failed(
                plan,
                "reconcile",
                orders=tuple(orders),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=Decimal("0"),
                execution_quote=transition.quote,
            )
        final = self._summarize(final_state)
        if final_state.position.signed_amount != 0 or self._active_protection_ids(final_state):
            return self._failed(
                plan,
                "position_mismatch",
                orders=tuple(orders),
                final_position=final,
                requires_attention=True,
                trace=tuple(trace),
                target_amount=Decimal("0"),
                execution_quote=transition.quote,
            )
        return self._completed(
            plan,
            Decimal("0"),
            tuple(orders),
            None,
            final,
            tuple(trace),
            transition.quote,
        )

    async def _finish_nonflat(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        orders: list[NormalizedOrder],
        trace: list[str],
        *,
        safe_amount: Decimal | None = None,
    ) -> ConnectionExecutionResult:
        expected = transition.expected_amount
        spec = ProtectionSpec(
            plan.pair,
            "long" if expected > 0 else "short",
            abs(expected),
            plan.stop_loss,
            plan.take_profit,
        )
        trace.append("replace_protection")
        try:
            protection = await self.session.replace_protection(spec)
        except VenueOperationError:
            if transition.risk_increase and orders and orders[-1].filled_amount > 0:
                return await self._compensate(
                    plan,
                    transition,
                    orders,
                    trace,
                    orders[-1],
                    safe_amount=transition.initial_state.position.signed_amount if safe_amount is None else safe_amount,
                    operation="replace_protection",
                )
            return await self._failed_after_ambiguous_reduction_protection(
                plan,
                orders,
                trace,
                spec,
                expected,
                transition.quote,
            )
        self._validate_protection_response(protection, spec)
        trace.append("reconcile")
        try:
            final_state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            if transition.risk_increase and orders and orders[-1].filled_amount > 0:
                return await self._compensate(
                    plan,
                    transition,
                    orders,
                    trace,
                    orders[-1],
                    safe_amount=transition.initial_state.position.signed_amount if safe_amount is None else safe_amount,
                    operation="reconcile",
                )
            return self._failed(
                plan,
                "reconcile",
                orders=tuple(orders),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=expected,
                execution_quote=transition.quote,
            )
        final = self._summarize(final_state, desired=spec)
        exact_protection = final.protections == (protection,)
        if final_state.position.signed_amount != expected or not final.protected or not exact_protection:
            if transition.risk_increase and orders and orders[-1].filled_amount > 0:
                return await self._compensate(
                    plan,
                    transition,
                    orders,
                    trace,
                    orders[-1],
                    safe_amount=transition.initial_state.position.signed_amount if safe_amount is None else safe_amount,
                    operation=(
                        "protection_mismatch" if final_state.position.signed_amount == expected else "position_mismatch"
                    ),
                )
            return self._failed(
                plan,
                "protection_mismatch" if final_state.position.signed_amount == expected else "position_mismatch",
                orders=tuple(orders),
                final_position=final,
                requires_attention=True,
                trace=tuple(trace),
                target_amount=expected,
                execution_quote=transition.quote,
            )
        return self._completed(plan, expected, tuple(orders), protection, final, tuple(trace), transition.quote)

    async def _finish_spot_nonflat(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        orders: list[NormalizedOrder],
        trace: list[str],
    ) -> ConnectionExecutionResult:
        trace.append("reconcile")
        try:
            final_state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            if transition.risk_increase and orders and orders[-1].filled_amount > 0:
                return await self._compensate(
                    plan,
                    transition,
                    orders,
                    trace,
                    orders[-1],
                    safe_amount=transition.initial_state.position.signed_amount,
                    operation="reconcile",
                )
            return self._failed(
                plan,
                "reconcile",
                orders=tuple(orders),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=transition.expected_amount,
                execution_quote=transition.quote,
            )
        final = self._summarize(final_state)
        if final_state.position.signed_amount != transition.expected_amount:
            if transition.risk_increase and orders and orders[-1].filled_amount > 0:
                return await self._compensate(
                    plan,
                    transition,
                    orders,
                    trace,
                    orders[-1],
                    safe_amount=transition.initial_state.position.signed_amount,
                    operation="position_mismatch",
                )
            return self._failed(
                plan,
                "position_mismatch",
                orders=tuple(orders),
                final_position=final,
                requires_attention=False,
                trace=tuple(trace),
                target_amount=transition.expected_amount,
                execution_quote=transition.quote,
            )
        return self._completed(
            plan,
            transition.expected_amount,
            tuple(orders),
            None,
            final,
            tuple(trace),
            transition.quote,
        )

    async def _handle_incomplete_order(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        orders: list[NormalizedOrder],
        trace: list[str],
        order: NormalizedOrder,
        *,
        safe_amount: Decimal | None = None,
    ) -> ConnectionExecutionResult:
        if transition.risk_increase and order.filled_amount > 0:
            return await self._compensate(
                plan,
                transition,
                orders,
                trace,
                order,
                safe_amount=transition.initial_state.position.signed_amount if safe_amount is None else safe_amount,
                operation="incomplete_fill",
            )
        return await self._failed_after_risk_reduction(
            plan,
            orders,
            trace,
            "incomplete_fill",
            transition.target_amount,
            transition.quote,
        )

    async def _compensate(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        orders: list[NormalizedOrder],
        trace: list[str],
        added_order: NormalizedOrder,
        *,
        safe_amount: Decimal,
        operation: str,
    ) -> ConnectionExecutionResult:
        return await self._compensate_amount(
            plan,
            transition,
            orders,
            trace,
            added_side=added_order.side,
            added_amount=added_order.filled_amount,
            safe_amount=safe_amount,
            operation=operation,
        )

    async def _compensate_amount(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        orders: list[NormalizedOrder],
        trace: list[str],
        *,
        added_side: str,
        added_amount: Decimal,
        safe_amount: Decimal,
        operation: str,
    ) -> ConnectionExecutionResult:
        compensation_intent = VenueOrderIntent(
            plan.pair,
            "sell" if added_side == "buy" else "buy",
            added_amount,
            "market",
            None,
            True,
        )
        trace.append("compensate_order")
        compensation_order: NormalizedOrder | None = None
        compensation_operation = ""
        try:
            compensation_order = await self.session.place_order(compensation_intent)
            self._validate_order(compensation_order, compensation_intent)
            if not self._fully_filled(compensation_order):
                compensation_operation = "incomplete_fill"
        except VenueOperationError:
            compensation_operation = "compensate_order"
        trace.append("compensate_reconcile")
        try:
            position_state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            position_state = None
            compensation_operation = compensation_operation or "compensate_reconcile"
        required_protection: ProtectionState | None = None
        if (
            compensation_operation == ""
            and position_state is not None
            and position_state.position.signed_amount == safe_amount
        ):
            final_state, required_protection, compensation_operation = await self._restore_compensation_state(
                plan,
                transition,
                position_state,
                safe_amount,
                trace,
            )
        else:
            final_state = position_state
        final = self._summarize(final_state) if final_state is not None else None
        restored = (
            compensation_order is not None
            and self._fully_filled(compensation_order)
            and final_state is not None
            and final_state.position.signed_amount == safe_amount
            and self._compensation_state_matches(plan, final_state, safe_amount, required_protection)
        )
        compensation = CompensationResult(
            True,
            restored,
            compensation_order,
            "" if restored else (compensation_operation or "compensate_reconcile"),
            safe_amount,
            required_protection,
        )
        return self._failed(
            plan,
            operation,
            orders=tuple(orders),
            compensation=compensation,
            final_position=final,
            requires_attention=not restored,
            trace=tuple(trace),
            target_amount=transition.target_amount,
            execution_quote=transition.quote,
        )

    async def _restore_compensation_state(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        position_state: OpenVenueState,
        safe_amount: Decimal,
        trace: list[str],
    ) -> tuple[OpenVenueState | None, ProtectionState | None, str]:
        trace.append("restore_protection")
        operation = ""
        required: ProtectionState | None = None
        if safe_amount == 0:
            try:
                await self.session.cancel_protection(self._active_protection_ids(position_state))
            except VenueOperationError:
                operation = "restore_protection"
        elif plan.pair.market_type != "spot":
            prior = self._initial_safe_protection(transition.initial_state, safe_amount)
            required = prior
            if prior is None:
                operation = "restore_protection"
            else:
                spec = ProtectionSpec(
                    plan.pair,
                    prior.position_side,
                    prior.amount,
                    prior.stop_loss,
                    prior.take_profit,
                )
                try:
                    required = await self.session.replace_protection(spec)
                    self._validate_protection_response(required, spec)
                except VenueOperationError:
                    operation = "restore_protection"
        try:
            final_state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            return None, required, operation or "compensate_reconcile"
        return final_state, required, operation

    async def _failed_after_order_error(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        orders: list[NormalizedOrder],
        trace: list[str],
        operation: str,
    ) -> ConnectionExecutionResult:
        try:
            state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            return self._failed(
                plan,
                operation,
                orders=tuple(orders),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=transition.target_amount,
                execution_quote=transition.quote,
            )
        observed_delta = state.position.signed_amount - transition.initial_state.position.signed_amount
        observed_added_risk = observed_delta != 0 and (observed_delta > 0) == (transition.side == "buy")
        if transition.risk_increase and observed_added_risk:
            return await self._compensate_amount(
                plan,
                transition,
                orders,
                trace,
                added_side=transition.side,
                added_amount=abs(observed_delta),
                safe_amount=transition.initial_state.position.signed_amount,
                operation=operation,
            )
        final = self._summarize(state)
        unprotected_derivative = (
            plan.pair.market_type != "spot" and state.position.signed_amount != 0 and not final.protected
        )
        return self._failed(
            plan,
            operation,
            orders=tuple(orders),
            final_position=final,
            requires_attention=unprotected_derivative
            or (
                transition.risk_increase
                and state.position.signed_amount != transition.initial_state.position.signed_amount
            ),
            trace=tuple(trace),
            target_amount=transition.target_amount,
            execution_quote=transition.quote,
        )

    async def _failed_after_risk_reduction(
        self,
        plan: ConnectionExecutionPlan,
        orders: list[NormalizedOrder],
        trace: list[str],
        operation: str,
        target_amount: Decimal,
        execution_quote: VenueQuote,
    ) -> ConnectionExecutionResult:
        try:
            state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            return self._failed(
                plan,
                operation,
                orders=tuple(orders),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=target_amount,
                execution_quote=execution_quote,
            )
        final = self._summarize(state)
        return self._failed(
            plan,
            operation,
            orders=tuple(orders),
            final_position=final,
            requires_attention=(
                plan.pair.market_type != "spot" and state.position.signed_amount != 0 and not final.protected
            ),
            trace=tuple(trace),
            target_amount=target_amount,
            execution_quote=execution_quote,
        )

    async def _failed_after_ambiguous_reduction_protection(
        self,
        plan: ConnectionExecutionPlan,
        orders: list[NormalizedOrder],
        trace: list[str],
        desired: ProtectionSpec,
        expected_amount: Decimal,
        execution_quote: VenueQuote,
    ) -> ConnectionExecutionResult:
        trace.append("reconcile")
        try:
            state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            return self._failed(
                plan,
                "replace_protection",
                orders=tuple(orders),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=expected_amount,
                execution_quote=execution_quote,
            )
        final = self._summarize(state, desired=desired)
        exact = state.position.signed_amount == expected_amount and self._has_exact_ambiguous_protection(state, desired)
        return self._failed(
            plan,
            "replace_protection",
            orders=tuple(orders),
            final_position=final,
            requires_attention=not exact,
            trace=tuple(trace),
            target_amount=expected_amount,
            execution_quote=execution_quote,
        )

    async def _failed_after_flip_close(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        orders: list[NormalizedOrder],
        trace: list[str],
        operation: str,
    ) -> ConnectionExecutionResult:
        try:
            state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            return self._failed(
                plan,
                operation,
                orders=tuple(orders),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=transition.target_amount,
                execution_quote=transition.quote,
            )
        if state.position.signed_amount != 0:
            final = self._summarize(state)
            return self._failed(
                plan,
                operation,
                orders=tuple(orders),
                final_position=final,
                requires_attention=not final.protected,
                trace=tuple(trace),
                target_amount=transition.target_amount,
                execution_quote=transition.quote,
            )

        trace.append("reconcile_flat")
        trace.append("cancel_old_protection")
        try:
            await self.session.cancel_protection(self._active_protection_ids(transition.initial_state))
        except VenueOperationError:
            return self._failed(
                plan,
                "cancel_old_protection",
                orders=tuple(orders),
                final_position=self._summarize(state),
                requires_attention=False,
                trace=tuple(trace),
                target_amount=transition.target_amount,
                execution_quote=transition.quote,
            )
        trace.append("reconcile")
        try:
            final_state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            return self._failed(
                plan,
                "reconcile",
                orders=tuple(orders),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=transition.target_amount,
                execution_quote=transition.quote,
            )
        final = self._summarize(final_state)
        safely_flat = final_state.position.signed_amount == 0 and not self._active_protection_ids(final_state)
        return self._failed(
            plan,
            operation,
            orders=tuple(orders),
            final_position=final,
            requires_attention=not safely_flat,
            trace=tuple(trace),
            target_amount=transition.target_amount,
            execution_quote=transition.quote,
        )

    async def _failed_while_flat(
        self,
        plan: ConnectionExecutionPlan,
        orders: list[NormalizedOrder],
        trace: list[str],
        operation: str,
        target_amount: Decimal,
        execution_quote: VenueQuote,
    ) -> ConnectionExecutionResult:
        try:
            state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            return self._failed(
                plan,
                operation,
                orders=tuple(orders),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=target_amount,
                execution_quote=execution_quote,
            )
        final = self._summarize(state)
        safely_flat = state.position.signed_amount == 0 and not self._active_protection_ids(state)
        return self._failed(
            plan,
            operation,
            orders=tuple(orders),
            final_position=final,
            requires_attention=not safely_flat,
            trace=tuple(trace),
            target_amount=target_amount,
            execution_quote=execution_quote,
        )

    async def _failed_after_flip_open_error(
        self,
        plan: ConnectionExecutionPlan,
        transition: _FreshTransition,
        orders: list[NormalizedOrder],
        trace: list[str],
    ) -> ConnectionExecutionResult:
        try:
            state = await self.session.list_open_state(plan.pair)
        except VenueOperationError:
            return self._failed(
                plan,
                "open_target_side",
                orders=tuple(orders),
                requires_attention=True,
                trace=tuple(trace),
                target_amount=transition.target_amount,
                execution_quote=transition.quote,
            )
        observed = state.position.signed_amount
        opened_target_side = observed != 0 and (observed > 0) == (transition.target_amount > 0)
        if opened_target_side:
            return await self._compensate_amount(
                plan,
                transition,
                orders,
                trace,
                added_side="buy" if observed > 0 else "sell",
                added_amount=abs(observed),
                safe_amount=Decimal("0"),
                operation="open_target_side",
            )
        final = self._summarize(state)
        safely_flat = observed == 0 and not self._active_protection_ids(state)
        return self._failed(
            plan,
            "open_target_side",
            orders=tuple(orders),
            final_position=final,
            requires_attention=not safely_flat,
            trace=tuple(trace),
            target_amount=transition.target_amount,
            execution_quote=transition.quote,
        )

    @staticmethod
    def _validate_fresh_state(plan: ConnectionExecutionPlan, state: OpenVenueState) -> None:
        if not isinstance(state, OpenVenueState) or state.position.pair != plan.pair:
            raise ValueError("fresh open state must match the execution pair")

    @classmethod
    def _validate_fresh_inputs(
        cls,
        plan: ConnectionExecutionPlan,
        state: OpenVenueState,
        quote: VenueQuote,
    ) -> None:
        cls._validate_fresh_state(plan, state)
        if not isinstance(quote, VenueQuote) or quote.pair != plan.pair:
            raise ValueError("latest quote must match the execution pair")

    @staticmethod
    def _target_transition(
        plan: ConnectionExecutionPlan,
        current: Decimal,
        quote: VenueQuote,
    ) -> tuple[Decimal, Decimal, str]:
        if plan.target_signed_notional == 0:
            target = Decimal("0")
            if current == 0:
                return target, Decimal("0"), plan.side
            return target, -current, "sell" if current > 0 else "buy"
        current_values = (current * quote.bid, current * quote.ask)
        if min(current_values) <= plan.target_signed_notional <= max(current_values):
            return current, Decimal("0"), plan.side
        current_notional = current * quote.last
        side = "buy" if plan.target_signed_notional > current_notional else "sell"
        execution_price = quote.ask if side == "buy" else quote.bid
        target = plan.target_signed_notional / execution_price
        delta = target - current
        if delta == 0:
            return target, delta, side
        if (delta > 0) != (side == "buy"):
            raise ValueError("latest quote produces an incoherent target transition")
        return target, delta, side

    @staticmethod
    def _notional_within_spread(amount: Decimal, target_notional: Decimal, quote: VenueQuote) -> bool:
        executable_values = (amount * quote.bid, amount * quote.ask)
        return min(executable_values) <= target_notional <= max(executable_values)

    @staticmethod
    def _validate_normalized_amount(normalized: Decimal, requested: Decimal, *, exact: bool = False) -> None:
        if not isinstance(normalized, Decimal) or not normalized.is_finite() or normalized <= 0:
            raise ValueError("session returned an invalid normalized amount")
        if normalized > requested or (exact and normalized != requested):
            raise ValueError("session normalized amount violates the requested risk transition")

    @staticmethod
    def _validate_order(order: NormalizedOrder, intent: VenueOrderIntent) -> None:
        if not isinstance(order, NormalizedOrder):
            raise TypeError("venue session must return a NormalizedOrder")
        if (
            order.pair != intent.pair
            or order.side != intent.side
            or order.amount != intent.amount
            or order.reduce_only is not intent.reduce_only
        ):
            raise ValueError("normalized order does not match its order intent")

    @staticmethod
    def _fully_filled(order: NormalizedOrder) -> bool:
        return order.status in {"filled", "closed"} and order.filled_amount == order.amount

    @staticmethod
    def _validate_protection_response(protection: ProtectionState, spec: ProtectionSpec) -> None:
        if not isinstance(protection, ProtectionState):
            raise TypeError("venue session must return a ProtectionState")
        if (
            protection.pair != spec.pair
            or protection.position_side != spec.position_side
            or protection.amount != spec.amount
            or protection.stop_loss != spec.stop_loss
            or protection.take_profit != spec.take_profit
            or not protection.active
            or protection.triggered
        ):
            raise ValueError("replacement protection does not match its specification")

    @staticmethod
    def _has_exact_ambiguous_protection(state: OpenVenueState, desired: ProtectionSpec) -> bool:
        active = tuple(protection for protection in state.protections if protection.active and not protection.triggered)
        if len(active) != 1:
            return False
        protection = active[0]
        return (
            len(protection.protection_ids) == 1
            and protection.pair == desired.pair
            and protection.position_side == desired.position_side
            and protection.amount == desired.amount
            and protection.stop_loss == desired.stop_loss
            and protection.take_profit == desired.take_profit
        )

    @classmethod
    def _summarize(
        cls,
        state: OpenVenueState,
        *,
        desired: ProtectionSpec | None = None,
    ) -> ExecutionFinalPosition:
        ids = cls._active_protection_ids(state)
        protected = False
        if state.position.signed_amount != 0:
            expected_side = "long" if state.position.signed_amount > 0 else "short"
            protected = any(
                protection.active
                and not protection.triggered
                and protection.position_side == expected_side
                and protection.amount == abs(state.position.signed_amount)
                and (
                    desired is None
                    or (protection.stop_loss == desired.stop_loss and protection.take_profit == desired.take_profit)
                )
                for protection in state.protections
            )
        active = tuple(protection for protection in state.protections if protection.active and not protection.triggered)
        return ExecutionFinalPosition(state.position, protected, ids, active)

    @staticmethod
    def _active_protection_ids(state: OpenVenueState) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                protection_id
                for protection in state.protections
                if protection.active and not protection.triggered
                for protection_id in protection.protection_ids
            )
        )

    @staticmethod
    def _initial_safe_protection(initial: OpenVenueState, safe_amount: Decimal) -> ProtectionState | None:
        active = tuple(
            protection for protection in initial.protections if protection.active and not protection.triggered
        )
        expected_side = "long" if safe_amount > 0 else "short"
        if len(active) != 1:
            return None
        protection = active[0]
        if protection.position_side != expected_side or protection.amount != abs(safe_amount):
            return None
        return protection

    @classmethod
    def _compensation_state_matches(
        cls,
        plan: ConnectionExecutionPlan,
        final: OpenVenueState,
        safe_amount: Decimal,
        required_protection: ProtectionState | None,
    ) -> bool:
        active = tuple(protection for protection in final.protections if protection.active and not protection.triggered)
        if safe_amount == 0 or plan.pair.market_type == "spot":
            return active == ()
        return required_protection is not None and active == (required_protection,)

    @staticmethod
    def _is_reduction(current: Decimal, target: Decimal) -> bool:
        return target == 0 or (current * target > 0 and abs(target) < abs(current))

    @staticmethod
    def _failed(
        plan: ConnectionExecutionPlan,
        operation: str,
        *,
        orders: tuple[NormalizedOrder, ...] = (),
        compensation: CompensationResult = NO_COMPENSATION,
        final_position: ExecutionFinalPosition | None = None,
        requires_attention: bool = False,
        trace: tuple[str, ...] = (),
        target_amount: Decimal | None = None,
        execution_quote: VenueQuote | None = None,
    ) -> ConnectionExecutionResult:
        return ConnectionExecutionResult.failed(
            plan,
            operation,
            orders=orders,
            compensation=compensation,
            final_position=final_position,
            requires_attention=requires_attention,
            trace=trace,
            target_signed_amount=target_amount,
            execution_quote=execution_quote,
        )

    @staticmethod
    def _completed(
        plan: ConnectionExecutionPlan,
        target_amount: Decimal,
        orders: tuple[NormalizedOrder, ...],
        protection: ProtectionState | None,
        final_position: ExecutionFinalPosition,
        trace: tuple[str, ...],
        execution_quote: VenueQuote,
    ) -> ConnectionExecutionResult:
        return ConnectionExecutionResult(
            plan.book_id,
            plan.connection_id,
            plan.pair,
            plan.target_signed_notional,
            target_amount,
            "completed",
            orders,
            protection,
            NO_COMPENSATION,
            final_position,
            "",
            False,
            trace,
            execution_quote,
        )
