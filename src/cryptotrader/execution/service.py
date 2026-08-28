"""顺序执行目标仓位差值并维护唯一交易所保护单。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cryptotrader.decision.models import OrderIntent
from cryptotrader.models import Order, OrderStatus

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from cryptotrader.decision.models import ExecutionPlan
    from cryptotrader.execution.exchange import ExchangeAdapter
    from cryptotrader.execution.order import OrderManager
    from cryptotrader.signals.models import SignalContext


@dataclass(frozen=True)
class ExecutionOrderResult:
    intent: OrderIntent
    status: str
    exchange_id: str | None
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
        exchange: ExchangeAdapter,
        *,
        manage_protection: bool = True,
    ) -> None:
        self.orders = orders
        self.exchange = exchange
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
        processor = getattr(self.exchange, "process_pending_protection", None)
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
        supports_protection = self.exchange.supports_protection_orders()
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
            algo_id = await self.exchange.place_algo_oco(
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
        placed, raw = await self.orders.place(order, self.exchange)
        filled_amount = self._filled_amount(intent, placed, raw)
        results.append(
            ExecutionOrderResult(
                intent=intent,
                status=placed.status.value,
                exchange_id=placed.exchange_id,
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
        pending = await self.exchange.list_pending_algos(pair=pair)
        return tuple(str(algo_id) for item in pending for algo_id in (item.get("algoId") or item.get("id"),) if algo_id)

    async def _cancel_algos(
        self,
        algo_ids: tuple[str, ...],
        pair: str,
    ) -> tuple[tuple[str, ...], Exception | None]:
        for index, algo_id in enumerate(algo_ids):
            try:
                await self.exchange.cancel_algo(algo_id, pair)
            except Exception as error:
                return algo_ids[index:], error
        return (), None
