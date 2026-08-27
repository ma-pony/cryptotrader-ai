"""顺序执行目标仓位差值并维护唯一交易所保护单。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cryptotrader.models import Order, OrderStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cryptotrader.decision.models import ExecutionPlan, OrderIntent
    from cryptotrader.execution.exchange import ExchangeAdapter
    from cryptotrader.execution.order import OrderManager
    from cryptotrader.signals.models import SignalContext


@dataclass(frozen=True)
class ExecutionOrderResult:
    intent: OrderIntent
    status: str
    exchange_id: str | None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    succeeded: bool
    orders: tuple[ExecutionOrderResult, ...]
    algo_id: str | None
    error: str | None


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
        final_signed_amount = context.current_position.signed_amount
        for intent in plan.intents:
            order = Order(
                pair=intent.pair,
                side=intent.side,
                amount=intent.amount,
                price=context.current_price,
                reduce_only=intent.reduce_only,
            )
            placed, raw = await self.orders.place(order, self.exchange)
            results.append(
                ExecutionOrderResult(
                    intent=intent,
                    status=placed.status.value,
                    exchange_id=placed.exchange_id,
                    raw=raw,
                )
            )
            if placed.status != OrderStatus.FILLED:
                reason = str(raw.get("reason") or raw.get("error_msg") or placed.status.value)
                return ExecutionResult(False, tuple(results), None, reason)
            final_signed_amount += intent.amount if intent.side == "buy" else -intent.amount

        if not self.manage_protection:
            return ExecutionResult(True, tuple(results), None, None)

        try:
            await self._cancel_existing(context.pair.canonical())
            if abs(final_signed_amount) < 1e-12:
                return ExecutionResult(True, tuple(results), None, None)
            if plan.stop_loss is None or plan.take_profit is None:
                raise ValueError("non-flat execution requires stop loss and take profit")
            pos_side = "long" if final_signed_amount > 0.0 else "short"
            algo_id = await self.exchange.place_algo_oco(
                context.pair.canonical(),
                side="sell" if pos_side == "long" else "buy",
                amount=abs(final_signed_amount),
                sl_trigger_px=plan.stop_loss,
                tp_trigger_px=plan.take_profit,
                pos_side=pos_side,
            )
        except Exception as error:
            return ExecutionResult(
                False,
                tuple(results),
                None,
                f"{type(error).__name__}: {error}",
            )
        return ExecutionResult(True, tuple(results), algo_id, None)

    async def _cancel_existing(self, pair: str) -> None:
        pending = await self.exchange.list_pending_algos(pair=pair)
        for item in pending:
            algo_id = item.get("algoId") or item.get("id")
            if algo_id:
                await self.exchange.cancel_algo(str(algo_id), pair)
