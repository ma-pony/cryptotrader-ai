"""把当前仓位与目标仓位转换为订单差值。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.decision.models import ExecutionPlan, OrderIntent

if TYPE_CHECKING:
    from cryptotrader.decision.models import TradePlan
    from cryptotrader.signals.models import SignalContext


class ExecutionPlanningError(ValueError):
    pass


class ExecutionPlanner:
    def __init__(self, max_single_pct: float) -> None:
        self.max_single_pct = max_single_pct

    def plan(self, context: SignalContext, trade_plan: TradePlan) -> ExecutionPlan:
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
            close_intent = OrderIntent(
                pair=context.pair.canonical(),
                side="sell" if current_signed > 0.0 else "buy",
                amount=abs(current_signed),
                reduce_only=True,
            )
            enter_intent = OrderIntent(
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
            intents: tuple[OrderIntent, ...] = ()
        else:
            intents = (
                OrderIntent(
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
