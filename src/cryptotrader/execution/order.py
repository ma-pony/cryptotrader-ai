"""Order management with state transitions."""

from __future__ import annotations

from cryptotrader.execution.exchange import ExchangeAdapter
from cryptotrader.models import Order, OrderStatus, VALID_TRANSITIONS


class OrderManager:
    def transition(self, order: Order, new_status: OrderStatus) -> Order:
        allowed = VALID_TRANSITIONS.get(order.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition: {order.status.value} -> {new_status.value}"
            )
        order.status = new_status
        return order

    async def place(self, order: Order, exchange: ExchangeAdapter) -> Order:
        self.transition(order, OrderStatus.SUBMITTED)
        try:
            result = await exchange.place_order(order)
            order.exchange_id = result.get("id")
            self.transition(order, OrderStatus.FILLED)
        except Exception:
            self.transition(order, OrderStatus.FAILED)
        return order
