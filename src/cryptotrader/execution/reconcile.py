"""Reconciler: compares local order state with exchange state."""

from __future__ import annotations

from cryptotrader.execution.exchange import ExchangeAdapter
from cryptotrader.models import Order, OrderStatus

_STATUS_MAP = {
    "open": OrderStatus.SUBMITTED,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
}


class Reconciler:
    def __init__(self, exchange: ExchangeAdapter) -> None:
        self._exchange = exchange

    async def reconcile(self, local_orders: list[Order]) -> list[tuple[Order, str]]:
        mismatches: list[tuple[Order, str]] = []
        for order in local_orders:
            if not order.exchange_id:
                continue
            remote = await self._exchange.get_order(order.exchange_id)
            remote_status = _STATUS_MAP.get(remote.get("status", ""), None)
            if remote_status and remote_status != order.status:
                mismatches.append((order, remote_status.value))
        return mismatches
