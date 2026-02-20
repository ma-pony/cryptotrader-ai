"""Reconciler: compares local order state with exchange state + orphan detection."""

from __future__ import annotations

import logging
from cryptotrader.execution.exchange import ExchangeAdapter
from cryptotrader.models import Order, OrderStatus

logger = logging.getLogger(__name__)

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
            try:
                remote = await self._exchange.get_order(order.exchange_id)
            except Exception as e:
                logger.warning("Failed to fetch order %s: %s", order.exchange_id, e)
                continue
            remote_status = _STATUS_MAP.get(remote.get("status", ""), None)
            if remote_status and remote_status != order.status:
                mismatches.append((order, remote_status.value))
                logger.warning("Mismatch: %s local=%s remote=%s", order.exchange_id, order.status, remote_status)
        return mismatches

    async def detect_orphans(self, local_ids: set[str]) -> list[dict]:
        """Detect exchange orders not tracked locally."""
        orphans = []
        try:
            open_orders = await self._exchange._exchange.fetch_open_orders()
            for o in open_orders:
                if o.get("id") not in local_ids:
                    orphans.append(o)
                    logger.warning("Orphan order detected: %s", o.get("id"))
        except Exception as e:
            logger.warning("Orphan detection failed: %s", e)
        return orphans
