"""Paper trading exchange simulator."""

from __future__ import annotations

import uuid
from typing import Any

from cryptotrader.models import Order


class PaperExchange:
    def __init__(self) -> None:
        self._orders: dict[str, dict[str, Any]] = {}
        self._balances: dict[str, float] = {"USDT": 10000.0}

    def estimate_slippage(self, order: Order) -> float:
        base = 0.0005
        impact = order.amount * order.price * 1e-8
        return base + impact

    async def place_order(self, order: Order) -> dict[str, Any]:
        order_id = str(uuid.uuid4())
        slippage = self.estimate_slippage(order)
        fill_price = order.price * (1 + slippage if order.side == "buy" else 1 - slippage)
        cost = order.amount * fill_price
        base = order.pair.split("/")[0]

        # Balance pre-check
        if order.side == "buy":
            available = self._balances.get("USDT", 0)
            if available < cost:
                return {"id": order_id, "status": "failed", "reason": f"Insufficient USDT: {available:.2f} < {cost:.2f}"}
        else:
            available = self._balances.get(base, 0)
            if available < order.amount:
                return {"id": order_id, "status": "failed", "reason": f"Insufficient {base}: {available:.6f} < {order.amount:.6f}"}

        record = {
            "id": order_id,
            "status": "filled",
            "pair": order.pair,
            "side": order.side,
            "amount": order.amount,
            "price": fill_price,
            "slippage": slippage,
        }
        self._orders[order_id] = record
        if order.side == "buy":
            self._balances["USDT"] -= cost
            # Credit base at the actual fill price (cost / fill_price = order.amount)
            self._balances[base] = self._balances.get(base, 0) + cost / fill_price
        else:
            # Debit base, credit USDT at fill price (with slippage)
            self._balances["USDT"] = self._balances.get("USDT", 0) + cost
            self._balances[base] -= order.amount
        return record

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} not found")
        self._orders[order_id]["status"] = "cancelled"
        return self._orders[order_id]

    async def get_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} not found")
        return self._orders[order_id]

    async def get_balance(self) -> dict[str, float]:
        return {k: v for k, v in self._balances.items() if v != 0}

    async def fetch_open_orders(self) -> list[dict[str, Any]]:
        return [o for o in self._orders.values() if o.get("status") == "open"]

    async def close(self) -> None:
        pass
