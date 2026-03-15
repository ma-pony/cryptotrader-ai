"""Paper trading exchange simulator."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cryptotrader.models import Order


class PaperExchange:
    def __init__(
        self,
        initial_balances: dict[str, float] | None = None,
        initial_positions: dict[str, dict[str, float]] | None = None,
    ) -> None:
        from cryptotrader.config import load_config

        _cfg = load_config()
        self._orders: dict[str, dict[str, Any]] = {}
        if initial_balances is not None:
            self._balances: dict[str, float] = dict(initial_balances)
        else:
            self._balances = {"USDT": _cfg.backtest.initial_capital}
        # Cost-basis tracking: {asset: {"total_cost": float, "total_amount": float}}
        self._cost_basis: dict[str, dict[str, float]] = {}
        if initial_positions:
            for pair, pos in initial_positions.items():
                asset = pair.split("/")[0]
                amount = pos.get("amount", 0.0)
                avg_price = pos.get("avg_price", 0.0)
                if amount != 0 and avg_price > 0:
                    self._cost_basis[asset] = {"total_cost": abs(amount) * avg_price, "total_amount": abs(amount)}
        self._lock = asyncio.Lock()
        self._slippage_base: float = _cfg.backtest.slippage_base
        self._fee_bps: float = _cfg.backtest.fee_bps

    def estimate_slippage(self, order: Order) -> float:
        impact = order.amount * order.price * 1e-8
        return self._slippage_base + impact

    async def place_order(self, order: Order) -> dict[str, Any]:
        async with self._lock:
            order_id = str(uuid.uuid4())
            slippage = self.estimate_slippage(order)
            fill_price = order.price * (1 + slippage if order.side == "buy" else 1 - slippage)
            fee = order.amount * fill_price * self._fee_bps / 10000
            cost = order.amount * fill_price
            base = order.pair.split("/")[0]

            # Balance pre-check (include fee in buy cost)
            if order.side == "buy":
                available = self._balances.get("USDT", 0)
                if available < cost + fee:
                    return {
                        "id": order_id,
                        "status": "failed",
                        "reason": f"Insufficient USDT: {available:.2f} < {cost + fee:.2f}",
                    }
            else:
                available = self._balances.get(base, 0)
                if available < order.amount:
                    return {
                        "id": order_id,
                        "status": "failed",
                        "reason": f"Insufficient {base}: {available:.6f} < {order.amount:.6f}",
                    }

            record = {
                "id": order_id,
                "status": "filled",
                "pair": order.pair,
                "side": order.side,
                "amount": order.amount,
                "price": fill_price,
                "slippage": slippage,
                "fee": fee,
            }
            self._orders[order_id] = record
            if order.side == "buy":
                self._balances["USDT"] -= cost + fee
                self._balances[base] = self._balances.get(base, 0) + order.amount
                # Update cost basis
                cb = self._cost_basis.get(base, {"total_cost": 0.0, "total_amount": 0.0})
                cb["total_cost"] += cost
                cb["total_amount"] += order.amount
                self._cost_basis[base] = cb
            else:
                self._balances["USDT"] = self._balances.get("USDT", 0) + cost - fee
                self._balances[base] -= order.amount
                # Reduce cost basis proportionally
                cb = self._cost_basis.get(base, {"total_cost": 0.0, "total_amount": 0.0})
                if cb["total_amount"] > 0:
                    ratio = min(1.0, order.amount / cb["total_amount"])
                    cb["total_cost"] *= 1 - ratio
                    cb["total_amount"] = max(0.0, cb["total_amount"] - order.amount)
                self._cost_basis[base] = cb
            return record

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        async with self._lock:
            if order_id not in self._orders:
                raise ValueError(f"Order {order_id} not found")
            self._orders[order_id]["status"] = "cancelled"
            return dict(self._orders[order_id])

    async def get_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        async with self._lock:
            if order_id not in self._orders:
                raise ValueError(f"Order {order_id} not found")
            return dict(self._orders[order_id])

    async def get_balance(self) -> dict[str, float]:
        async with self._lock:
            return {k: v for k, v in self._balances.items() if v != 0}

    async def get_positions(self, current_prices: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
        """Return current positions with cost-basis and unrealized PnL.

        Args:
            current_prices: {pair: price} for PnL calculation. If not provided, PnL is 0.

        Returns: {pair: {"amount", "side", "avg_price", "unrealized_pnl", "liquidation_price"}}
        """
        async with self._lock:
            positions: dict[str, dict[str, Any]] = {}
            for asset, amount in self._balances.items():
                if asset == "USDT" or amount == 0:
                    continue
                pair = f"{asset}/USDT"
                cb = self._cost_basis.get(asset, {"total_cost": 0.0, "total_amount": 0.0})
                avg_price = cb["total_cost"] / cb["total_amount"] if cb["total_amount"] > 0 else 0.0
                current_price = (current_prices or {}).get(pair, 0.0)
                unrealized_pnl = (current_price - avg_price) * amount if avg_price > 0 and current_price > 0 else 0.0
                positions[pair] = {
                    "amount": amount,
                    "side": "long" if amount > 0 else "short",
                    "avg_price": avg_price,
                    "unrealized_pnl": unrealized_pnl,
                    "liquidation_price": None,
                }
            return positions

    async def fetch_open_orders(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [o for o in self._orders.values() if o.get("status") == "open"]

    async def close(self) -> None:
        pass
