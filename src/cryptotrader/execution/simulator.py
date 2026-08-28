"""Paper trading exchange simulator."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from cryptotrader.data.market import _timeframe_ms
from cryptotrader.execution.service import ProtectionTriggerResult
from cryptotrader.models import Order
from cryptotrader.pair import Pair

if TYPE_CHECKING:
    from cryptotrader.signals.models import SignalContext


class PaperExchange:
    def __init__(
        self,
        initial_balances: dict[str, float] | None = None,
        initial_positions: dict[str, dict[str, float]] | None = None,
    ) -> None:
        from cryptotrader.config import load_config

        _cfg = load_config()
        self._orders: dict[str, dict[str, Any]] = {}
        self._algos: dict[str, dict[str, Any]] = {}
        self._derivative_positions: dict[str, dict[str, Any]] = {}
        if initial_balances is not None:
            self._balances: dict[str, float] = dict(initial_balances)
        else:
            self._balances = {"USDT": _cfg.backtest.initial_capital}
        # Cost-basis tracking: {asset: {"total_cost": float, "total_amount": float}}
        self._cost_basis: dict[str, dict[str, float]] = {}
        if initial_positions:
            for pair, pos in initial_positions.items():
                pair_object = Pair.parse(pair)
                if pair_object.market_type != "spot":
                    amount = float(pos.get("amount", 0.0) or 0.0)
                    if amount:
                        self._derivative_positions[pair] = {
                            "amount": amount,
                            "side": "long" if amount > 0.0 else "short",
                            "avg_price": float(pos.get("avg_price", 0.0) or 0.0),
                            "unrealized_pnl": 0.0,
                            "liquidation_price": None,
                        }
                    continue
                asset = Pair.parse(pair).base
                amount = pos.get("amount", 0.0)
                avg_price = pos.get("avg_price", 0.0)
                if amount != 0 and avg_price > 0:
                    self._cost_basis[asset] = {"total_cost": abs(amount) * avg_price, "total_amount": abs(amount)}
        self._lock = asyncio.Lock()
        self._slippage_base: float = _cfg.backtest.slippage_base
        self._fee_bps: float = _cfg.backtest.fee_bps

    def supports_protection_orders(self) -> bool:
        return True

    def estimate_slippage(self, order: Order) -> float:
        impact = order.amount * order.price * 1e-8
        return self._slippage_base + impact

    def _update_derivative_position(self, order: Order, fill_price: float) -> float:
        current = self._derivative_positions.get(order.pair)
        current_amount = float((current or {}).get("amount", 0.0) or 0.0)
        delta = order.amount if order.side == "buy" else -order.amount
        next_amount = current_amount + delta
        realized_pnl = 0.0
        if current_amount * delta < 0.0:
            closed_amount = min(abs(current_amount), abs(delta))
            direction = 1.0 if current_amount > 0.0 else -1.0
            avg_price = float((current or {}).get("avg_price", fill_price))
            realized_pnl = (fill_price - avg_price) * closed_amount * direction
        if abs(next_amount) < 1e-12:
            self._derivative_positions.pop(order.pair, None)
            return realized_pnl
        if current_amount * next_amount <= 0.0:
            avg_price = fill_price
        elif abs(next_amount) > abs(current_amount):
            current_cost = abs(current_amount) * float((current or {}).get("avg_price", fill_price))
            avg_price = (current_cost + abs(delta) * fill_price) / abs(next_amount)
        else:
            avg_price = float((current or {}).get("avg_price", fill_price))
        self._derivative_positions[order.pair] = {
            "amount": next_amount,
            "side": "long" if next_amount > 0.0 else "short",
            "avg_price": avg_price,
            "unrealized_pnl": 0.0,
            "liquidation_price": None,
        }
        return realized_pnl

    async def place_order(self, order: Order) -> dict[str, Any]:
        async with self._lock:
            order_id = str(uuid.uuid4())
            slippage = self.estimate_slippage(order)
            fill_price = order.price * (1 + slippage if order.side == "buy" else 1 - slippage)
            fee = order.amount * fill_price * self._fee_bps / 10000
            cost = order.amount * fill_price
            pair_obj = Pair.parse(order.pair)
            base = pair_obj.base
            # Derivative orders (perp / future, ccxt symbol "BTC/USDT:USDT") are
            # margin-denominated, NOT spot. Sell-side does NOT require holding
            # the base; both sides only need free USDT margin = notional/leverage.
            # We don't model perp positions inside ``_balances`` (which is asset-
            # denominated for spot); the canonical perp position state lives in
            # the DB-backed PortfolioManager, populated by the journal after a
            # successful order. Here we just gate on USDT margin and succeed.
            is_derivative = pair_obj.market_type != "spot"

            # Balance pre-check (include fee in buy cost)
            if is_derivative:
                # Conservative paper-mode margin: assume 1x (worst-case full notional).
                # When real leverage > 1, USDT cash buffer is even larger than needed,
                # so this never spuriously fails an order that the live exchange
                # would accept. PaperExchange does not consult the leverage config.
                if not order.reduce_only:
                    margin_required = cost
                    available = self._balances.get("USDT", 0)
                    if available < margin_required + fee:
                        return {
                            "id": order_id,
                            "status": "failed",
                            "reason": f"Insufficient USDT margin: {available:.2f} < {margin_required + fee:.2f}",
                        }
            elif order.side == "buy":
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
            if is_derivative:
                # Derivative fills: cash buffer for fees; canonical perp position
                # state is owned by PortfolioManager (populated post-fill via
                # nodes/execution._update_portfolio reading exchange position).
                # We do NOT pretend to track perp positions in _balances (which
                # is asset-denominated for spot only). This means PaperExchange's
                # ``get_positions`` will not surface the perp until the live
                # exchange path or DB read merges them in — acceptable because
                # the journal records the order and the portfolio API uses DB.
                realized_pnl = self._update_derivative_position(order, fill_price)
                self._balances["USDT"] = self._balances.get("USDT", 0) + realized_pnl - fee
            elif order.side == "buy":
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

    async def get_free_balance(self) -> dict[str, float]:
        return await self.get_balance()

    async def get_positions(self, current_prices: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
        """Return current positions with cost-basis and unrealized PnL.

        Args:
            current_prices: {pair: price} for PnL calculation. If not provided, PnL is 0.

        Returns: {pair: {"amount", "side", "avg_price", "unrealized_pnl", "liquidation_price"}}
        """
        async with self._lock:
            positions: dict[str, dict[str, Any]] = {
                pair: dict(position) for pair, position in self._derivative_positions.items()
            }
            for pair, position in positions.items():
                current_price = (current_prices or {}).get(pair, 0.0)
                if current_price > 0.0:
                    direction = 1.0 if position["amount"] > 0.0 else -1.0
                    position["unrealized_pnl"] = (
                        (current_price - position["avg_price"]) * abs(position["amount"]) * direction
                    )
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

    async def place_algo_oco(
        self,
        pair: str,
        *,
        side: str,
        amount: float,
        sl_trigger_px: float,
        tp_trigger_px: float,
        pos_side: str,
        created_as_of: datetime | None = None,
    ) -> str:
        async with self._lock:
            algo_id = str(uuid.uuid4())
            self._algos[algo_id] = {
                "algoId": algo_id,
                "pair": pair,
                "side": side,
                "amount": amount,
                "sl_trigger_px": sl_trigger_px,
                "tp_trigger_px": tp_trigger_px,
                "pos_side": pos_side,
                "status": "pending",
                "bar_watermark": created_as_of or datetime.now(UTC),
            }
            return algo_id

    async def process_pending_protection(self, context: SignalContext) -> ProtectionTriggerResult | None:
        """Trigger at most one pending OCO from a bar closed after its creation."""
        pair = context.pair.canonical()
        async with self._lock:
            pending = [
                dict(algo) for algo in self._algos.values() if algo["status"] == "pending" and algo["pair"] == pair
            ]

        for algo in pending:
            latest = self._latest_new_closed_bar(context, algo["bar_watermark"])
            if latest is None:
                continue
            closed_at, high, low = latest
            trigger = self._protection_trigger(algo, high, low)

            algo_id = str(algo["algoId"])
            async with self._lock:
                current = self._algos.get(algo_id)
                if current is None or current["status"] != "pending":
                    continue
                current["bar_watermark"] = closed_at
                if trigger is None:
                    continue
                current["status"] = "triggering"

            trigger_price, trigger_reason = trigger
            result = await self.place_order(
                Order(
                    pair=pair,
                    side=str(algo["side"]),
                    amount=float(algo["amount"]),
                    price=float(trigger_price),
                    reduce_only=True,
                )
            )
            async with self._lock:
                current = self._algos[algo_id]
                if result.get("status") == "filled":
                    current.update(
                        status="triggered",
                        trigger_reason=trigger_reason,
                        trigger_price=float(trigger_price),
                        trigger_order_id=result.get("id"),
                    )
                    return ProtectionTriggerResult(
                        algo_id=algo_id,
                        trigger_reason=trigger_reason,
                        trigger_price=float(trigger_price),
                        order_id=str(result.get("id")),
                    )
                current["status"] = "pending"
                current["trigger_error"] = result.get("reason") or result.get("status")
        return None

    @staticmethod
    def _protection_trigger(
        algo: dict[str, Any],
        high: float,
        low: float,
    ) -> tuple[float, str] | None:
        if algo["pos_side"] == "long":
            if low <= algo["sl_trigger_px"]:
                return float(algo["sl_trigger_px"]), "stop_loss"
            if high >= algo["tp_trigger_px"]:
                return float(algo["tp_trigger_px"]), "take_profit"
            return None
        if high >= algo["sl_trigger_px"]:
            return float(algo["sl_trigger_px"]), "stop_loss"
        if low <= algo["tp_trigger_px"]:
            return float(algo["tp_trigger_px"]), "take_profit"
        return None

    @staticmethod
    def _latest_new_closed_bar(
        context: SignalContext,
        watermark: datetime,
    ) -> tuple[datetime, float, float] | None:
        candidates: list[tuple[pd.Timestamp, Any]] = []
        watermark_ns = pd.Timestamp(watermark).value
        as_of_ns = pd.Timestamp(context.as_of).value
        for timeframe, snapshot in context.snapshots.items():
            frame = snapshot.market.ohlcv
            if frame.empty:
                continue
            closed_at = pd.Timestamp(frame.index[-1]) + pd.Timedelta(milliseconds=_timeframe_ms(timeframe))
            if watermark_ns < closed_at.value <= as_of_ns:
                candidates.append((closed_at, frame.iloc[-1]))
        if not candidates:
            return None
        closed_at, bar = max(candidates, key=lambda item: item[0].value)
        return closed_at.to_pydatetime(), float(bar["high"]), float(bar["low"])

    async def cancel_algo(self, algo_id: str, pair: str) -> None:
        async with self._lock:
            algo = self._algos.get(algo_id)
            if algo is not None and algo["pair"] == pair:
                algo["status"] = "cancelled"

    async def list_pending_algos(self, pair: str | None = None) -> list[dict[str, Any]]:
        async with self._lock:
            return [
                dict(algo)
                for algo in self._algos.values()
                if algo["status"] == "pending" and (pair is None or algo["pair"] == pair)
            ]

    async def close(self) -> None:
        pass
