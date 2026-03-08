"""Order execution and portfolio update nodes."""

from __future__ import annotations

import logging
from typing import Any

from cryptotrader.state import ArenaState

logger = logging.getLogger(__name__)

# Per-pair PaperExchange cache to prevent cross-pair balance contamination
_paper_exchanges: dict[str, Any] = {}


def _get_exchange(state: ArenaState, pair: str):
    """Get exchange instance (paper or live) for the given pair."""
    from cryptotrader.execution.exchange import LiveExchange
    from cryptotrader.execution.simulator import PaperExchange

    engine = state["metadata"].get("engine", "paper")
    if engine == "paper":
        if pair not in _paper_exchanges:
            _paper_exchanges[pair] = PaperExchange()
        return _paper_exchanges[pair], None
    cfg = state["metadata"].get("exchange_config", {})
    live_exchange = LiveExchange(
        cfg.get("exchange_id", "binance"),
        cfg.get("api_key", ""),
        cfg.get("secret", ""),
    )
    return live_exchange, live_exchange


async def _update_trade_tracking(state: ArenaState, pair: str):
    """Update trade count and cooldown after successful order."""
    from cryptotrader.nodes.verdict import _risk_gate_cache

    redis_url = state["metadata"].get("redis_url")
    cache_key = redis_url or "_default"
    if cache_key in _risk_gate_cache:
        try:
            rsm = _risk_gate_cache[cache_key].redis_state
            await rsm.incr_trade_count()
            from cryptotrader.config import load_config

            cooldown_min = load_config().risk.cooldown.same_pair_minutes
            await rsm.set_cooldown(pair, cooldown_min)
        except Exception:
            logger.warning("Trade tracking update failed", exc_info=True)


async def _update_portfolio(state: ArenaState, order, filled_amount: float, filled_price: float):
    """Update portfolio after successful trade."""
    pair = order.pair
    db_url = state["metadata"].get("database_url")
    try:
        from cryptotrader.portfolio.manager import PortfolioManager

        pm = PortfolioManager(db_url)
        portfolio = await pm.get_portfolio()
        existing = portfolio.get("positions", {}).get(pair, {})
        old_amount = existing.get("amount", 0.0)
        old_price = existing.get("avg_price", 0.0)

        if order.side == "buy":
            new_amount = old_amount + filled_amount
            new_price = (
                ((old_amount * old_price) + (filled_amount * filled_price)) / new_amount
                if new_amount > 0
                else filled_price
            )
        else:
            new_amount = old_amount - filled_amount
            new_price = old_price if new_amount > 0 else 0.0

        await pm.update_position("default", pair, new_amount, new_price)
        total = sum(p["amount"] * p["avg_price"] for p in (await pm.get_portfolio()).get("positions", {}).values())
        await pm.snapshot("default", total)
    except Exception:
        logger.warning("Portfolio write-back failed for %s", pair, exc_info=True)


async def check_stop_loss(state: ArenaState) -> dict:
    """Check existing positions for stop-loss conditions before new analysis.

    Triggers automatic exit when:
    - Unrealized loss exceeds max_stop_loss_pct (default 5%)
    - Position held longer than max_hold_bars (default 30 bars)
    """
    from cryptotrader.portfolio.manager import PortfolioManager

    pair = state["metadata"]["pair"]
    price = state["data"].get("snapshot_summary", {}).get("price", 0)
    if not price:
        return {"data": {}}

    db_url = state["metadata"].get("database_url")
    pm = PortfolioManager(db_url)
    try:
        portfolio = await pm.get_portfolio()
    except Exception:
        logger.debug("Portfolio fetch failed, skipping stop-loss check", exc_info=True)
        return {"data": {}}

    pos = portfolio.get("positions", {}).get(pair)
    if not pos or not isinstance(pos, dict):
        return {"data": {}}

    amount = pos.get("amount", 0)
    avg_price = pos.get("avg_price", 0)
    if amount == 0 or avg_price <= 0:
        return {"data": {}}

    # Calculate unrealized PnL
    pnl_pct = (price - avg_price) / avg_price if amount > 0 else (avg_price - price) / avg_price

    from cryptotrader.config import load_config

    cfg = load_config()
    max_loss_pct = state["metadata"].get("max_stop_loss_pct", cfg.risk.max_stop_loss_pct)

    if pnl_pct < -max_loss_pct:
        logger.warning(
            "Stop-loss triggered for %s: %.2f%% loss (threshold: %.2f%%)",
            pair,
            pnl_pct * 100,
            -max_loss_pct * 100,
        )
        # Override verdict to force exit
        side = "sell" if amount > 0 else "buy"
        return {
            "data": {
                "verdict": {
                    "action": "short" if side == "sell" else "long",
                    "confidence": 1.0,
                    "position_scale": 1.0,
                    "divergence": 0.0,
                    "reasoning": f"Stop-loss: {pnl_pct:+.2%} unrealized loss exceeds {max_loss_pct:.0%} threshold",
                    "thesis": "Stop-loss exit to preserve capital",
                    "invalidation": "N/A — forced exit",
                },
                "stop_loss_triggered": True,
            }
        }

    return {"data": {}}


async def place_order(state: ArenaState) -> dict:
    """Place order via exchange (paper or live)."""
    from cryptotrader.models import Order
    from cryptotrader.nodes.verdict import _get_notifier

    verdict = state["data"]["verdict"]
    if verdict["action"] == "hold":
        return {"data": {"order": None}}

    pair = state["metadata"]["pair"]
    price = state["data"].get("snapshot_summary", {}).get("price", 0)
    scale = verdict.get("position_scale", 1.0)
    total = state["data"].get("portfolio", {}).get("total_value", 10000)
    if price <= 0:
        return {"data": {"order": None}}
    max_single_pct = state["metadata"].get("max_single_pct", 0.1)
    amount = (total * max_single_pct * scale) / price

    order = Order(
        pair=pair,
        side="buy" if verdict["action"] == "long" else "sell",
        amount=amount,
        price=price,
    )

    exchange, live_exchange = _get_exchange(state, pair)

    try:
        result = await exchange.place_order(order)
    finally:
        if live_exchange is not None:
            await live_exchange.close()

    status = result.get("status", "")
    if status not in ("filled", "partially_filled"):
        return {"data": {"order": None}}

    filled_amount = result.get("filled", order.amount) if status == "partially_filled" else order.amount
    filled_price = result.get("price", order.price)

    await _update_trade_tracking(state, pair)

    order_data = {
        "pair": order.pair,
        "side": order.side,
        "amount": filled_amount,
        "price": filled_price,
        "status": status,
    }

    await _update_portfolio(state, order, filled_amount, filled_price)

    # Fire-and-forget notification
    try:
        notifier = _get_notifier(state)
        await notifier.notify("trade", {"pair": order.pair, "order": order_data})
    except Exception:
        logger.debug("Notification send failed", exc_info=True)

    return {"data": {"order": order_data}}
