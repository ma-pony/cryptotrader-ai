"""Order execution and portfolio update nodes."""

from __future__ import annotations

import logging
from typing import Any

from cryptotrader.state import ArenaState

logger = logging.getLogger(__name__)

# Per-pair PaperExchange cache to prevent cross-pair balance contamination
_paper_exchanges: dict[str, Any] = {}

# Module-level cache for live exchanges (reuse connections, avoid repeated load_markets())
_live_exchanges: dict[str, Any] = {}


def _get_exchange(state: ArenaState, pair: str):
    """Get exchange instance (paper or live) for the given pair."""
    from cryptotrader.execution.simulator import PaperExchange

    engine = state["metadata"].get("engine", "paper")
    if engine == "paper":
        if pair not in _paper_exchanges:
            _paper_exchanges[pair] = PaperExchange()
        return _paper_exchanges[pair], None

    from cryptotrader.config import load_config
    from cryptotrader.execution.exchange import LiveExchange

    exchange_id = state["metadata"].get("exchange_id", "binance")
    if exchange_id in _live_exchanges:
        return _live_exchanges[exchange_id], None  # cached — don't close

    config = load_config()
    creds = config.exchanges.get(exchange_id)
    if creds is None or not creds.api_key or not creds.secret:
        raise RuntimeError(
            f"No credentials configured for exchange '{exchange_id}'. "
            f"Set api_key/secret in config/local.toml under [exchanges.{exchange_id}]"
        )

    live_exchange = LiveExchange(
        exchange_id,
        creds.api_key,
        creds.secret,
        sandbox=creds.sandbox,
        passphrase=creds.passphrase,
    )
    _live_exchanges[exchange_id] = live_exchange
    return live_exchange, None  # cached — don't close


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


async def _update_portfolio(state: ArenaState, order, filled_amount: float, filled_price: float) -> bool:
    """Update portfolio after successful trade. Returns True on success."""
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
        return True
    except Exception:
        logger.warning("Portfolio write-back failed for %s", pair, exc_info=True)
        return False


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
        return {
            "data": {
                "verdict": {
                    "action": "close",
                    "confidence": 1.0,
                    "position_scale": 0.0,
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
    from cryptotrader.portfolio.manager import PortfolioManager

    verdict = state["data"]["verdict"]
    if verdict["action"] == "hold":
        return {"data": {"order": None}}

    pair = state["metadata"]["pair"]
    price = state["data"].get("snapshot_summary", {}).get("price", 0)
    if price <= 0:
        return {"data": {"order": None}}

    # Handle close action — flatten the current position
    if verdict["action"] == "close":
        db_url = state["metadata"].get("database_url")
        pm = PortfolioManager(db_url)
        try:
            portfolio = await pm.get_portfolio()
            pos = portfolio.get("positions", {}).get(pair, {})
            pos_amount = pos.get("amount", 0)
        except Exception:
            logger.debug("Portfolio fetch for close failed", exc_info=True)
            pos_amount = 0
        if pos_amount == 0:
            return {"data": {"order": None}}
        order = Order(
            pair=pair,
            side="sell" if pos_amount > 0 else "buy",
            amount=abs(pos_amount),
            price=price,
        )
    else:
        scale = verdict.get("position_scale", 1.0)
        total = state["data"].get("portfolio", {}).get("total_value", 10000)
        max_single_pct = state["metadata"].get("max_single_pct", 0.1)
        amount = (total * max_single_pct * scale) / price
        order = Order(
            pair=pair,
            side="buy" if verdict["action"] == "long" else "sell",
            amount=amount,
            price=price,
        )

    exchange, _ = _get_exchange(state, pair)

    from cryptotrader.execution.order import OrderManager
    from cryptotrader.models import OrderStatus

    logger.info("Placing order: %s %s %.6f @ %.2f", order.side, pair, order.amount, order.price)
    om = OrderManager()
    order, result = await om.place(order, exchange)

    status = order.status
    if status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
        logger.warning("Order not filled: %s %s status=%s", order.side, pair, status)
        return {"data": {"order": None}}

    filled_amount = result.get("filled", order.amount) if status == OrderStatus.PARTIALLY_FILLED else order.amount
    filled_price = result.get("price", order.price)
    logger.info(
        "Order filled: %s %s %.6f @ %.2f (status=%s)",
        order.side,
        pair,
        filled_amount,
        filled_price,
        status.value if hasattr(status, "value") else status,
    )

    await _update_trade_tracking(state, pair)

    order_data = {
        "pair": order.pair,
        "side": order.side,
        "amount": filled_amount,
        "price": filled_price,
        "status": status.value if hasattr(status, "value") else str(status),
    }

    portfolio_ok = await _update_portfolio(state, order, filled_amount, filled_price)

    # Fire-and-forget notifications
    try:
        notifier = _get_notifier(state)
        await notifier.notify("trade", {"pair": order.pair, "order": order_data})
        if not portfolio_ok:
            await notifier.notify("portfolio_stale", {"pair": order.pair, "order": order_data})
    except Exception:
        logger.debug("Notification send failed", exc_info=True)

    return {"data": {"order": order_data}}
