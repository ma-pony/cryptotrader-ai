"""Order execution and portfolio update nodes."""

from __future__ import annotations

import logging
from typing import Any

from cryptotrader.portfolio.manager import read_portfolio_from_exchange
from cryptotrader.state import ArenaState
from cryptotrader.tracing import node_logger

# Re-export for callers that already import read_portfolio_from_exchange
# from this module (e.g. external scripts and tests written before task 8.1).
__all__ = ["read_portfolio_from_exchange"]

logger = logging.getLogger(__name__)

# Per-pair PaperExchange cache to prevent cross-pair balance contamination
_paper_exchanges: dict[str, Any] = {}

# Module-level cache for live exchanges (reuse connections, avoid repeated load_markets())
_live_exchanges: dict[str, Any] = {}


async def _get_exchange(state: ArenaState, pair: str):
    """Get exchange instance (paper or live) for the given pair."""
    from cryptotrader.execution.simulator import PaperExchange

    engine = state["metadata"].get("engine", "paper")
    if engine == "paper":
        if pair not in _paper_exchanges:
            balances, positions = await _load_balances_from_db(state)
            _paper_exchanges[pair] = PaperExchange(initial_balances=balances, initial_positions=positions)
        return _paper_exchanges[pair], None

    from cryptotrader.config import load_config
    from cryptotrader.execution.exchange import LiveExchange

    config = load_config()
    exchange_id = state["metadata"].get("exchange_id") or config.exchange_id
    if exchange_id in _live_exchanges:
        return _live_exchanges[exchange_id], None  # cached — don't close

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


async def _load_balances_from_db(
    state: ArenaState,
) -> tuple[dict[str, float] | None, dict[str, dict[str, float]] | None]:
    """Load saved balances and positions from DB to initialize PaperExchange.

    Returns (balances, positions) — both None if no saved state.
    """
    db_url = state["metadata"].get("database_url")
    if not db_url:
        return None, None
    try:
        from cryptotrader.portfolio.manager import PortfolioManager

        pm = PortfolioManager(db_url)
        portfolio = await pm.get_portfolio()
        cash = portfolio.get("cash", 0.0)
        positions = portfolio.get("positions", {})
        if cash == 0 and not positions:
            return None, None  # No saved state, use default
        balances: dict[str, float] = {"USDT": cash}
        pos_data: dict[str, dict[str, float]] = {}
        for pair, pos in positions.items():
            asset = pair.split("/")[0]
            amount = pos.get("amount", 0.0)
            if amount != 0:
                balances[asset] = amount
                avg_price = pos.get("avg_price", 0.0)
                pos_data[pair] = {"amount": amount, "avg_price": avg_price}
        return balances, pos_data if pos_data else None
    except Exception:
        logger.debug("Failed to load balances from DB for PaperExchange", exc_info=True)
        return None, None


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


async def _update_portfolio(
    state: ArenaState, order, _filled_amount: float, filled_price: float, exchange=None
) -> bool:
    """Sync portfolio from exchange after trade. Exchange is the source of truth."""
    db_url = state["metadata"].get("database_url")
    if exchange is None:
        logger.warning("No exchange available for portfolio sync")
        return False
    try:
        from cryptotrader.portfolio.manager import PortfolioManager

        pm = PortfolioManager(db_url)
        await _sync_portfolio_from_exchange(pm, exchange, order.pair, filled_price)
        return True
    except Exception:
        logger.warning("Portfolio sync failed for %s", order.pair, exc_info=True)
        return False


async def _sync_portfolio_from_exchange(pm, exchange, traded_pair: str, current_price: float) -> None:
    """Read actual balances from exchange and persist to DB.

    Exchange.get_balance() returns {asset: amount} — this is the single source
    of truth for both paper and live trading.
    """
    balances = await exchange.get_balance()
    cash = balances.pop("USDT", 0.0)

    # Persist cash
    await pm.update_cash("default", cash)

    # Persist each non-zero asset as a position
    # For the just-traded pair, use fill price. For others, keep existing avg_price.
    old_portfolio = await pm.get_portfolio()
    old_positions = old_portfolio.get("positions", {})

    total_pos_value = 0.0
    seen_pairs = set()

    for asset, amount in balances.items():
        if amount == 0:
            continue
        pair = f"{asset}/USDT"
        seen_pairs.add(pair)

        # Determine price: use fill price for just-traded pair, else keep old avg_price
        old_pos = old_positions.get(pair, {})
        if pair == traded_pair:
            price = current_price
        elif old_pos.get("avg_price", 0) > 0:
            price = old_pos["avg_price"]
        else:
            price = current_price  # new position, use current price

        await pm.update_position("default", pair, amount, price)
        total_pos_value += abs(amount) * price

    # Clear positions that are no longer on the exchange
    for pair in old_positions:
        if pair not in seen_pairs:
            await pm.update_position("default", pair, 0.0, 0.0)

    total = cash + total_pos_value
    await pm.snapshot("default", total, cash)


@node_logger()
async def check_stop_loss(state: ArenaState) -> dict:
    """Check existing positions for stop-loss conditions before new analysis.

    Triggers automatic exit when:
    - Unrealized loss exceeds max_stop_loss_pct (default 5%)
    """
    pair = state["metadata"]["pair"]
    price = state["data"].get("snapshot_summary", {}).get("price", 0)
    if not price:
        return {"data": {}}

    # Read position from exchange (source of truth)
    try:
        exchange_portfolio = await read_portfolio_from_exchange(state)
    except Exception:
        logger.debug("Exchange portfolio fetch failed, skipping stop-loss check", exc_info=True)
        return {"data": {}}

    if not exchange_portfolio:
        return {"data": {}}

    pos = exchange_portfolio.get("positions", {}).get(pair)
    if not pos or not isinstance(pos, dict):
        return {"data": {}}

    amount = pos.get("amount", 0)
    if amount == 0:
        return {"data": {}}

    avg_price = pos.get("avg_price", 0)
    if avg_price <= 0:
        return {"data": {}}

    # Calculate unrealized PnL from exchange position data
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


async def _build_close_order(pair: str, price: float, state: ArenaState):
    """Build an order to flatten an existing position (reads from exchange)."""
    from cryptotrader.models import Order

    try:
        exchange_portfolio = await read_portfolio_from_exchange(state)
        pos = (exchange_portfolio or {}).get("positions", {}).get(pair, {})
        pos_amount = pos.get("amount", 0)
    except Exception:
        logger.debug("Exchange portfolio fetch for close failed", exc_info=True)
        pos_amount = 0
    if pos_amount == 0:
        return None
    return Order(pair=pair, side="sell" if pos_amount > 0 else "buy", amount=abs(pos_amount), price=price)


async def _build_entry_order(verdict: dict, pair: str, price: float, state: ArenaState):
    """Build an order for new entry or add-to-position (加仓).

    Mirrors backtest logic: compute target_size from scale, subtract existing position.
    """
    from cryptotrader.config import load_config as _lc_exec
    from cryptotrader.models import Order

    scale = verdict.get("position_scale", 1.0)
    config = _lc_exec()

    # Read total portfolio value from exchange (source of truth)
    exchange_portfolio = await read_portfolio_from_exchange(state)
    total = exchange_portfolio["total_value"] if exchange_portfolio else 0
    if not total or total <= 0:
        logger.warning("No portfolio total_value available, cannot size order")
        return None
    max_single_pct = state["metadata"].get("max_single_pct", config.risk.position.max_single_pct)
    target_amount = (total * max_single_pct * scale) / price
    side = "buy" if verdict["action"] == "long" else "sell"

    # Check existing position from exchange — if already holding same direction, only order the delta
    existing_amount = 0.0
    try:
        pos = (exchange_portfolio or {}).get("positions", {}).get(pair, {})
        existing_amount = pos.get("amount", 0.0)
    except Exception:
        logger.debug("Position sizing from exchange failed", exc_info=True)

    if side == "buy" and existing_amount > 0:
        amount = max(0.0, target_amount - existing_amount)
    elif side == "sell" and existing_amount < 0:
        amount = max(0.0, target_amount - abs(existing_amount))
    else:
        amount = target_amount

    if amount < 1e-12:
        logger.info("Position already at or above target scale, skipping order")
        return None
    return Order(pair=pair, side=side, amount=amount, price=price)


@node_logger()
async def place_order(state: ArenaState) -> dict:
    """Place order via exchange (paper or live)."""
    from cryptotrader.nodes.verdict import _get_notifier

    verdict = state["data"]["verdict"]
    if verdict["action"] == "hold":
        return {"data": {"order": None}}

    pair = state["metadata"]["pair"]
    price = state["data"].get("snapshot_summary", {}).get("price", 0)
    if price <= 0:
        return {"data": {"order": None}}

    if verdict["action"] == "close":
        order = await _build_close_order(pair, price, state)
    else:
        order = await _build_entry_order(verdict, pair, price, state)
    if order is None:
        return {"data": {"order": None}}

    exchange, _ = await _get_exchange(state, pair)

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

    portfolio_ok = await _update_portfolio(state, order, filled_amount, filled_price, exchange)

    # Fire-and-forget notifications
    try:
        notifier = _get_notifier(state)
        await notifier.notify("trade", {"pair": order.pair, "order": order_data})
        if not portfolio_ok:
            await notifier.notify("portfolio_stale", {"pair": order.pair, "order": order_data})
    except Exception:
        logger.debug("Notification send failed", exc_info=True)

    return {"data": {"order": order_data}}
