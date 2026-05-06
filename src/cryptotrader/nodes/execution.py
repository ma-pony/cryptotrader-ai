"""Order execution and portfolio update nodes."""

from __future__ import annotations

import logging
from typing import Any

from cryptotrader.portfolio.manager import read_portfolio_from_exchange
from cryptotrader.state import ArenaState, get_pair
from cryptotrader.tracing import node_logger

# Re-export for callers that already import read_portfolio_from_exchange
# from this module (e.g. external scripts and tests written before task 8.1).
__all__ = ["read_portfolio_from_exchange"]

logger = logging.getLogger(__name__)

# Per-pair PaperExchange cache to prevent cross-pair balance contamination.
# Key: (pair, is_backtest) tuple. Call clear_paper_exchanges() between backtest runs.
_paper_exchanges: dict[tuple[str, bool], Any] = {}

# Module-level cache for live exchanges (reuse connections, avoid repeated load_markets())
_live_exchanges: dict[str, Any] = {}


def clear_paper_exchanges(*, backtest_only: bool = True) -> None:
    """Remove cached PaperExchange instances. Called between backtest runs."""
    if backtest_only:
        keys = [k for k in _paper_exchanges if k[1] is True]
        for k in keys:
            _paper_exchanges.pop(k, None)
    else:
        _paper_exchanges.clear()


async def close_live_exchanges() -> None:
    """Close all cached live exchange connections. Call on graceful shutdown."""
    for exchange_id, exchange in list(_live_exchanges.items()):
        try:
            await exchange.close()
            logger.info("Closed live exchange: %s", exchange_id)
        except Exception:
            logger.info("Failed to close exchange %s", exchange_id, exc_info=True)
    _live_exchanges.clear()


async def _get_exchange(state: ArenaState, pair: str):
    """Get exchange instance (paper or live) for the given pair."""
    from cryptotrader.execution.simulator import PaperExchange

    engine = state["metadata"].get("engine", "paper")
    if engine == "paper":
        # Key by (pair, backtest_mode) to prevent cross-contamination between
        # live paper trading and backtest runs sharing the same process.
        is_backtest = state["metadata"].get("backtest_mode", False)
        cache_key = (pair, is_backtest)
        if cache_key not in _paper_exchanges:
            balances, positions = await _load_balances_from_db(state)
            _paper_exchanges[cache_key] = PaperExchange(initial_balances=balances, initial_positions=positions)
        return _paper_exchanges[cache_key], None

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
        leverage=creds.leverage,
        margin_mode=creds.margin_mode,
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
        from cryptotrader.pair import Pair
        from cryptotrader.portfolio.manager import PortfolioManager

        pm = PortfolioManager(db_url)
        portfolio = await pm.get_portfolio()
        cash = portfolio.get("cash", 0.0)
        positions = portfolio.get("positions", {})
        if cash == 0 and not positions:
            return None, None  # No saved state, use default
        balances: dict[str, float] = {"USDT": cash}
        pos_data: dict[str, dict[str, float]] = {}
        for pair_str, pos in positions.items():
            try:
                p_obj = Pair.parse(pair_str)
            except ValueError:
                logger.debug("Invalid pair %s in DB; skipping for PaperExchange seed", pair_str)
                continue
            # PaperExchange tracks asset balances ("BTC", "USDT"); only spot
            # positions map cleanly to that model. Derivatives are margin-
            # denominated and don't survive the asset-balance translation.
            if p_obj.market_type != "spot":
                continue
            amount = pos.get("amount", 0.0)
            if amount != 0:
                balances[p_obj.base] = amount
                avg_price = pos.get("avg_price", 0.0)
                pos_data[pair_str] = {"amount": amount, "avg_price": avg_price}
        return balances, pos_data if pos_data else None
    except Exception:
        logger.warning("Failed to load balances from DB for PaperExchange", exc_info=True)
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


# Sub-threshold residue exchanges leave behind after a close (production
# observation: ETH=2.91e-07, BTC=3.19e-09 after successful flatten).
# Persisting these creates phantom positions that the orphan sweep won't
# clear — the pair is still "seen" by the exchange, just essentially zero.
# 1e-6 is comfortably below any economically meaningful position size for
# any asset we trade (0.000001 BTC ≈ $0.08, 0.000001 ETH ≈ $0.002).
_DUST_AMOUNT_THRESHOLD = 1e-6


def _is_dust(amount: float) -> bool:
    return abs(amount) < _DUST_AMOUNT_THRESHOLD


async def _get_market_price(exchange: Any, pair: str) -> float:
    """Fetch live ticker price for ``pair`` via the exchange.

    Returns 0.0 when the exchange does not expose ``fetch_ticker`` (e.g.
    ``PaperExchange``) or the call fails. Callers MUST treat 0.0 as
    "unknown" — never substitute another pair's price (production bug
    2026-04-30: BTC trade price was being assigned to ETH/OKB rows when
    cost basis was missing, inflating total_value to ~$7.3M).
    """
    fetcher = getattr(exchange, "fetch_ticker", None)
    if fetcher is None:
        return 0.0
    try:
        ticker = await fetcher(pair)
    except Exception:
        logger.warning("fetch_ticker failed for %s; writing avg_price=0", pair, exc_info=True)
        return 0.0
    if not isinstance(ticker, dict):
        return 0.0
    last = ticker.get("last") or ticker.get("close")
    try:
        return float(last) if last is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


async def _sync_spot_from_balances(
    pm,
    balances: dict[str, float],
    old_positions: dict[str, dict],
    traded_pair: str,
    current_price: float,
    exchange: Any = None,
) -> tuple[float, set[str]]:
    """Persist spot positions from {asset: amount} balances. Returns (value, seen)."""
    total = 0.0
    seen: set[str] = set()
    for asset, amount in balances.items():
        # Treat dust (post-close residue like 2.91e-07) as zero — don't add to
        # ``seen`` so the orphan sweep can clear any stale DB row for it.
        if _is_dust(amount):
            continue
        pair = f"{asset}/USDT"
        seen.add(pair)
        if pair == traded_pair:
            price = current_price
        else:
            old_avg = float(old_positions.get(pair, {}).get("avg_price", 0) or 0.0)
            price = old_avg if old_avg > 0 else await _get_market_price(exchange, pair)
        await pm.update_position("default", pair, amount, price)
        total += abs(amount) * price
    return total, seen


async def _sync_derivatives_from_positions(
    pm,
    derivs: dict[str, dict],
    traded_pair: str,
    current_price: float,
    exchange: Any = None,
) -> tuple[float, set[str]]:
    """Persist non-spot positions from get_positions().

    Returns (equity_contribution, seen). The contribution is the sum of
    unrealized PnL across open derivative positions — NOT notional. Notional
    is not an asset; the margin is already in cash. See companion comment in
    portfolio.manager.read_portfolio_from_exchange.
    """
    from cryptotrader.pair import Pair

    total = 0.0
    seen: set[str] = set()
    for pair, pos in (derivs or {}).items():
        try:
            p_obj = Pair.parse(pair)
        except ValueError:
            continue
        if p_obj.market_type == "spot":
            continue
        amount = pos.get("amount", 0.0)
        # See ``_DUST_AMOUNT_THRESHOLD`` — same dust handling as spot path so
        # closed perps don't linger as 1e-09-sized phantom positions.
        if _is_dust(amount):
            continue
        seen.add(pair)
        if pair == traded_pair:
            price = current_price
        else:
            entry = float(pos.get("avg_price", 0.0) or 0.0)
            price = entry if entry > 0 else await _get_market_price(exchange, pair)
        await pm.update_position("default", pair, amount, price)
        total += float(pos.get("unrealized_pnl", 0.0) or 0.0)
    return total, seen


async def _sweep_orphaned_positions(
    pm,
    old_positions: dict[str, dict],
    seen_pairs: set[str],
    *,
    derivatives_observed: bool,
) -> None:
    """Zero out DB rows the exchange no longer reports.

    Skip non-spot rows when ``derivatives_observed`` is False — we cannot
    distinguish "position closed" from "get_positions transiently failed".
    """
    from cryptotrader.pair import Pair

    for pair in old_positions:
        if pair in seen_pairs:
            continue
        if not derivatives_observed:
            try:
                if Pair.parse(pair).market_type != "spot":
                    continue
            except ValueError:
                continue
        await pm.update_position("default", pair, 0.0, 0.0)


async def _sync_portfolio_from_exchange(pm, exchange, traded_pair: str, current_price: float) -> None:
    """Read balances + positions from exchange and persist to DB.

    Spot derives from ``get_balance()``; derivatives from ``get_positions()``
    (keyed by ccxt unified symbol per spec 013).
    """
    balances = await exchange.get_balance()
    cash = balances.pop("USDT", 0.0)
    await pm.update_cash("default", cash)

    old_portfolio = await pm.get_portfolio()
    old_positions = old_portfolio.get("positions", {})

    spot_value, spot_seen = await _sync_spot_from_balances(
        pm, balances, old_positions, traded_pair, current_price, exchange=exchange
    )

    # Track success explicitly: an empty positions list is legitimate (no open
    # derivatives), but an exception is NOT — sweeping perp rows on a transient
    # get_positions() failure re-introduces the close-on-flat production bug
    # via a different code path (deep-review correctness FINDING-1).
    derivs_success = True
    derivs: dict[str, dict] = {}
    try:
        derivs = await exchange.get_positions()
    except Exception:
        logger.warning(
            "exchange.get_positions() failed during portfolio sync — skipping derivative sweep",
            exc_info=True,
        )
        derivs_success = False
    deriv_value, deriv_seen = await _sync_derivatives_from_positions(
        pm, derivs, traded_pair, current_price, exchange=exchange
    )

    seen = spot_seen | deriv_seen
    await _sweep_orphaned_positions(
        pm,
        old_positions,
        seen,
        derivatives_observed=derivs_success,
    )

    await pm.snapshot("default", cash + spot_value + deriv_value, cash)


@node_logger()
async def check_stop_loss(state: ArenaState) -> dict:
    """Check existing positions for stop-loss conditions before new analysis.

    Triggers automatic exit when:
    - Unrealized loss exceeds max_stop_loss_pct (default 5%)
    """

    pair = get_pair(state).canonical()
    price = state["data"].get("snapshot_summary", {}).get("price", 0)
    if not price:
        return {"data": {}}

    # Read position from exchange (source of truth)
    try:
        exchange_portfolio = await read_portfolio_from_exchange(state)
    except Exception:
        logger.warning("Exchange portfolio fetch failed, skipping stop-loss check", exc_info=True)
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
        logger.warning("Exchange portfolio fetch for close failed", exc_info=True)
        pos_amount = 0
    if pos_amount == 0:
        return None
    return Order(pair=pair, side="sell" if pos_amount > 0 else "buy", amount=abs(pos_amount), price=price)


async def _build_entry_order(verdict: dict, pair: str, price: float, state: ArenaState):
    """Build an order for new entry or add-to-position (加仓).

    Mirrors backtest logic: compute target_size from scale, subtract existing position.
    Returns None when the order cannot be built; reason is stashed at
    ``state["data"]["execution_error"]`` for downstream journaling (rule 3 of
    docs/logging-conventions.md).
    """
    from cryptotrader.config import load_config as _lc_exec
    from cryptotrader.models import Order
    from cryptotrader.pair import Pair

    scale = verdict.get("position_scale", 1.0)
    config = _lc_exec()

    # Read total portfolio value from exchange (source of truth)
    exchange_portfolio = await read_portfolio_from_exchange(state)
    total = exchange_portfolio["total_value"] if exchange_portfolio else 0
    if not total or total <= 0:
        logger.warning("No portfolio total_value available, cannot size order")
        state["data"]["execution_error"] = "no_portfolio: total_value unavailable"
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
        logger.warning("Position sizing from exchange failed", exc_info=True)

    # Pre-flight: spot accounts cannot short without inventory. Skip the
    # whole exchange round-trip + ValueError dance when the math is obvious.
    try:
        market_type = Pair.parse(pair).market_type
    except (ValueError, NotImplementedError):
        market_type = "spot"
    if market_type == "spot" and side == "sell" and existing_amount <= 0:
        logger.info(
            "Cannot short on spot without inventory: %s holdings=%g — skipping order",
            pair,
            existing_amount,
        )
        state["data"]["execution_error"] = (
            f"spot_short_no_inventory: cannot sell {pair} on spot — current holdings={existing_amount:g}"
        )
        return None

    if side == "buy" and existing_amount > 0:
        amount = max(0.0, target_amount - existing_amount)
    elif side == "sell" and existing_amount < 0:
        amount = max(0.0, target_amount - abs(existing_amount))
    else:
        amount = target_amount

    if amount < 1e-12:
        logger.info("Position already at or above target scale, skipping order")
        state["data"]["execution_error"] = "position_at_target: no delta to order"
        return None
    return Order(pair=pair, side=side, amount=amount, price=price)


@node_logger()
async def place_order(state: ArenaState) -> dict:
    """Place order via exchange (paper or live)."""
    from cryptotrader.nodes.verdict import _get_notifier

    verdict = state["data"]["verdict"]
    if verdict["action"] == "hold":
        return {"data": {"order": None}}

    pair = get_pair(state).canonical()
    price = state["data"].get("snapshot_summary", {}).get("price", 0)
    if price <= 0:
        return {"data": {"order": None}}

    if verdict["action"] == "close":
        order = await _build_close_order(pair, price, state)
    else:
        order = await _build_entry_order(verdict, pair, price, state)
    if order is None:
        # Surface the pre-flight bail reason via risk_gate so it shows up in
        # the journal commit and on /decisions as reject_reason. Risk gate
        # itself passed earlier — we override here because the cycle ended
        # in execution failure, which is what the user needs to see.
        err = state["data"].pop("execution_error", None) or "order_skipped: see logs"
        return {
            "data": {
                "order": None,
                "risk_gate": {
                    "passed": False,
                    "rejected_by": "execution_skipped",
                    "reason": err,
                },
            }
        }

    exchange, _ = await _get_exchange(state, pair)

    from cryptotrader.execution.order import OrderManager
    from cryptotrader.models import OrderStatus

    logger.info("Placing order: %s %s %.6f @ %.2f", order.side, pair, order.amount, order.price)
    om = OrderManager()
    import time as _time

    from cryptotrader.metrics import get_metrics_collector

    exec_t0 = _time.monotonic()
    order, result = await om.place(order, exchange)
    # Histogram observation populates execution_p50_ms / p95_ms in
    # /api/metrics/summary. Engine label = paper / live.
    get_metrics_collector().observe_execution_latency(
        engine=state["metadata"].get("engine", "paper"),
        ms=(_time.monotonic() - exec_t0) * 1000.0,
    )

    status = order.status
    if status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
        err_type = result.get("error_type")
        err_msg = result.get("error_msg")
        reason = f"{err_type}: {err_msg}" if err_type and err_msg else (err_msg or err_type or f"status={status}")
        logger.warning("Order not filled: %s %s status=%s reason=%s", order.side, pair, status, reason)
        return {
            "data": {
                "order": None,
                "risk_gate": {
                    "passed": False,
                    "rejected_by": "execution_failed",
                    "reason": reason,
                },
            }
        }

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
        logger.info("Notification send failed", exc_info=True)

    return {"data": {"order": order_data}}
