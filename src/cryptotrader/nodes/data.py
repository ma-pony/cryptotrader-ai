"""Data collection and verbal reinforcement nodes."""

from __future__ import annotations

import logging

from cryptotrader.state import ArenaState

logger = logging.getLogger(__name__)


def _calc_price_change_7d(snapshot) -> float | None:
    """Calculate 7-day price change from snapshot OHLCV data."""
    ohlcv = snapshot.market.ohlcv
    if ohlcv is None or ohlcv.empty:
        return None
    closes = ohlcv["close"].dropna().tolist()
    if len(closes) < 2:
        return None
    # Estimate bars per day from timestamps
    bars_per_day = 1
    if "timestamp" in ohlcv.columns:
        timestamps = ohlcv["timestamp"].dropna().tolist()
        if len(timestamps) >= 2:
            delta = abs(timestamps[-1] - timestamps[-2])
            bar_ms = delta.total_seconds() * 1000 if hasattr(delta, "total_seconds") else float(delta)
            if bar_ms > 0:
                bars_per_day = 86_400_000 / bar_ms
    bars_7d = int(7 * bars_per_day)
    if len(closes) <= bars_7d:
        return None
    past = closes[-(bars_7d + 1)]
    if past <= 0:
        return None
    return (closes[-1] - past) / past


async def collect_snapshot(state: ArenaState) -> dict:
    """Collect market snapshot or reuse pre-provided one (backtest)."""
    if state.get("data", {}).get("snapshot"):
        snapshot = state["data"]["snapshot"]
        summary = {
            "pair": snapshot.pair,
            "price": snapshot.market.ticker.get("last", 0),
            "funding_rate": snapshot.market.funding_rate,
            "volatility": snapshot.market.volatility,
            "orderbook_imbalance": snapshot.market.orderbook_imbalance,
        }
        # Add 7d price change for similarity matching
        price_change_7d = _calc_price_change_7d(snapshot)
        if price_change_7d is not None:
            summary["price_change_7d"] = price_change_7d
        return {"data": {"snapshot_summary": summary}}

    from cryptotrader.config import load_config
    from cryptotrader.data.snapshot import SnapshotAggregator

    pair = state["metadata"]["pair"]
    exchange_id = state["metadata"].get("exchange_id", "binance")
    timeframe = state["metadata"].get("timeframe", "1h")
    limit = state["metadata"].get("ohlcv_limit", 100)

    providers_cfg = load_config().providers
    agg = SnapshotAggregator(providers_cfg)
    snapshot = await agg.collect(pair, exchange_id, timeframe, limit)
    summary = {
        "pair": pair,
        "price": snapshot.market.ticker.get("last", 0),
        "funding_rate": snapshot.market.funding_rate,
        "volatility": snapshot.market.volatility,
        "orderbook_imbalance": snapshot.market.orderbook_imbalance,
    }
    price_change_7d = _calc_price_change_7d(snapshot)
    if price_change_7d is not None:
        summary["price_change_7d"] = price_change_7d
    return {"data": {"snapshot": snapshot, "snapshot_summary": summary}}


async def update_past_pnl(state: ArenaState) -> dict:
    """Back-fill PnL for recent trades that haven't been evaluated yet.

    Runs at the start of each cycle: looks up recent journal entries
    with orders but no PnL, compares entry price with current price.
    This closes the feedback loop so calibration has data to work with.
    """
    from cryptotrader.journal.store import JournalStore

    db_url = state["metadata"].get("database_url")
    store = JournalStore(db_url)
    current_price = state["data"].get("snapshot_summary", {}).get("price", 0)
    pair = state["metadata"].get("pair")
    if not current_price or not pair:
        return {"data": {}}

    try:
        commits = await store.log(limit=50, pair=pair)
        updated = 0
        for dc in commits:
            if dc.order is None:
                continue  # No trade was placed
            # Skip trades that have a fill_price — those are closed/realized
            if dc.pnl is not None and dc.fill_price is not None:
                continue  # Realized PnL already recorded with actual close price
            entry_price = dc.order.price
            if not entry_price or entry_price <= 0:
                continue
            # Calculate unrealized PnL (will be updated each cycle until position is closed)
            if dc.order.side == "buy":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            pnl_abs = pnl_pct * dc.order.amount * entry_price
            retro = f"Entry {entry_price:.2f} → Current {current_price:.2f} = {pnl_pct:+.2%} (unrealized)"
            await store.update_pnl(dc.hash, pnl_abs, retro)
            updated += 1
        if updated:
            logger.info("Updated PnL for %d past trades on %s", updated, pair)
    except Exception:
        logger.debug("PnL back-fill failed", exc_info=True)

    return {"data": {}}


async def verbal_reinforcement(state: ArenaState) -> dict:
    """Inject past experience + per-agent bias corrections + reflections into agent context."""
    import asyncio

    from cryptotrader.config import load_config
    from cryptotrader.journal.calibrate import (
        detect_biases,
        generate_per_agent_corrections,
        generate_verdict_calibration,
    )
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.learning.reflect import load_reflections, maybe_reflect
    from cryptotrader.learning.verbal import get_experience

    db_url = state["metadata"].get("database_url")
    store = JournalStore(db_url)
    summary = state["data"].get("snapshot_summary", {})
    experience = await get_experience(store, summary)

    # Phase 4D: Detect biases and generate per-agent corrections + verdict calibration
    agent_corrections: dict[str, str] = {}
    verdict_calibration = ""
    try:
        biases = await detect_biases(store, days=30)
        agent_corrections = generate_per_agent_corrections(biases)
        verdict_calibration = generate_verdict_calibration(biases)
    except Exception:
        logger.debug("Bias detection failed, continuing without calibration", exc_info=True)

    # Load existing agent reflections (fast SQLite read)
    agent_reflections: dict[str, str] = {}
    try:
        agent_reflections = await load_reflections()
    except Exception:
        logger.debug("Failed to load agent reflections", exc_info=True)

    # Maybe trigger background reflection (fire-and-forget, doesn't block trading)
    config = load_config()
    cycle_count = state["metadata"].get("cycle_count", 0)
    if config.reflection.enabled and cycle_count > 0:
        try:
            task = asyncio.ensure_future(maybe_reflect(store, cycle_count, config.reflection))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        except Exception:
            logger.debug("Failed to schedule reflection", exc_info=True)

    return {
        "data": {
            "experience": experience,
            "agent_corrections": agent_corrections,
            "verdict_calibration": verdict_calibration,
            "agent_reflections": agent_reflections,
        }
    }


def _build_trend_from_ohlcv(snapshot) -> dict | None:
    """Build trend context from snapshot OHLCV data."""
    ohlcv = snapshot.market.ohlcv
    if ohlcv is None or ohlcv.empty:
        return None
    closes = ohlcv["close"].dropna().tolist()
    if len(closes) < 2:
        return None
    current = closes[-1]
    ctx: dict = {"current_price": current}
    # Infer bar duration from timestamps to calculate lookback indices
    bars_per_day = 1  # default: assume daily bars
    if "timestamp" in ohlcv.columns:
        timestamps = ohlcv["timestamp"].dropna().tolist()
        if len(timestamps) >= 2:
            delta = abs(timestamps[-1] - timestamps[-2])
            bar_ms = delta.total_seconds() * 1000 if hasattr(delta, "total_seconds") else float(delta)
            if bar_ms > 0:
                bars_per_day = 86_400_000 / bar_ms
    for days, label in [(7, "7d"), (14, "14d"), (30, "30d")]:
        bars_back = int(days * bars_per_day)
        if len(closes) > bars_back:
            past = closes[-(bars_back + 1)]
            if past > 0:
                ctx[f"change_{label}"] = (current - past) / past
    # 30d high/low (or as much data as available)
    lookback_30 = int(30 * bars_per_day)
    n = min(lookback_30, len(closes))
    highs = ohlcv["high"].dropna().tolist()[-n:]
    lows = ohlcv["low"].dropna().tolist()[-n:]
    if highs and lows:
        ctx["high_30d"] = max(highs)
        ctx["low_30d"] = min(lows)
    return ctx


async def _build_position_from_portfolio(pair: str, price: float, db_url: str | None) -> dict:
    """Build position context from PortfolioManager."""
    from cryptotrader.portfolio.manager import PortfolioManager

    pm = PortfolioManager(db_url)
    try:
        portfolio = await pm.get_portfolio()
        pos = portfolio.get("positions", {}).get(pair, {})
        amount = pos.get("amount", 0)
        avg_price = pos.get("avg_price", 0)
        if amount == 0 or avg_price <= 0:
            return {"side": "flat"}
        return {
            "side": "long" if amount > 0 else "short",
            "entry_price": avg_price,
            "current_price": price,
        }
    except Exception:
        logger.debug("Portfolio fetch for position context failed", exc_info=True)
        return {"side": "flat"}


async def enrich_verdict_context(state: ArenaState) -> dict:
    """Build position_context and trend_context for the verdict node.

    This runs in ALL graph variants (full, lite, debate), ensuring the verdict AI
    always has consistent position awareness and price trend data.

    - trend_context: always built from snapshot OHLCV (authoritative)
    - position_context: uses pre-set value if provided (backtest), else reads PortfolioManager (live)
    """
    result: dict = {}
    snapshot = state["data"].get("snapshot")

    # Always build trend_context from snapshot OHLCV (single source of truth)
    if snapshot:
        trend = _build_trend_from_ohlcv(snapshot)
        if trend:
            result["trend_context"] = trend

    # Position context: respect caller-provided value (backtest), else read live portfolio
    if state["data"].get("position_context"):
        # Already set by caller (e.g., backtest script) — keep it
        pass
    else:
        pair = state["metadata"].get("pair", "")
        price = state["data"].get("snapshot_summary", {}).get("price", 0)
        db_url = state["metadata"].get("database_url")
        pos_ctx = await _build_position_from_portfolio(pair, price, db_url)
        result["position_context"] = pos_ctx

    return {"data": result}
