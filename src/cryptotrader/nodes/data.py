"""Data collection and verbal reinforcement nodes."""

from __future__ import annotations

import logging

from cryptotrader.state import ArenaState  # noqa: TCH001

logger = logging.getLogger(__name__)


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
            if dc.pnl is not None:
                continue  # Already evaluated
            if dc.order is None:
                continue  # No trade was placed
            entry_price = dc.order.price
            if not entry_price or entry_price <= 0:
                continue
            # Calculate unrealized/realized PnL
            if dc.order.side == "buy":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            pnl_abs = pnl_pct * dc.order.amount * entry_price
            retro = f"Entry {entry_price:.2f} → Current {current_price:.2f} = {pnl_pct:+.2%}"
            await store.update_pnl(dc.hash, pnl_abs, retro)
            updated += 1
        if updated:
            logger.info("Updated PnL for %d past trades on %s", updated, pair)
    except Exception:
        logger.debug("PnL back-fill failed", exc_info=True)

    return {"data": {}}


async def verbal_reinforcement(state: ArenaState) -> dict:
    """Inject past experience + bias corrections into agent context."""
    from cryptotrader.journal.calibrate import detect_biases, generate_bias_correction, generate_verdict_calibration
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.learning.verbal import get_experience

    db_url = state["metadata"].get("database_url")
    store = JournalStore(db_url)
    summary = state["data"].get("snapshot_summary", {})
    experience = await get_experience(store, summary)

    # Phase 4D: Detect biases and generate corrections
    bias_correction = ""
    verdict_calibration = ""
    try:
        biases = await detect_biases(store, days=30)
        bias_correction = generate_bias_correction(biases)
        verdict_calibration = generate_verdict_calibration(biases)
    except Exception:
        logger.debug("Bias detection failed, continuing without calibration")

    if bias_correction:
        experience = f"{experience}\n\n{bias_correction}" if experience else bias_correction

    return {"data": {"experience": experience, "verdict_calibration": verdict_calibration}}
