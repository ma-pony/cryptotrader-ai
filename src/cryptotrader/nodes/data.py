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
