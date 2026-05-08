"""Data collection and verbal reinforcement nodes."""

from __future__ import annotations

import hashlib
import json
import logging

from cryptotrader.state import ArenaState, get_pair
from cryptotrader.tracing import node_logger

logger = logging.getLogger(__name__)


def _compute_snapshot_hash(summary: dict) -> str:
    """Compute SHA256 hash over the four key snapshot fields.

    Only price, funding_rate, volatility, and orderbook_imbalance are hashed;
    these are the fields whose changes warrant fresh agent analysis.
    """
    key_fields = {
        "price": summary.get("price"),
        "funding_rate": summary.get("funding_rate"),
        "volatility": summary.get("volatility"),
        "orderbook_imbalance": summary.get("orderbook_imbalance"),
    }
    payload = json.dumps(key_fields, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


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


@node_logger()
async def init_decision(state: ArenaState) -> dict:
    """Initialise per-decision observability state.

    Currently binds a fresh :class:`~cryptotrader.llm.token_tracker.TokenLedger`
    to the ContextVar so every downstream ``create_llm()`` call accumulates
    token usage. Skipped in backtest mode to avoid resetting the ledger once
    per backtest step (see deep-review C-m2).
    """
    from cryptotrader.llm.token_tracker import start_ledger

    is_backtest = (state.get("metadata") or {}).get("backtest_mode")
    if not is_backtest:
        start_ledger()
    return {}


@node_logger()
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
        snapshot_hash = _compute_snapshot_hash(summary)
        return {"data": {"snapshot_summary": summary, "snapshot_hash": snapshot_hash}}

    from cryptotrader.config import load_config
    from cryptotrader.data.snapshot import SnapshotAggregator

    pair = get_pair(state).canonical()
    exchange_id = state["metadata"].get("exchange_id", "binance")
    timeframe = state["metadata"].get("timeframe", "1h")
    limit = state["metadata"].get("ohlcv_limit", 100)
    backtest_mode = state["metadata"].get("backtest_mode", False)

    cfg = load_config()
    agg = SnapshotAggregator(cfg.providers)

    adapter = None
    if cfg.mcp.enabled and not backtest_mode:
        try:
            from cryptotrader.mcp.adapter import MCPAdapter
            from cryptotrader.mcp.registry import MCPRegistry

            registry = MCPRegistry.from_config(cfg.mcp)
            adapter = MCPAdapter(registry, cfg.mcp)
        except Exception:
            logger.warning("MCP adapter creation failed, using direct providers", exc_info=True)

    snapshot = await agg.collect(
        pair,
        exchange_id,
        timeframe,
        limit,
        adapter=adapter,
        backtest_mode=backtest_mode,
    )
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
    snapshot_hash = _compute_snapshot_hash(summary)
    return {"data": {"snapshot": snapshot, "snapshot_summary": summary, "snapshot_hash": snapshot_hash}}


async def _update_one_commit_pnl(store, dc, current_price: float) -> bool:
    """Backfill realized PnL for an *unsettled close* commit.

    2026-05-07 contract change: only ``close`` actions are eligible. Previously
    this also "filled in" unrealized snapshots on open commits, which polluted
    realized-PnL aggregations (frontend avg_trade_pnl / win_rate / realized_30d)
    and produced the "every trade negative but total return positive" paradox.
    Open commits should stay ``pnl=None`` forever — their P&L is realized on the
    matching close commit instead.

    For close commits where ``state["data"]["realized_pnl"]`` was captured at
    execution time (see ``nodes/execution.py:_build_close_order``), the journal
    already stored the true realized P&L and this function exits early.

    The remaining (entry_price - current_price) calculation is a *fallback* for
    legacy commits that have no realized_pnl snapshot and that we cannot
    reconstruct without the original entry price. It is approximate.
    """
    if dc.order is None:
        return False  # No trade was placed
    # 2026-05-08: previously also required ``fill_price is not None`` here, but
    # close commits are persisted with top-level ``fill_price=None`` (the
    # actual fill price lives on the order row, not the commit). The compound
    # guard was therefore always falsy on close commits and the function ran
    # on every cycle, retroactively overwriting the realized PnL with a
    # mark-to-market estimate using the *current* price. Result: a SOL close
    # booked at +$90.56 on cycle T became +$54.17 on cycle T+1 because price
    # had drifted. Realized PnL is set once at close time (see
    # nodes/execution.py:_build_close_order writing state.realized_pnl) — once
    # ``pnl is not None`` it must never change.
    if dc.pnl is not None:
        return False  # Realized PnL already recorded — never recompute
    action = ((dc.verdict.action if dc.verdict else "") or "").lower()
    if action != "close":
        return False  # Open commits never get pnl set — see docstring
    entry_price = dc.order.price
    if not entry_price or entry_price <= 0:
        return False
    if dc.order.side == "buy":
        pnl_pct = (current_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - current_price) / entry_price
    pnl_abs = pnl_pct * dc.order.amount * entry_price
    retro = f"Entry {entry_price:.2f} → Current {current_price:.2f} = {pnl_pct:+.2%} (legacy fallback)"
    await store.update_pnl(dc.hash, pnl_abs, retro)
    return True


@node_logger()
async def update_past_pnl(state: ArenaState) -> dict:
    """Back-fill PnL for recent trades that haven't been evaluated yet.

    Runs at the start of each cycle: looks up recent journal entries
    with orders but no PnL, compares entry price with current price.
    This closes the feedback loop so calibration has data to work with.
    """
    from cryptotrader.journal.store import JournalStore

    # Skip in backtest mode — no real journal to update, and would corrupt live data
    if state["metadata"].get("backtest_mode"):
        return {"data": {}}

    db_url = state["metadata"].get("database_url")
    store = JournalStore(db_url)
    current_price = state["data"].get("snapshot_summary", {}).get("price", 0)
    try:
        pair = get_pair(state).canonical()
    except (KeyError, TypeError, ValueError):
        pair = None
    if not current_price or not pair:
        return {"data": {}}

    try:
        commits = await store.log(limit=50, pair=pair)
        updated = sum([await _update_one_commit_pnl(store, dc, current_price) for dc in commits])
        if updated:
            logger.info("Updated PnL for %d past trades on %s", updated, pair)
    except Exception:
        logger.warning("PnL back-fill failed", exc_info=True)

    return {"data": {}}


@node_logger()
async def verbal_reinforcement(state: ArenaState) -> dict:
    """Inject past experience + per-agent bias corrections + structured experience memory."""
    from cryptotrader.config import load_config
    from cryptotrader.journal.calibrate import (
        detect_biases,
        generate_per_agent_corrections,
        generate_verdict_calibration,
    )
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.learning.regime import tag_regime
    from cryptotrader.learning.verbal import get_experience

    config = load_config()
    db_url = state["metadata"].get("database_url")
    is_backtest = state["metadata"].get("backtest_mode", False)
    store = JournalStore(db_url, backtest_mode=is_backtest)
    summary = state["data"].get("snapshot_summary", {})

    # Tag current regime
    regime_tags = tag_regime(summary, config.experience.regime_thresholds)

    # Skip experience injection, bias detection, and reflection in backtest mode
    # to prevent look-ahead bias (live journal contains future data relative to backtest candle)
    historical_cases: list = []
    agent_corrections: dict[str, str] = {}
    verdict_calibration = ""

    if not is_backtest:
        # Fetch historical cases (regime-aware)
        historical_cases = await get_experience(
            store,
            summary,
            regime_tags=regime_tags,
            thresholds=config.experience.regime_thresholds,
        )

        # Detect biases and generate per-agent corrections + verdict calibration
        try:
            biases = await detect_biases(store, days=30)
            agent_corrections = generate_per_agent_corrections(biases)
            verdict_calibration = generate_verdict_calibration(biases)
        except Exception:
            logger.warning("Bias detection failed, continuing without calibration", exc_info=True)

        # Skills injection is handled by SkillsInjectionMiddleware in ToolAgent.analyze().
        # Background distillation is triggered by run_reflection() node (nodes/reflection.py).

    return {
        "data": {
            "regime_tags": regime_tags,
            "historical_cases": historical_cases,
            "agent_corrections": agent_corrections,
            "verdict_calibration": verdict_calibration,
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

    # ATR(14) on the snapshot timeframe — average true range over last 14 bars,
    # used by the verdict prompt + server-side guardrail to enforce stops wider
    # than typical noise. true_range = max(high-low, |high-prev_close|, |low-prev_close|).
    # Falls back to None when fewer than 15 bars are available so callers can
    # degrade gracefully.
    closes_full = ohlcv["close"].dropna().tolist()
    highs_full = ohlcv["high"].dropna().tolist()
    lows_full = ohlcv["low"].dropna().tolist()
    if len(closes_full) >= 15 and len(highs_full) >= 15 and len(lows_full) >= 15:
        trs = []
        for i in range(-14, 0):
            h = highs_full[i]
            lo = lows_full[i]
            pc = closes_full[i - 1]
            trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
        if trs:
            ctx["atr_14"] = sum(trs) / len(trs)

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
            "amount": abs(amount),
        }
    except Exception:
        logger.warning("Portfolio fetch for position context failed", exc_info=True)
        return {"side": "flat"}


@node_logger()
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
        try:
            pair = get_pair(state).canonical()
        except (KeyError, TypeError, ValueError):
            pair = ""
        price = state["data"].get("snapshot_summary", {}).get("price", 0)
        db_url = state["metadata"].get("database_url")
        pos_ctx = await _build_position_from_portfolio(pair, price, db_url)
        result["position_context"] = pos_ctx

    return {"data": result}
