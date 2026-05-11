"""Portfolio snapshot + equity-curve endpoints (FR-800/FR-801)."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal, cast

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.routes._utils import coerce_timestamp as _coerce_timestamp  # backwards compat
from cryptotrader._compat import UTC

if TYPE_CHECKING:
    from cryptotrader.state import ArenaState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

_MAX_POINTS = 1000  # NFR-P-004

# OKX live-portfolio fetch tuning. Three knobs:
#   1. _OKX_FETCH_TIMEOUT_SEC: per-call budget. Healthy OKX takes ~500ms - 8s
#      depending on network (VPN tunnel adds ~5s). 8s is generous enough to
#      avoid spurious DB fallback while staying under React Query's idle
#      tolerance.
#   2. _OKX_FAIL_COOLDOWN_SEC: after a timeout/error, skip live read entirely
#      for this long (use DB fallback). Prevents dashboard from hanging
#      every poll when OKX is unreachable.
#   3. _OKX_CACHE_TTL_SEC: cache successful live result so the per-second
#      poll loop doesn't slam OKX with redundant calls. The whole purpose of
#      live read is up-to-date P&L; 30s freshness is more than enough.
_OKX_FETCH_TIMEOUT_SEC = 8.0
_OKX_FAIL_COOLDOWN_SEC = 60.0
_OKX_CACHE_TTL_SEC = 30.0
_OKX_LAST_FAIL_AT: float = 0.0
_OKX_LAST_OK_AT: float = 0.0
_OKX_LAST_OK_RESULT: dict | None = None


# ── Response models (data-model §1) ──


class PositionOut(BaseModel):
    pair: str  # ccxt canonical: "BTC/USDT" (spot) or "BTC/USDT:USDT" (perp)
    pair_display: str  # spec 013: human form, e.g. "BTC/USDT (perp)"
    market_type: Literal["spot", "swap", "future", "option"] = "spot"
    side: Literal["long", "short"] = "long"
    size: float
    avg_price: float
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    opened_at: str | None = None


class PortfolioSnapshotOut(BaseModel):
    equity: float
    cash: float
    positions: list[PositionOut]
    pnl_24h: float
    pnl_24h_pct: float
    drawdown: float  # ∈ [0, 1]
    updated_at: str
    # Alignment with frontend prototype (2026-04-24):
    sharpe_90d: float | None = None
    win_rate: float | None = None  # fraction ∈ [0, 1]
    total_trades: int = 0
    realized_pnl_30d: float = 0.0
    # Inception-to-date total return (current equity - first snapshot equity).
    # Both fields are 0.0 when no snapshot history exists yet.
    total_return: float = 0.0
    total_return_pct: float = 0.0
    # Mean realized PnL per filled trade (commits with non-null pnl).
    # None when no settled trades exist yet.
    avg_trade_pnl: float | None = None


class EquityPointOut(BaseModel):
    ts: str
    equity: float


class EquityCurveOut(BaseModel):
    range: Literal["24h", "7d", "30d", "all"]
    points: list[EquityPointOut]


# ── Helpers ──


def _compute_pnl_pct(equity: float, pnl_24h: float) -> float:
    """Return pnl_24h / baseline, guarding against zero/negative baseline."""
    baseline = equity - pnl_24h
    if baseline <= 0:
        return 0.0
    return pnl_24h / baseline


def _build_state(pair: str | None = None) -> dict:
    """Minimum ArenaState shape for read_portfolio_from_exchange.

    Spec 013 deep-review fix: when ``pair`` is None, derive from the first
    configured scheduler pair (canonical str). Hardcoded ``BTC/USDT`` was
    silently under-reporting equity on perp accounts because ``BTC/USDT``
    !=  ``BTC/USDT:USDT`` in the position dict.
    """
    from cryptotrader.config import load_config

    config = load_config()
    if pair is None:
        configured = list(getattr(config.scheduler, "pairs", []) or [])
        pair = configured[0].canonical() if configured else "BTC/USDT"
    return {
        "metadata": {"pair": pair, "engine": config.engine, "exchange_id": config.exchange_id},
        "data": {"snapshot_summary": {}},
    }


def _daily_last_equity(snaps: list[dict], cutoff: datetime) -> dict[str, float]:
    """Reduce snapshot list to {date_iso: last equity that day} since cutoff."""
    out: dict[str, float] = {}
    for s in snaps:
        ts = s.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if ts < cutoff:
            continue
        out[ts.date().isoformat()] = float(s.get("total_value", 0.0) or 0.0)
    return out


def _sharpe_from_daily(daily_last: dict[str, float]) -> float | None:
    """Annualised Sharpe from daily equity series; None when fewer than 30 samples."""
    from math import sqrt
    from statistics import mean, pstdev

    if len(daily_last) < 30:
        return None
    series = [daily_last[k] for k in sorted(daily_last.keys())]
    returns = [(series[i] - series[i - 1]) / series[i - 1] for i in range(1, len(series)) if series[i - 1] > 0]
    if not returns:
        return None
    sd = pstdev(returns)
    if sd <= 0:
        return None
    return round((mean(returns) / sd) * sqrt(365), 2)


async def _load_snapshots(database_url: str | None) -> list[dict]:
    from cryptotrader.portfolio.manager import PortfolioManager

    try:
        pm = PortfolioManager(database_url)
        return await pm.load_snapshots("default")
    except Exception:
        logger.info("equity snapshot read failed for sharpe", exc_info=True)
        return []


async def _load_commits(database_url: str | None) -> list:
    from cryptotrader.journal.store import JournalStore

    try:
        store = JournalStore(database_url)
        return await store.log(limit=1000)
    except Exception:
        logger.info("journal log read failed for pnl stats", exc_info=True)
        return []


def _commit_pnl_stats(
    commits: list,
    cutoff_30d: datetime,
    inception_cutoff: datetime | None = None,
) -> tuple[int, float | None, float, float | None, float]:
    """Return (total_trades, win_rate, realized_pnl_30d, avg_trade_pnl, realized_cumulative).

    Two filters applied so the trade-level metrics align with the snapshot-
    based ``total_return`` instead of double-counting (2026-05-07 design):

    (B) ``inception_cutoff``: drop commits older than the first portfolio
        snapshot. Pre-snapshot trades are already absorbed into the equity
        baseline used by ``total_return``; counting them again here
        creates a phantom "every trade is negative but total is positive"
        paradox.

    (C) ``action == "close"`` only: open-action commits get their ``pnl``
        column polluted by ``_update_one_commit_pnl`` with an unrealized
        snapshot at the cycle right after open. That value is calibration
        feedback for the LLM, NOT realized round-trip P&L. Real realized
        P&L lives on the matching close-action commit. Counting both
        double-counts each round-trip.

    ``cutoff_30d`` still applies on top for the 30-day realized window.
    """
    eligible = []
    for c in commits:
        if getattr(c, "order", None) is None:
            continue
        ts_dt = _coerce_timestamp(c.timestamp)
        if inception_cutoff is not None and (ts_dt is None or ts_dt < inception_cutoff):
            continue
        action = ((c.verdict.action if c.verdict else "") or "").lower()
        if action != "close":
            continue
        eligible.append(c)

    total = len(eligible)
    win_rate: float | None = None
    avg_trade_pnl: float | None = None
    if eligible:
        settled = [c for c in eligible if c.pnl is not None]
        if settled:
            wins = [c for c in settled if c.pnl > 0]
            win_rate = round(len(wins) / len(settled), 4)
            avg_trade_pnl = round(sum(float(c.pnl) for c in settled) / len(settled), 2)

    realized = 0.0
    realized_cumulative = 0.0
    for c in eligible:
        if c.pnl is None:
            continue
        pnl_val = float(c.pnl)
        # Inception-to-date trading PnL (used by total_return).
        realized_cumulative += pnl_val
        # 30d-window realized PnL (legacy field).
        ts_dt = _coerce_timestamp(c.timestamp)
        if ts_dt is None or ts_dt < cutoff_30d:
            continue
        realized += pnl_val

    return total, win_rate, round(realized, 2), avg_trade_pnl, round(realized_cumulative, 2)


def _inception_equity(snaps: list[dict]) -> float | None:
    """Return total_value of the earliest snapshot, or None when unavailable.

    Snapshots come from ``portfolio_snapshots`` (one row per scheduler cycle
    since 685ca52, 2026-04-30). The first row is treated as inception baseline
    for total-return calculation.
    """
    if not snaps:
        return None
    earliest = min(snaps, key=lambda s: s.get("timestamp") or "")
    val = float(earliest.get("total_value", 0.0) or 0.0)
    return val if val > 0 else None


def _inception_timestamp(snaps: list[dict]) -> datetime | None:
    """Return timestamp of the earliest snapshot for trade-stats inception cutoff."""
    if not snaps:
        return None
    earliest = min(snaps, key=lambda s: s.get("timestamp") or "")
    return _coerce_timestamp(earliest.get("timestamp"))


async def _compute_extras(
    database_url: str | None,
    current_equity: float,
    raw_positions: dict | None = None,
) -> dict[str, object]:
    """Derive (sharpe_90d, win_rate, total_trades, realized_pnl_30d, total_return, total_return_pct).

    - **sharpe_90d**: mean/std of daily equity returns over last 90 days, annualised
      by sqrt(365). ``None`` when fewer than 30 daily samples.
    - **win_rate**: share of close-action commits since inception with positive
      realized ``pnl``. ``None`` when no closes recorded.
    - **total_trades**: count of close-action (round-trip-completing) commits
      since inception. Open-action commits are excluded — their ``pnl`` is an
      unrealized snapshot, not a realized trade outcome.
    - **realized_pnl_30d**: sum of close-action realized ``pnl`` in the last
      30 days, also subject to the inception cutoff.
    - **total_return / total_return_pct**: *trading* PnL since inception:
      cumulative realized PnL (sum of all closed-trade ``pnl`` since the first
      portfolio snapshot) plus current unrealized PnL (mark-to-market on open
      positions). Pct denominator is the baseline (config.portfolio.initial_capital
      when set, otherwise inception equity).

    *Why not ``equity - baseline``?* The earlier formula ``current_equity -
    inception_equity`` silently included USDT deposits and withdrawals — a
    $3,500 user top-up showed as $3,500 of "总收益". Trader-grade dashboards
    measure trading P&L (closed + unrealized), independent of capital flows.
    See 2026-05-11 deep-review for the paradox this resolves
    ("17 trades each negative but total positive").
    """
    from cryptotrader.config import load_config

    cfg = load_config()
    now = datetime.now(UTC)
    snaps = await _load_snapshots(database_url)
    sharpe = _sharpe_from_daily(_daily_last_equity(snaps, now - timedelta(days=90)))

    commits = await _load_commits(database_url)
    inception_ts = _inception_timestamp(snaps)
    total, win_rate, realized_30d, avg_trade_pnl, realized_cumulative = _commit_pnl_stats(
        commits, now - timedelta(days=30), inception_cutoff=inception_ts
    )

    # Sum unrealized PnL across all current positions (mark-to-market).
    total_unrealized = 0.0
    for pos in (raw_positions or {}).values():
        if not isinstance(pos, dict):
            continue
        total_unrealized += float(pos.get("unrealized_pnl", 0.0) or 0.0)

    # Trading PnL (excludes deposits/withdrawals).
    total_return = realized_cumulative + total_unrealized

    # Pct denominator: baseline equity. Preference: explicit config > first snapshot.
    configured = float(cfg.portfolio.initial_capital or 0.0)
    baseline: float | None = configured if configured > 0 else _inception_equity(snaps)
    if baseline is not None and baseline > 0:
        total_return_pct = total_return / baseline
    else:
        total_return_pct = 0.0

    return {
        "sharpe_90d": sharpe,
        "win_rate": win_rate,
        "total_trades": total,
        "realized_pnl_30d": realized_30d,
        "total_return": round(total_return, 2),
        "total_return_pct": round(total_return_pct, 6),
        "avg_trade_pnl": avg_trade_pnl,
    }


def _serialize_positions(raw_positions: dict) -> list[PositionOut]:
    from cryptotrader.pair import Pair

    out: list[PositionOut] = []
    for pair, pos in (raw_positions or {}).items():
        if not isinstance(pos, dict):
            logger.debug("Skipping non-dict position for %s: %r", pair, type(pos).__name__)
            continue
        amount = float(pos.get("amount", 0.0) or 0.0)
        if amount == 0.0:
            continue
        avg_price = float(pos.get("avg_price", 0.0) or 0.0)
        side: Literal["long", "short"] = pos.get("side") or ("long" if amount > 0 else "short")
        unrealized = float(pos.get("unrealized_pnl", 0.0) or 0.0)
        cost_basis = abs(amount) * avg_price
        unrealized_pct = (unrealized / cost_basis) if cost_basis > 0 else 0.0
        # Spec 013: prefer DB-stored market_type when present (Phase 5 column);
        # fall back to deriving from pair via Pair.parse.
        market_type = pos.get("market_type")
        try:
            p_obj = Pair.parse(pair)
            display = p_obj.display()
            if not market_type:
                market_type = p_obj.market_type
        except (ValueError, NotImplementedError):
            display = pair
            market_type = market_type or "spot"
        out.append(
            PositionOut(
                pair=pair,
                pair_display=display,
                market_type=market_type,  # type: ignore[arg-type]
                side=side,
                size=amount,
                avg_price=avg_price,
                unrealized_pnl=unrealized,
                unrealized_pnl_pct=unrealized_pct,
                opened_at=pos.get("opened_at"),
            )
        )
    return out


# ── Routes ──


@router.get("/snapshot", response_model=PortfolioSnapshotOut)
async def get_portfolio_snapshot() -> PortfolioSnapshotOut:
    """Return current portfolio snapshot. Prefer live exchange over DB."""
    from cryptotrader.config import load_config
    from cryptotrader.portfolio import manager as pm_mod
    from cryptotrader.portfolio.manager import PortfolioManager

    config = load_config()
    pm = PortfolioManager(config.infrastructure.database_url)

    # 3s budget for live exchange read + 60s cooldown after a failure. Original
    # 15s with no cooldown was acceptable when OKX was reachable, but on networks
    # where OKX DNS is poisoned (e.g. mainland CN without VPN — see 2026-05-07
    # incident: 198.18.0.138 returned for www.okx.com) every snapshot poll hung
    # 15s, making the dashboard appear offline. The cooldown keeps subsequent
    # polls fast-path through DB until OKX is likely healthy again.
    now = time.monotonic()
    # Cache hit: a successful read in the last _OKX_CACHE_TTL_SEC.
    if _OKX_LAST_OK_RESULT is not None and now - _OKX_LAST_OK_AT < _OKX_CACHE_TTL_SEC:
        live = _OKX_LAST_OK_RESULT
    elif _OKX_LAST_FAIL_AT and now - _OKX_LAST_FAIL_AT < _OKX_FAIL_COOLDOWN_SEC:
        live = None  # cooldown after recent failure
    else:
        try:
            live = await asyncio.wait_for(
                pm_mod.read_portfolio_from_exchange(cast("ArenaState", _build_state())),
                timeout=_OKX_FETCH_TIMEOUT_SEC,
            )
            globals()["_OKX_LAST_FAIL_AT"] = 0.0
            globals()["_OKX_LAST_OK_AT"] = now
            globals()["_OKX_LAST_OK_RESULT"] = live
        except Exception:
            logger.info("read_portfolio_from_exchange timed out / failed; using DB", exc_info=True)
            globals()["_OKX_LAST_FAIL_AT"] = now
            live = None

    try:
        if live:
            cash = float(live.get("cash", 0.0))
            equity = float(live.get("total_value", cash))
            raw_positions = live.get("positions", {}) or {}
        else:
            portfolio = await pm.get_portfolio()
            cash = float(portfolio.get("cash", 0.0))
            equity = float(portfolio.get("total_value", 0.0))
            raw_positions = portfolio.get("positions", {}) or {}

        # get_daily_pnl returns None when no snapshot exists in today's UTC window;
        # surface as 0.0 in the API response (frontend cannot render null PnL cards).
        pnl_24h_raw = await pm.get_daily_pnl()
        pnl_24h = float(pnl_24h_raw) if pnl_24h_raw is not None else 0.0
        drawdown_raw = float(await pm.get_drawdown())
    except Exception as exc:
        logger.warning("Portfolio snapshot read failed: %s", exc)
        raise HTTPException(status_code=503, detail="Portfolio data unavailable") from exc

    extras = await _compute_extras(
        config.infrastructure.database_url,
        current_equity=equity,
        raw_positions=raw_positions,
    )

    return PortfolioSnapshotOut(
        equity=equity,
        cash=cash,
        positions=_serialize_positions(raw_positions),
        pnl_24h=pnl_24h,
        pnl_24h_pct=_compute_pnl_pct(equity, pnl_24h),
        drawdown=abs(drawdown_raw),
        updated_at=datetime.now(UTC).isoformat(),
        sharpe_90d=cast("float | None", extras["sharpe_90d"]),
        win_rate=cast("float | None", extras["win_rate"]),
        total_trades=int(cast("int", extras["total_trades"])),
        realized_pnl_30d=float(cast("float", extras["realized_pnl_30d"])),
        total_return=float(cast("float", extras["total_return"])),
        total_return_pct=float(cast("float", extras["total_return_pct"])),
        avg_trade_pnl=cast("float | None", extras["avg_trade_pnl"]),
    )


_RANGE_HOURS = {"24h": 24, "7d": 24 * 7, "30d": 24 * 30}


@router.get("/equity-curve", response_model=EquityCurveOut)
async def get_equity_curve(
    range: Literal["24h", "7d", "30d", "all"] = Query(...),
) -> EquityCurveOut:
    from cryptotrader.config import load_config
    from cryptotrader.portfolio.manager import PortfolioManager

    config = load_config()
    pm = PortfolioManager(config.infrastructure.database_url)
    snaps = await pm.load_snapshots("default")

    # Window filter
    if range != "all":
        cutoff = datetime.now(UTC) - timedelta(hours=_RANGE_HOURS[range])
        filtered = []
        for s in snaps:
            ts = s.get("timestamp")
            if ts is None:
                continue
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts >= cutoff:
                filtered.append({**s, "timestamp": ts})
        snaps = filtered

    # Cap at 1000 points (NFR-P-004) by uniform downsampling
    if len(snaps) > _MAX_POINTS:
        step = max(1, len(snaps) // _MAX_POINTS)
        snaps = snaps[::step][:_MAX_POINTS]

    points = [
        EquityPointOut(
            ts=(s["timestamp"].isoformat() if hasattr(s["timestamp"], "isoformat") else str(s["timestamp"])),
            equity=float(s.get("total_value", 0.0)),
        )
        for s in snaps
    ]
    return EquityCurveOut(range=range, points=points)
