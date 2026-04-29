"""Portfolio snapshot + equity-curve endpoints (FR-800/FR-801)."""

from __future__ import annotations

import asyncio
import logging
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


# ── Response models (data-model §1) ──


class PositionOut(BaseModel):
    pair: str
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


def _build_state(pair: str = "BTC/USDT") -> dict:
    """Minimum ArenaState shape for read_portfolio_from_exchange."""
    from cryptotrader.config import load_config

    config = load_config()
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
        logger.debug("equity snapshot read failed for sharpe", exc_info=True)
        return []


async def _load_commits(database_url: str | None) -> list:
    from cryptotrader.journal.store import JournalStore

    try:
        store = JournalStore(database_url)
        return await store.log(limit=1000)
    except Exception:
        logger.debug("journal log read failed for pnl stats", exc_info=True)
        return []


def _commit_pnl_stats(commits: list, cutoff_30d: datetime) -> tuple[int, float | None, float]:
    """Return (total_trades, win_rate_or_None, realized_pnl_30d)."""
    filled = [c for c in commits if getattr(c, "order", None) is not None]
    total = len(filled)
    win_rate: float | None = None
    if filled:
        wins = [c for c in filled if (c.pnl is not None and c.pnl > 0)]
        settled = [c for c in filled if c.pnl is not None]
        if settled:
            win_rate = round(len(wins) / len(settled), 4)
    realized = 0.0
    for c in filled:
        ts_dt = _coerce_timestamp(c.timestamp)
        if ts_dt is None or ts_dt < cutoff_30d or c.pnl is None:
            continue
        realized += float(c.pnl)
    return total, win_rate, round(realized, 2)


async def _compute_extras(database_url: str | None) -> dict[str, object]:
    """Derive (sharpe_90d, win_rate, total_trades, realized_pnl_30d).

    - **sharpe_90d**: mean/std of daily equity returns over last 90 days, annualised
      by sqrt(365). ``None`` when fewer than 30 daily samples.
    - **win_rate**: share of filled commits with positive ``pnl``; ``None`` when no
      filled trades exist.
    - **total_trades**: number of commits with a non-null ``order`` (executed).
    - **realized_pnl_30d**: sum of ``commit.pnl`` for commits in the last 30 days.
    """
    now = datetime.now(UTC)
    snaps = await _load_snapshots(database_url)
    sharpe = _sharpe_from_daily(_daily_last_equity(snaps, now - timedelta(days=90)))

    commits = await _load_commits(database_url)
    total, win_rate, realized_30d = _commit_pnl_stats(commits, now - timedelta(days=30))

    return {
        "sharpe_90d": sharpe,
        "win_rate": win_rate,
        "total_trades": total,
        "realized_pnl_30d": realized_30d,
    }


def _serialize_positions(raw_positions: dict) -> list[PositionOut]:
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
        out.append(
            PositionOut(
                pair=pair,
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

    try:
        live = await asyncio.wait_for(
            pm_mod.read_portfolio_from_exchange(cast("ArenaState", _build_state())),
            timeout=15.0,
        )
    except Exception:
        logger.debug("read_portfolio_from_exchange failed", exc_info=True)
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

        pnl_24h = float(await pm.get_daily_pnl())
        drawdown_raw = float(await pm.get_drawdown())
    except Exception as exc:
        logger.warning("Portfolio snapshot read failed: %s", exc)
        raise HTTPException(status_code=503, detail="Portfolio data unavailable") from exc

    extras = await _compute_extras(config.infrastructure.database_url)

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
