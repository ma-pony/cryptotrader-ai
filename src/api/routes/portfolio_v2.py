"""Portfolio snapshot + equity-curve endpoints (FR-800/FR-801)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, cast

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

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
    return {"metadata": {"pair": pair}, "data": {"snapshot_summary": {}}}


def _serialize_positions(raw_positions: dict) -> list[PositionOut]:
    out: list[PositionOut] = []
    for pair, pos in (raw_positions or {}).items():
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
        live = await pm_mod.read_portfolio_from_exchange(cast("ArenaState", _build_state()))
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

    return PortfolioSnapshotOut(
        equity=equity,
        cash=cash,
        positions=_serialize_positions(raw_positions),
        pnl_24h=pnl_24h,
        pnl_24h_pct=_compute_pnl_pct(equity, pnl_24h),
        drawdown=abs(drawdown_raw),
        updated_at=datetime.now(UTC).isoformat(),
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
    snaps = await pm._load_snapshots("default")

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
