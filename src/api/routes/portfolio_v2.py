"""Portfolio snapshot + equity-curve endpoints (FR-800/FR-801)."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from api.routes._utils import coerce_timestamp as _coerce_timestamp  # backwards compat
from cryptotrader._compat import UTC

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


class PnlBreakdown(BaseModel):
    """Decompose a window's equity change into 4 fundamental buckets.

    Identity:  delta = realized + funding + fees + unrealized_delta (+ residual)

    - `realized`         : sum of close-action commit PnL in window (journal).
    - `funding`          : net perp funding payments received/paid (OKX history).
                           Positive = received (short on positive funding,
                           or long on negative funding).
    - `fees`              : maker/taker fees paid on trades in window
                           (always negative for fee outflow; sign-flipped here
                           so a $15 fee shows as -$15).
    - `unrealized_delta` : derived = delta - (realized + funding + fees).
                           Captures Δunrealized on open positions + any minor
                           residuals (e.g. capped fetch_my_trades pagination,
                           micro-adjustments). Not a separate exchange query.

    Fields with `_known=False` indicate the exchange call failed or paper-mode
    is on; the UI should de-emphasize those buckets but still show realized
    (which always comes from the local journal).
    """

    window: str  # "24h" | "7d" | "30d"
    delta: float
    realized: float
    funding: float
    fees: float
    unrealized_delta: float
    exchange_data_available: bool = True


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
    # 24h / 7d PnL attribution for the dashboard breakdown card.
    pnl_breakdowns: list[PnlBreakdown] = []


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


def _configured_pair(config, pair: str | None = None) -> str:
    if pair is None:
        configured = list(getattr(config.scheduler, "pairs", []) or [])
        pair = str(configured[0]) if configured else "BTC/USDT"
    return pair


async def _read_live_portfolio(request: Request, config, pair: str | None = None) -> dict | None:
    """Read one configured execution book through the shared Runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or runtime.cycle is None:
        return None

    from cryptotrader.pair import Pair

    book = next((item for item in runtime.snapshot.document.execution.books if item.enabled), None)
    if book is None:
        return None
    snapshot = await runtime.cycle.portfolios.read(
        book,
        runtime.sessions,
        Pair.parse(_configured_pair(config, pair)),
    )
    positions = {
        item.position.pair.canonical(): {
            "amount": float(item.position.signed_amount),
            "side": "long" if item.position.signed_amount > 0 else "short",
            "avg_price": float(item.position.entry_price or 0),
            "unrealized_pnl": 0.0,
        }
        for item in snapshot.connections
        if item.position.signed_amount != 0
    }
    return {
        "total_value": float(snapshot.total_equity),
        "cash": float(sum((sum(item.balances.values()) for item in snapshot.connections), 0)),
        "positions": positions,
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


async def _load_cycles(database_url: str | None, journal_store=None) -> list:
    if journal_store is None:
        return []
    # The strict multi-venue execution DTO does not yet carry realized-PnL
    # attribution. Task 12 will restore these derived statistics from the new
    # fill ledger instead of reading the removed single-exchange Journal.
    return []


def _cycle_realized_pnl(cycle: Any) -> float:
    execution = cycle.execution_result or {}
    realized = float(execution.get("realized_pnl", 0.0) or 0.0)
    for order in execution.get("orders", []) or []:
        raw = order.get("raw") or {}
        realized += float(raw.get("realized_pnl", raw.get("pnl", 0.0)) or 0.0)
    return realized


def _cycle_reduces_position(cycle: Any) -> bool:
    execution = cycle.execution_result or {}
    return any(bool((order.get("intent") or {}).get("reduce_only")) for order in execution.get("orders", []) or [])


def _cycle_pnl_stats(
    cycles: list,
    cutoff_30d: datetime,
    inception_cutoff: datetime | None = None,
) -> tuple[int, float | None, float, float | None, float]:
    """Aggregate realized PnL from completed cycle execution results."""
    eligible = []
    for cycle in cycles:
        ts_dt = _coerce_timestamp(cycle.created_at)
        if inception_cutoff is not None and (ts_dt is None or ts_dt < inception_cutoff):
            continue
        if not _cycle_reduces_position(cycle):
            continue
        eligible.append(cycle)

    total = len(eligible)
    win_rate: float | None = None
    avg_trade_pnl: float | None = None
    if eligible:
        pnl_values = [_cycle_realized_pnl(cycle) for cycle in eligible]
        wins = [value for value in pnl_values if value > 0.0]
        win_rate = round(len(wins) / len(pnl_values), 4)
        avg_trade_pnl = round(sum(pnl_values) / len(pnl_values), 2)

    realized = 0.0
    realized_cumulative = 0.0
    for cycle in eligible:
        pnl_val = _cycle_realized_pnl(cycle)
        realized_cumulative += pnl_val
        ts_dt = _coerce_timestamp(cycle.created_at)
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
    journal_store=None,
) -> dict[str, object]:
    """Derive (sharpe_90d, win_rate, total_trades, realized_pnl_30d, total_return, total_return_pct).

    - **sharpe_90d**: mean/std of daily equity returns over last 90 days, annualised
      by sqrt(365). ``None`` when fewer than 30 daily samples.
    - **win_rate**: share of reducing execution cycles with positive realized PnL.
    - **total_trades**: count of completed cycles that reduce an existing position.
    - **realized_pnl_30d**: realized PnL from their execution result payloads.
    - **total_return / total_return_pct**: *realized* trading PnL since
      inception — sum of all closed-trade ``pnl`` since the first portfolio
      snapshot. **Does not include unrealized PnL on open positions** to avoid
      the "paper-profit illusion" where a large open winner masks a string of
      realized losers (a SOL short with +$441 unrealized hides 23 realized
      trades averaging -$16). Pct denominator is the baseline
      (config.portfolio.initial_capital when set, otherwise inception equity).

    *Why not ``equity - baseline``?* The earlier formula ``current_equity -
    inception_equity`` silently included USDT deposits and withdrawals — a
    $3,500 user top-up showed as $3,500 of "总收益".

    *Why not ``realized + unrealized``?* (2026-05-11 design) The combined
    measure was abandoned 2026-05-14: a single oversized open position can
    flip the headline from -$376 to +$25, hiding the underlying losing
    streak. Realized-only is conservative — open-position MTM is exposed
    separately via ``unrealized_pnl`` per position in the snapshot.
    """
    now = datetime.now(UTC)
    snaps = await _load_snapshots(database_url)
    sharpe = _sharpe_from_daily(_daily_last_equity(snaps, now - timedelta(days=90)))

    cycles = await _load_cycles(database_url, journal_store)
    inception_ts = _inception_timestamp(snaps)
    total, win_rate, realized_30d, avg_trade_pnl, realized_cumulative = _cycle_pnl_stats(
        cycles, now - timedelta(days=30), inception_cutoff=inception_ts
    )

    # Realized-only trading PnL (excludes deposits / withdrawals and excludes
    # unrealized MTM on open positions — those are exposed per-position in
    # the snapshot.positions[].unrealized_pnl field for transparency).
    total_return = realized_cumulative

    baseline = _inception_equity(snaps)
    total_return_pct = total_return / baseline if baseline is not None and baseline > 0 else 0.0

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


# ── Exchange history cache (funding + fees) ─────────────────────────────────
#
# OKX `fetch_funding_history` + `fetch_my_trades` are ~300-800ms each and the
# dashboard polls /api/portfolio/snapshot every 30s. Cache aggressively so the
# attribution breakdown doesn't double the snapshot latency. 60s TTL is well
# inside the user's perception threshold for fresh PnL.
_EX_HISTORY_TTL_SEC = 60.0
_ex_history_cache: dict[str, tuple[float, dict]] = {}


async def _fetch_funding_window(ex: Any, since_ms: int, window: str) -> float:
    """Sum funding payments since `since_ms`; 0.0 on failure (logged INFO)."""
    try:
        fundings = await asyncio.wait_for(ex.fetch_funding_history(since=since_ms, limit=200), timeout=4.0)
        return sum(float(f.get("amount", 0) or 0) for f in (fundings or []))
    except Exception:
        logger.info("fetch_funding_history failed for %s", window, exc_info=True)
        return 0.0


async def _fetch_fees_window(ex: Any, since_ms: int, window: str) -> float:
    """Sum absolute fees since `since_ms` and return as a negative cost; 0.0 on failure."""
    try:
        trades = await asyncio.wait_for(ex.fetch_my_trades(since=since_ms, limit=200), timeout=4.0)
        total_fee = 0.0
        for t in trades or []:
            fee = t.get("fee") or {}
            total_fee += abs(float(fee.get("cost", 0) or 0))
        return -total_fee  # sign-flip so positive means cost
    except Exception:
        logger.info("fetch_my_trades failed for %s", window, exc_info=True)
        return 0.0


async def _fetch_exchange_history(now: datetime) -> dict | None:
    """Platform history attribution moves to the book API in Task 17."""
    return None


def _norm_ts(t: Any) -> datetime | None:
    """Return a tz-aware datetime or None for snapshot timestamps."""
    if t is None:
        return None
    if isinstance(t, datetime):
        return t if t.tzinfo else t.replace(tzinfo=UTC)
    return None


def _first_eq_at_or_after(snaps: list[dict], cutoff: datetime) -> float | None:
    """Earliest snapshot total_value at or after `cutoff`; None if no snapshot covers the window."""
    for s in snaps:
        t = _norm_ts(s.get("timestamp"))
        if t and t >= cutoff:
            return float(s.get("total_value", 0.0) or 0.0)
    return None


def _sum_realized_pnl_since(cycles: list[Any], cutoff: datetime) -> float:
    """Sum realized execution PnL since ``cutoff``."""
    realized = 0.0
    for cycle in cycles:
        ts = _coerce_timestamp(cycle.created_at)
        if not ts:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if ts < cutoff:
            continue
        if not _cycle_reduces_position(cycle):
            continue
        realized += _cycle_realized_pnl(cycle)
    return realized


async def _compute_pnl_breakdowns(
    database_url: str | None,
    current_equity: float,
    journal_store=None,
) -> list[PnlBreakdown]:
    """Build 24h / 7d / 30d attribution with 4 buckets:
    realized / funding / fees / unrealized_delta.

    realized comes from completed Cycle Journal execution results; funding + fees from OKX history
    (cached 60s). unrealized_delta is derived to make the identity hold.
    """
    snaps = await _load_snapshots(database_url)
    cycles = await _load_cycles(database_url, journal_store)
    now = datetime.now(UTC)
    ex_hist = await _fetch_exchange_history(now)
    exchange_ok = bool(ex_hist and ex_hist.get("_available"))

    out: list[PnlBreakdown] = []
    for window, hours in (("24h", 24), ("7d", 24 * 7), ("30d", 24 * 30)):
        cutoff = now - timedelta(hours=hours)
        eq_start = _first_eq_at_or_after(snaps, cutoff)
        if eq_start is None:
            continue

        delta = current_equity - eq_start
        realized = _sum_realized_pnl_since(cycles, cutoff)
        funding = float(ex_hist.get(f"{window}:funding", 0.0)) if ex_hist else 0.0
        fees = float(ex_hist.get(f"{window}:fees", 0.0)) if ex_hist else 0.0
        unrealized_delta = delta - realized - funding - fees

        out.append(
            PnlBreakdown(
                window=window,
                delta=round(delta, 2),
                realized=round(realized, 2),
                funding=round(funding, 2),
                fees=round(fees, 2),
                unrealized_delta=round(unrealized_delta, 2),
                exchange_data_available=exchange_ok,
            )
        )
    return out


@router.get("/snapshot", response_model=PortfolioSnapshotOut)
async def get_portfolio_snapshot(request: Request) -> PortfolioSnapshotOut:
    """Return current portfolio snapshot. Prefer live exchange over DB."""
    from cryptotrader.portfolio.manager import PortfolioManager

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Trading runtime is not initialized")
    config = runtime.snapshot.document
    database_url = getattr(runtime.repository, "database_url", None)
    journal = runtime.cycle.journal if runtime.cycle is not None else None
    pm = PortfolioManager(database_url)

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
                _read_live_portfolio(request, config),
                timeout=_OKX_FETCH_TIMEOUT_SEC,
            )
            globals()["_OKX_LAST_FAIL_AT"] = 0.0
            globals()["_OKX_LAST_OK_AT"] = now
            globals()["_OKX_LAST_OK_RESULT"] = live
        except Exception:
            logger.info("live portfolio read timed out / failed; using DB", exc_info=True)
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
        database_url,
        current_equity=equity,
        raw_positions=raw_positions,
        journal_store=journal,
    )
    pnl_breakdowns = await _compute_pnl_breakdowns(
        database_url,
        current_equity=equity,
        journal_store=journal,
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
        pnl_breakdowns=pnl_breakdowns,
    )


_RANGE_HOURS = {"24h": 24, "7d": 24 * 7, "30d": 24 * 30}


@router.get("/equity-curve", response_model=EquityCurveOut)
async def get_equity_curve(
    request: Request,
    range: Literal["24h", "7d", "30d", "all"] = Query(...),
) -> EquityCurveOut:
    from cryptotrader.portfolio.manager import PortfolioManager

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Trading runtime is not initialized")
    pm = PortfolioManager(getattr(runtime.repository, "database_url", None))
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
