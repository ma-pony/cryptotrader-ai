"""Risk status + circuit-breaker reset endpoints (FR-807 / FR-404)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cryptotrader._compat import UTC

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/risk", tags=["risk"])

# Circuit-breaker default TTL (24h) — matches RedisStateManager.set_circuit_breaker default.
_CB_TTL_SECONDS = 86400


# ── Response models (data-model §4) ──


class CircuitBreakerStatus(BaseModel):
    state: Literal["active", "inactive"]
    triggered_at: str | None = None
    expires_at: str | None = None
    reason: str | None = None


class RiskThresholds(BaseModel):
    max_position_pct: float
    max_daily_loss_pct: float
    max_stop_loss_pct: float
    max_trades_per_hour: int
    max_trades_per_day: int
    post_loss_cooldown_seconds: int


class CorrelationGroupOut(BaseModel):
    name: str
    open: int
    max: int
    pairs: list[str]


class CooldownOut(BaseModel):
    pair: str
    until_seconds: int  # seconds remaining; 0 when free
    kind: str  # "same-pair" | "post-loss" | "free"


class RecentBlockOut(BaseModel):
    ts: str
    commit_hash: str
    rule: str
    detail: str


class RiskStatusOut(BaseModel):
    trade_count_hour: int | None
    trade_count_day: int | None
    circuit_breaker: CircuitBreakerStatus
    thresholds: RiskThresholds
    redis_available: bool
    # Alignment with frontend prototype (2026-04-24):
    daily_loss_pct: float | None = None
    drawdown_pct: float | None = None
    total_exposure_pct: float | None = None
    cvar_95: float | None = None
    correlation_groups: list[CorrelationGroupOut] = []
    cooldowns: list[CooldownOut] = []
    recent_blocks: list[RecentBlockOut] = []


class CircuitBreakerResetOut(BaseModel):
    success: bool
    message: str


# Correlation groups — mirrors RiskGate's CorrelationCheck list (cryptotrader/risk/checks/correlation.py).
_CORR_GROUPS: dict[str, list[str]] = {
    "BTC-correlated": ["BTC/USDT", "BTC/USD"],
    "ETH-correlated": ["ETH/USDT", "ETH/USD"],
    "L1-alt": ["SOL/USDT", "AVAX/USDT", "BNB/USDT"],
    "DeFi": ["LINK/USDT", "UNI/USDT", "AAVE/USDT"],
    "meme": ["DOGE/USDT", "PEPE/USDT", "SHIB/USDT"],
}

_CORR_GROUP_MAX = 2


async def _build_correlation_groups(
    database_url: str | None,
    *,
    portfolio: dict | None = None,
) -> list[CorrelationGroupOut]:
    """Count currently-open positions per correlation group."""
    raw_positions: dict = {}
    if portfolio is not None:
        raw_positions = portfolio.get("positions", {}) or {}
    else:
        try:
            from cryptotrader.portfolio.manager import PortfolioManager

            pm = PortfolioManager(database_url)
            portfolio = await pm.get_portfolio()
            raw_positions = portfolio.get("positions", {}) or {}
        except Exception:
            logger.info("correlation groups: portfolio read failed", exc_info=True)
            raw_positions = {}

    open_pairs = {p for p, pos in raw_positions.items() if float(pos.get("amount", 0.0) or 0.0) != 0.0}
    out: list[CorrelationGroupOut] = []
    for name, pairs in _CORR_GROUPS.items():
        open_count = sum(1 for p in pairs if p in open_pairs)
        out.append(CorrelationGroupOut(name=name, open=open_count, max=_CORR_GROUP_MAX, pairs=pairs))
    return out


async def _known_pairs(
    database_url: str | None,
    *,
    portfolio: dict | None = None,
) -> list[str]:
    """Union of currently-open pairs and pairs active in the last 30 journal commits.

    This is the candidate set we probe for Redis cooldown TTLs — avoiding a scan.
    """
    pairs: set[str] = set()
    if portfolio is not None:
        for p, pos in (portfolio.get("positions", {}) or {}).items():
            if float(pos.get("amount", 0.0) or 0.0) != 0.0:
                pairs.add(p)
    else:
        try:
            from cryptotrader.portfolio.manager import PortfolioManager

            pm = PortfolioManager(database_url)
            pf = await pm.get_portfolio()
            for p, pos in (pf.get("positions", {}) or {}).items():
                if float(pos.get("amount", 0.0) or 0.0) != 0.0:
                    pairs.add(p)
        except Exception:
            logger.info("known pairs: portfolio read failed", exc_info=True)

    try:
        from cryptotrader.journal.store import JournalStore

        store = JournalStore(database_url)
        commits = await store.log(limit=30)
        for c in commits:
            if c.pair:
                pairs.add(c.pair)
    except Exception:
        logger.info("known pairs: journal read failed", exc_info=True)

    return sorted(pairs)


async def _build_cooldowns(
    rsm: Any,
    database_url: str | None,
    *,
    portfolio: dict | None = None,
) -> list[CooldownOut]:
    """Emit cooldown rows by querying Redis TTLs directly.

    Two cooldown kinds are reported:
      - ``same-pair``: ``cooldown:{pair}`` TTL (set by ``CooldownCheck``).
      - ``post-loss``: global ``cooldown:post_loss`` TTL (set after losses).

    Pairs with no active cooldown appear as ``free`` rows so the UI can render
    "这些 pair 已就绪". The pair candidate set comes from open positions + recent
    journal activity (bounded); pairs the bot has never touched are skipped.
    """
    pairs = await _known_pairs(database_url, portfolio=portfolio)
    out: list[CooldownOut] = []
    try:
        for pair in pairs:
            ttl = await rsm.ttl(f"cooldown:{pair}")
            if ttl > 0:
                out.append(CooldownOut(pair=pair, until_seconds=ttl, kind="same-pair"))
            else:
                out.append(CooldownOut(pair=pair, until_seconds=0, kind="free"))
        # Global post-loss cooldown — surfaced as a synthetic row so the UI can
        # render it alongside pair-specific entries.
        post_loss_ttl = await rsm.ttl("cooldown:post_loss")
        if post_loss_ttl > 0:
            out.insert(0, CooldownOut(pair="*", until_seconds=post_loss_ttl, kind="post-loss"))
    except Exception:
        logger.info("cooldown TTL read failed", exc_info=True)
    return out


async def _build_recent_blocks(database_url: str | None) -> list[RecentBlockOut]:
    """Last 10 risk-gate rejections."""
    from cryptotrader.journal.store import JournalStore

    try:
        store = JournalStore(database_url)
        commits = await store.log(limit=200)
    except Exception:
        logger.info("recent blocks: journal read failed", exc_info=True)
        return []

    blocks: list[RecentBlockOut] = []
    for c in commits:
        gate = c.risk_gate
        if gate is None or gate.passed:
            continue
        ts = c.timestamp
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        blocks.append(
            RecentBlockOut(
                ts=ts_str,
                commit_hash=c.hash,
                rule=getattr(gate, "rejected_by", "") or "unknown",
                detail=getattr(gate, "reason", "") or "",
            )
        )
        if len(blocks) >= 10:
            break
    return blocks


async def _compute_drawdown_pct(
    database_url: str | None,
    *,
    snaps: list | None = None,
) -> float | None:
    """Return current drawdown as positive percentage (e.g. ``33.34``).

    Always delegates to ``PortfolioManager.get_drawdown`` so the
    operator-initiated baseline reset (``arena portfolio reset-baseline``,
    spec drawdown decoupling 2026-05-07) is honored. The earlier inline
    fast-path that recomputed peak/trough from a pre-loaded ``snaps`` list
    bypassed the baseline cutoff and is no longer used; ``snaps`` is left
    in the signature only to avoid breaking the existing caller in
    ``risk_status`` that passes it through ``asyncio.gather``.
    """
    try:
        from cryptotrader.portfolio.manager import PortfolioManager

        pm = PortfolioManager(database_url)
        return round(abs(float(await pm.get_drawdown())) * 100.0, 2)
    except Exception:
        logger.info("drawdown_pct: read failed", exc_info=True)
        return None


async def _compute_daily_loss_pct(
    database_url: str | None,
    *,
    portfolio: dict | None = None,
    pnl_24h: float | None = None,
) -> float | None:
    """Current daily loss as a percentage of equity (positive when losing)."""
    try:
        if portfolio is not None and pnl_24h is not None:
            equity = float(portfolio.get("total_value", 0.0) or 0.0)
        else:
            from cryptotrader.portfolio.manager import PortfolioManager

            pm = PortfolioManager(database_url)
            portfolio = portfolio or await pm.get_portfolio()
            equity = float(portfolio.get("total_value", 0.0) or 0.0)
            if pnl_24h is None:
                raw = await pm.get_daily_pnl()
                pnl_24h = float(raw) if raw is not None else 0.0
        if equity <= 0:
            return None
        # Positive value = loss magnitude; negative = profit. Frontend meter expects percent.
        return round(-float(pnl_24h) / equity * 100.0, 2)
    except Exception:
        logger.info("daily_loss_pct: read failed", exc_info=True)
        return None


async def _compute_total_exposure_pct(
    database_url: str | None,
    *,
    portfolio: dict | None = None,
) -> float | None:
    """Sum of absolute notional / equity * 100."""
    try:
        if portfolio is None:
            from cryptotrader.portfolio.manager import PortfolioManager

            pm = PortfolioManager(database_url)
            portfolio = await pm.get_portfolio()
        equity = float(portfolio.get("total_value", 0.0) or 0.0)
        raw_positions = portfolio.get("positions", {}) or {}
        if equity <= 0:
            return None
        notional = 0.0
        for pos in raw_positions.values():
            amount = abs(float(pos.get("amount", 0.0) or 0.0))
            price = float(pos.get("avg_price", 0.0) or 0.0)
            notional += amount * price
        return round(notional / equity * 100.0, 2)
    except Exception:
        logger.info("total_exposure_pct: read failed", exc_info=True)
        return None


async def _compute_cvar_95(
    database_url: str | None,
    *,
    snaps: list | None = None,
) -> float | None:
    """Historical CVaR at 95% from equity-curve daily returns.

    CVaR_95 = mean of the worst 5% of daily returns (in percent, positive = loss).
    Returns ``None`` when fewer than 30 samples available. If ``snaps`` is
    provided, avoids re-reading the snapshot table.
    """
    if snaps is None:
        try:
            from cryptotrader.portfolio.manager import PortfolioManager

            pm = PortfolioManager(database_url)
            snaps = await pm.load_snapshots("default")
        except Exception:
            logger.info("cvar_95: read failed", exc_info=True)
            return None

    daily_last: dict[str, float] = {}
    for s in snaps:
        ts = s.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
        if not hasattr(ts, "date"):
            continue
        day = ts.date().isoformat()  # type: ignore[union-attr]
        daily_last[day] = float(s.get("total_value", 0.0) or 0.0)

    if len(daily_last) < 30:
        return None
    series = [daily_last[k] for k in sorted(daily_last.keys())]
    returns = [(series[i] - series[i - 1]) / series[i - 1] for i in range(1, len(series)) if series[i - 1] > 0]
    if not returns:
        return None
    returns_sorted = sorted(returns)
    tail_count = max(1, int(len(returns_sorted) * 0.05))
    tail = returns_sorted[:tail_count]
    cvar = -sum(tail) / len(tail)  # negative returns → positive CVaR
    return round(cvar * 100.0, 2)


def _build_thresholds(config: object) -> RiskThresholds:
    """Translate ``RiskConfig`` (internal) → data-model §4 ``RiskThresholds``.

    Field name mapping handles two divergences from the spec:
    - ``cooldown.post_loss_minutes`` (config) -> ``post_loss_cooldown_seconds`` (x 60)
    - ``rate_limit.max_trades_per_*`` (config) → ``max_trades_per_*`` (alias only)
    """
    risk = config.risk  # type: ignore[attr-defined]
    return RiskThresholds(
        max_position_pct=float(risk.position.max_single_pct),
        max_daily_loss_pct=float(risk.loss.max_daily_loss_pct),
        max_stop_loss_pct=float(risk.max_stop_loss_pct),
        max_trades_per_hour=int(risk.rate_limit.max_trades_per_hour),
        max_trades_per_day=int(risk.rate_limit.max_trades_per_day),
        post_loss_cooldown_seconds=int(risk.cooldown.post_loss_minutes) * 60,
    )


# ── Routes ──


@router.get("/status", response_model=RiskStatusOut)
async def get_risk_status() -> RiskStatusOut:
    from cryptotrader.config import load_config
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    rsm = RedisStateManager(config.infrastructure.redis_url)
    # Real liveness check — `available` only checks the client exists, not that
    # the server is reachable. Use ping() so /api/risk/status reports the same
    # truth as /health.
    redis_alive = await rsm.ping()

    if redis_alive:
        try:
            hourly, daily = await rsm.get_trade_counts()
            cb_active = await rsm.is_circuit_breaker_active()
        except Exception:
            logger.warning("Redis read failed for risk status", exc_info=True)
            hourly = daily = None  # type: ignore[assignment]
            cb_active = False
        trade_count_hour: int | None = hourly
        trade_count_day: int | None = daily
    else:
        trade_count_hour = None
        trade_count_day = None
        cb_active = False

    if cb_active:
        now = datetime.now(UTC)
        cb = CircuitBreakerStatus(
            state="active",
            triggered_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=_CB_TTL_SECONDS)).isoformat(),
            reason="active",
        )
    else:
        cb = CircuitBreakerStatus(state="inactive")

    db_url = config.infrastructure.database_url

    # Fetch portfolio + snapshots + pnl_24h once, then reuse across all 7 helpers.
    # Previously: 4x get_portfolio() + 2x _load_snapshots() sequential = ~400-800ms.
    # Now: 2 fetches in parallel + 1 gather across 7 helpers.
    from cryptotrader.portfolio.manager import PortfolioManager

    pm = PortfolioManager(db_url)
    try:
        portfolio, snaps, pnl_24h_raw = await asyncio.gather(
            pm.get_portfolio(),
            pm.load_snapshots("default"),
            pm.get_daily_pnl(),
            return_exceptions=False,
        )
    except Exception:
        logger.warning("risk status: portfolio prefetch failed", exc_info=True)
        portfolio, snaps, pnl_24h_raw = {}, [], None
    # get_daily_pnl returns None when no snapshot exists in today's UTC window;
    # surface as 0.0 for downstream serialization (frontend cannot render null).
    pnl_24h = float(pnl_24h_raw) if pnl_24h_raw is not None else 0.0

    results = await asyncio.gather(
        _compute_daily_loss_pct(db_url, portfolio=portfolio, pnl_24h=pnl_24h),
        _compute_drawdown_pct(db_url, snaps=snaps),
        _compute_total_exposure_pct(db_url, portfolio=portfolio),
        _compute_cvar_95(db_url, snaps=snaps),
        _build_correlation_groups(db_url, portfolio=portfolio),
        _build_cooldowns(rsm, db_url, portfolio=portfolio),
        _build_recent_blocks(db_url),
    )
    daily_loss_pct = cast("float | None", results[0])
    drawdown_pct = cast("float | None", results[1])
    total_exposure_pct = cast("float | None", results[2])
    cvar_95 = cast("float | None", results[3])
    correlation_groups = cast("list[CorrelationGroupOut]", results[4])
    cooldowns = cast("list[CooldownOut]", results[5])
    recent_blocks = cast("list[RecentBlockOut]", results[6])

    return RiskStatusOut(
        trade_count_hour=trade_count_hour,
        trade_count_day=trade_count_day,
        circuit_breaker=cb,
        thresholds=_build_thresholds(config),
        redis_available=redis_alive,
        daily_loss_pct=daily_loss_pct,
        drawdown_pct=drawdown_pct,
        total_exposure_pct=total_exposure_pct,
        cvar_95=cvar_95,
        correlation_groups=correlation_groups,
        cooldowns=cooldowns,
        recent_blocks=recent_blocks,
    )


@router.post("/circuit-breaker/reset", response_model=CircuitBreakerResetOut)
async def reset_circuit_breaker() -> CircuitBreakerResetOut:
    from cryptotrader.config import load_config
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    rsm = RedisStateManager(config.infrastructure.redis_url)

    if not await rsm.ping():
        raise HTTPException(status_code=503, detail="Redis 不可达")

    if not await rsm.is_circuit_breaker_active():
        raise HTTPException(status_code=409, detail="断路器当前未触发, 无需重置")

    await rsm.reset_circuit_breaker()
    return CircuitBreakerResetOut(success=True, message="断路器已重置")
