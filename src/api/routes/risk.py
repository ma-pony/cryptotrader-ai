"""Risk status + circuit-breaker reset endpoints (FR-807 / FR-404)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


class RiskStatusOut(BaseModel):
    trade_count_hour: int | None
    trade_count_day: int | None
    circuit_breaker: CircuitBreakerStatus
    thresholds: RiskThresholds
    redis_available: bool


class CircuitBreakerResetOut(BaseModel):
    success: bool
    message: str


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

    if rsm.available:
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

    return RiskStatusOut(
        trade_count_hour=trade_count_hour,
        trade_count_day=trade_count_day,
        circuit_breaker=cb,
        thresholds=_build_thresholds(config),
        redis_available=rsm.available,
    )


@router.post("/circuit-breaker/reset", response_model=CircuitBreakerResetOut)
async def reset_circuit_breaker() -> CircuitBreakerResetOut:
    from cryptotrader.config import load_config
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    rsm = RedisStateManager(config.infrastructure.redis_url)

    if not rsm.available:
        raise HTTPException(status_code=503, detail="Redis 不可达")

    if not await rsm.is_circuit_breaker_active():
        raise HTTPException(status_code=409, detail="断路器当前未触发, 无需重置")

    await rsm.reset_circuit_breaker()
    return CircuitBreakerResetOut(success=True, message="断路器已重置")
