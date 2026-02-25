"""Loss limit risk checks."""

from __future__ import annotations

import logging

from cryptotrader.config import LossConfig
from cryptotrader.models import CheckResult, TradeVerdict
from cryptotrader.risk.state import RedisStateManager

logger = logging.getLogger(__name__)


class DailyLossLimit:
    name = "daily_loss_limit"

    def __init__(self, config: LossConfig, redis_state: RedisStateManager | None = None) -> None:
        self._max_pct = config.max_daily_loss_pct
        self._redis = redis_state
        self.circuit_breaker = False

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        # Check Redis-backed circuit breaker first (survives restarts)
        if self._redis and self._redis.available:
            if await self._redis.is_circuit_breaker_active():
                self.circuit_breaker = True
                return CheckResult(passed=False, reason="Circuit breaker active (persistent)")
        if self.circuit_breaker:
            return CheckResult(passed=False, reason="Circuit breaker active")
        total = portfolio.get("total_value", 0)
        if total <= 0:
            return CheckResult(passed=False, reason="Invalid portfolio value")
        daily_pnl = portfolio.get("daily_pnl", 0)
        loss_pct = abs(daily_pnl) / total if daily_pnl < 0 else 0
        if loss_pct > self._max_pct:
            self.circuit_breaker = True
            # Persist to Redis so it survives process restarts
            if self._redis and self._redis.available:
                await self._redis.set_circuit_breaker()
            logger.warning("Circuit breaker triggered: daily loss %.2f%% exceeds max %.2f%%",
                           loss_pct * 100, self._max_pct * 100)
            return CheckResult(passed=False, reason=f"Daily loss {loss_pct:.2%} exceeds max {self._max_pct:.2%}")
        return CheckResult(passed=True)


class DrawdownLimit:
    name = "drawdown_limit"

    def __init__(self, config: LossConfig) -> None:
        self._max_pct = config.max_drawdown_pct

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        drawdown = abs(portfolio.get("drawdown", 0))
        if drawdown > self._max_pct:
            return CheckResult(passed=False, reason=f"Drawdown {drawdown:.2%} exceeds max {self._max_pct:.2%}")
        return CheckResult(passed=True)
