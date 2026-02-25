"""Rate limit risk check."""

from __future__ import annotations

import logging

from cryptotrader.config import RateLimitConfig
from cryptotrader.models import CheckResult, TradeVerdict
from cryptotrader.risk.state import RedisStateManager

logger = logging.getLogger(__name__)


class RateLimitCheck:
    name = "rate_limit"

    def __init__(self, config: RateLimitConfig, redis_state: RedisStateManager) -> None:
        self._max_hour = config.max_trades_per_hour
        self._max_day = config.max_trades_per_day
        self._redis = redis_state

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        # Works via Redis or in-memory fallback
        hourly, daily = await self._redis.get_trade_counts()
        if hourly >= self._max_hour:
            return CheckResult(passed=False, reason=f"Hourly trade limit ({self._max_hour}) reached")
        if daily >= self._max_day:
            return CheckResult(passed=False, reason=f"Daily trade limit ({self._max_day}) reached")
        return CheckResult(passed=True)
