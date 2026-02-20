"""Cooldown risk check."""

from __future__ import annotations

from cryptotrader.config import CooldownConfig
from cryptotrader.models import CheckResult, TradeVerdict
from cryptotrader.risk.state import RedisStateManager


class CooldownCheck:
    name = "cooldown"

    def __init__(self, config: CooldownConfig, redis_state: RedisStateManager) -> None:
        self._minutes = config.same_pair_minutes
        self._redis = redis_state

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        if not self._redis.available:
            return CheckResult(passed=True, reason="Redis unavailable, skipping cooldown check")
        pair = portfolio.get("pair", "")
        key = f"cooldown:{pair}"
        val = await self._redis.get(key)
        if val is not None:
            return CheckResult(passed=False, reason=f"Cooldown active for {pair}")
        return CheckResult(passed=True)
