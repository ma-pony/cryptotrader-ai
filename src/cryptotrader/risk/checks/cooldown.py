"""Cooldown risk check."""

from __future__ import annotations

import logging

from cryptotrader.config import CooldownConfig
from cryptotrader.models import CheckResult, TradeVerdict
from cryptotrader.risk.state import RedisStateManager

logger = logging.getLogger(__name__)


class CooldownCheck:
    name = "cooldown"

    def __init__(self, config: CooldownConfig, redis_state: RedisStateManager) -> None:
        self._same_pair_minutes = config.same_pair_minutes
        self._post_loss_minutes = config.post_loss_minutes
        self._redis = redis_state

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        pair = portfolio.get("pair", "")
        # Per-pair cooldown (works via Redis or in-memory fallback)
        val = await self._redis.get(f"cooldown:{pair}")
        if val is not None:
            return CheckResult(passed=False, reason=f"Cooldown active for {pair}")
        # Global post-loss cooldown
        val = await self._redis.get("cooldown:post_loss")
        if val is not None:
            return CheckResult(passed=False, reason="Post-loss cooldown active")
        return CheckResult(passed=True)
