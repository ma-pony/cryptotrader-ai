"""Risk gate that runs all checks."""

from __future__ import annotations

import logging

from cryptotrader.config import RiskConfig
from cryptotrader.models import GateResult, TradeVerdict
from cryptotrader.risk.checks.cooldown import CooldownCheck
from cryptotrader.risk.checks.correlation import CorrelationCheck
from cryptotrader.risk.checks.cvar import CVaRCheck
from cryptotrader.risk.checks.exchange import ExchangeHealthCheck
from cryptotrader.risk.checks.loss import DailyLossLimit, DrawdownLimit
from cryptotrader.risk.checks.position import MaxPositionSize, MaxTotalExposure
from cryptotrader.risk.checks.rate_limit import RateLimitCheck
from cryptotrader.risk.checks.token_security import TokenSecurityCheck
from cryptotrader.risk.checks.volatility import FundingRateGate, VolatilityGate
from cryptotrader.risk.state import RedisStateManager

logger = logging.getLogger(__name__)


class RiskGate:
    def __init__(self, config: RiskConfig, redis_state: RedisStateManager) -> None:
        self.redis_state = redis_state
        self._redis_was_configured = getattr(redis_state, "_redis", None) is not None
        self._checks = [
            MaxPositionSize(config.position),
            MaxTotalExposure(config.position),
            DailyLossLimit(config.loss, redis_state),
            DrawdownLimit(config.loss),
            CVaRCheck(config.loss),
            CorrelationCheck(),
            CooldownCheck(config.cooldown, redis_state),
            VolatilityGate(config.volatility),
            FundingRateGate(config.volatility),
            RateLimitCheck(config.rate_limit, redis_state),
            ExchangeHealthCheck(config.exchange),
            TokenSecurityCheck(),
        ]

    async def check(self, verdict: TradeVerdict, portfolio: dict) -> GateResult:
        # Conservative rejection: if Redis was configured but is now unavailable,
        # reject trades since cooldown/rate-limit state may be lost
        if self._redis_was_configured and not await self.redis_state.ping():
            logger.warning("Redis unavailable, rejecting trade (conservative)")
            return GateResult(
                passed=False,
                rejected_by="redis_health",
                reason="Redis configured but unavailable — rejecting conservatively",
            )
        for c in self._checks:
            result = await c.evaluate(verdict, portfolio)
            if not result.passed:
                return GateResult(passed=False, rejected_by=c.name, reason=result.reason)
        return GateResult(passed=True)
