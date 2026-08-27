"""依次运行全部目标仓位风险检查并聚合最严格 cap。"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from cryptotrader.risk.checks.available_margin import AvailableMargin
from cryptotrader.risk.checks.concentration import MacroConcentrationCheck
from cryptotrader.risk.checks.cooldown import CooldownCheck
from cryptotrader.risk.checks.correlation import CorrelationCheck
from cryptotrader.risk.checks.cvar import CVaRCheck
from cryptotrader.risk.checks.exchange import ExchangeHealthCheck
from cryptotrader.risk.checks.loss import DailyLossLimit, DrawdownLimit
from cryptotrader.risk.checks.position import MaxPositionSize, MaxTotalExposure
from cryptotrader.risk.checks.rate_limit import RateLimitCheck
from cryptotrader.risk.checks.token_security import TokenSecurityCheck
from cryptotrader.risk.checks.volatility import FundingRateGate, VolatilityGate
from cryptotrader.risk.models import RiskDecision, RiskRequest

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cryptotrader.config import RiskConfig
    from cryptotrader.risk.state import RedisStateManager

logger = logging.getLogger(__name__)


class RiskGate:
    def __init__(
        self,
        config: RiskConfig,
        redis_state: RedisStateManager,
        *,
        leverage: int = 1,
        checks: Iterable[Any] | None = None,
    ) -> None:
        self.redis_state = redis_state
        self._redis_was_configured = getattr(redis_state, "_redis", None) is not None
        default_checks = [
            MaxPositionSize(config.position),
            MaxTotalExposure(config.position, leverage=leverage),
            AvailableMargin(config.position, leverage=leverage),
            DailyLossLimit(config.loss, redis_state, post_loss_minutes=config.cooldown.post_loss_minutes),
            DrawdownLimit(config.loss, redis_state),
            CVaRCheck(config.loss),
            CorrelationCheck(config.position),
            MacroConcentrationCheck(config.position),
            CooldownCheck(config.cooldown, redis_state),
            VolatilityGate(config.volatility),
            FundingRateGate(config.volatility),
            RateLimitCheck(config.rate_limit, redis_state),
            ExchangeHealthCheck(config.exchange),
            TokenSecurityCheck(),
        ]
        self._checks = list(checks) if checks is not None else default_checks

    async def check(self, request: RiskRequest, portfolio: dict) -> RiskDecision:
        if request.reduces_exposure:
            return RiskDecision(passed=True, plan=request.plan)

        redis_rejection = await self._redis_rejection(request)
        if redis_rejection is not None:
            return redis_rejection

        failed, proposals = await self._evaluate_checks(request, portfolio)
        if failed is not None:
            return failed
        return self._apply_cap(request, proposals)

    async def _redis_rejection(self, request: RiskRequest) -> RiskDecision | None:
        if not self._redis_was_configured or await self.redis_state.ping():
            return None
        error = getattr(self.redis_state, "_last_ping_error", None) or {}
        error_suffix = f" [{error['type']}: {error['msg']}]" if error else ""
        logger.warning("Redis configured but unreachable; rejecting trade conservatively%s", error_suffix)
        return RiskDecision(
            passed=False,
            plan=request.plan,
            rejected_by="redis_unavailable",
            reason=f"Redis configured but unreachable; cannot verify risk state{error_suffix}",
        )

    async def _evaluate_checks(
        self,
        request: RiskRequest,
        portfolio: dict,
    ) -> tuple[RiskDecision | None, list[float]]:
        failed: RiskDecision | None = None
        proposals: list[float] = []
        for check in self._checks:
            try:
                result = await check.evaluate(request, portfolio)
            except Exception:
                logger.warning(
                    "Risk check %s raised an unexpected exception; treating as check_error",
                    check.name,
                    exc_info=True,
                )
                if failed is None:
                    failed = RiskDecision(
                        passed=False,
                        plan=request.plan,
                        rejected_by=check.name,
                        reason=f"check_error: {check.name} raised an unexpected exception",
                    )
                continue

            if not result.passed and failed is None:
                failed = RiskDecision(
                    passed=False,
                    plan=request.plan,
                    rejected_by=check.name,
                    reason=result.reason,
                )
            if result.passed and result.size_ratio_cap is not None:
                proposals.append(result.size_ratio_cap)
        return failed, proposals

    @staticmethod
    def _apply_cap(request: RiskRequest, proposals: list[float]) -> RiskDecision:
        if proposals:
            cap = min(proposals)
            if cap < request.target.size_ratio:
                capped_target = replace(request.target, size_ratio=cap)
                capped_plan = replace(request.plan, target=capped_target)
                return RiskDecision(passed=True, plan=capped_plan)
        return RiskDecision(passed=True, plan=request.plan)
