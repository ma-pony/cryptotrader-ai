"""依次运行全部目标仓位风险检查并聚合最严格 cap。"""

from __future__ import annotations

import logging
from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from cryptotrader.decision.models import TargetPosition
from cryptotrader.execution.allocation import WeightedAllocationPolicy
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
from cryptotrader.risk.models import (
    BookRiskDecision,
    BookRiskLimits,
    BookRiskRequest,
    ConnectionRiskDecision,
    ConnectionRiskLimits,
    ConnectionRiskRequest,
    RiskDecision,
    RiskRequest,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cryptotrader.config import RiskConfig
    from cryptotrader.risk.state import RedisStateManager

logger = logging.getLogger(__name__)


class BookRiskGate:
    """Cap one whole execution book and rerun its fixed allocation."""

    def __init__(
        self,
        limits: BookRiskLimits,
        allocation_policy: WeightedAllocationPolicy | None = None,
    ) -> None:
        if not isinstance(limits, BookRiskLimits):
            raise ValueError("limits must be BookRiskLimits")
        self._limits = limits
        self._allocation_policy = allocation_policy or WeightedAllocationPolicy()

    def evaluate(self, request: BookRiskRequest) -> BookRiskDecision:
        if not isinstance(request, BookRiskRequest):
            raise ValueError("request must be a BookRiskRequest")
        requested = request.target_exposure
        absolute_cap = Decimal("1")
        cap_source = ""
        candidates = (
            (self._limits.max_net_exposure, "max_net_exposure"),
            (self._limits.max_gross_exposure, "max_gross_exposure"),
            (self._concentration_cap(request), "max_connection_concentration"),
        )
        for candidate, source in candidates:
            if candidate < absolute_cap:
                absolute_cap = candidate
                cap_source = source

        drawdown = max(
            Decimal("0"),
            (request.peak_equity - request.portfolio.total_equity) / request.peak_equity,
        )
        if drawdown > self._limits.max_drawdown:
            absolute_cap = Decimal("0")
            cap_source = "max_drawdown"

        capped = self._signed_cap(requested, absolute_cap)
        connection_targets = self._allocation_policy.allocate_exposure(
            capped,
            request.book,
            request.portfolio,
        )
        weights = tuple(item.weight for item in connection_targets)
        return BookRiskDecision(
            passed=True,
            requested_target_exposure=requested,
            capped_target_exposure=capped,
            connection_weights=weights,
            connection_targets=connection_targets,
            reason="" if capped == requested else f"book target exposure capped by {cap_source}",
            cap_source=cap_source if capped != requested else "",
        )

    def _concentration_cap(self, request: BookRiskRequest) -> Decimal:
        enabled_weights = tuple(
            Decimal(str(allocation.weight)) for allocation in request.book.allocations if allocation.enabled
        )
        if not enabled_weights:
            return Decimal("0")
        largest_weight = max(enabled_weights)
        if largest_weight == 0:
            return Decimal("0")
        return min(Decimal("1"), self._limits.max_connection_concentration / largest_weight)

    @staticmethod
    def _signed_cap(requested: Decimal, absolute_cap: Decimal) -> Decimal:
        if abs(requested) <= absolute_cap:
            return requested
        return absolute_cap if requested > 0 else -absolute_cap


class ConnectionRiskGate:
    """Preflight normalized venue state while preserving safe risk reduction."""

    def __init__(self, limits: ConnectionRiskLimits) -> None:
        if not isinstance(limits, ConnectionRiskLimits):
            raise ValueError("limits must be ConnectionRiskLimits")
        self._limits = limits

    def evaluate(self, request: ConnectionRiskRequest) -> ConnectionRiskDecision:
        if not isinstance(request, ConnectionRiskRequest):
            raise ValueError("request must be a ConnectionRiskRequest")
        connection_id = request.target.connection_id
        increase = request.risk_increase
        if not request.available:
            return ConnectionRiskDecision(connection_id, False, increase, "connection unavailable", "availability")

        reason = self._available_rejection_reason(request)
        return ConnectionRiskDecision(connection_id, reason == "", increase, reason, "risk" if reason else "")

    def _available_rejection_reason(self, request: ConnectionRiskRequest) -> str:
        increase = request.risk_increase

        pair = request.portfolio.position.pair
        capabilities = request.capabilities
        assert capabilities is not None
        if pair.market_type not in capabilities.market_types:
            return "market type unsupported by connection"
        if "market" not in capabilities.supported_order_types:
            return "market order unsupported by connection"

        target_notional = request.target.target_signed_notional
        if pair.market_type == "spot" and target_notional < 0:
            return "spot markets do not support short exposure"

        if not increase:
            current_notional = request.portfolio.position.signed_notional
            reducing_derivative = pair.market_type != "spot" and abs(target_notional) < abs(current_notional)
            if reducing_derivative and not capabilities.reduce_only:
                return "reduce-only capability required for derivative risk reduction"
            return ""

        if pair.market_type != "spot" and not capabilities.native_protection:
            return "native protection required for derivative risk increase"
        equity = request.portfolio.equity
        if equity <= 0:
            return "positive equity required for risk increase"
        if abs(target_notional) > equity * self._limits.max_margin_fraction:
            return "insufficient margin for risk increase"
        return ""


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
    ) -> tuple[RiskDecision | None, list[tuple[float, str, str]]]:
        failed: RiskDecision | None = None
        proposals: list[tuple[float, str, str]] = []
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
                proposals.append((result.size_ratio_cap, check.name, result.reason))
        return failed, proposals

    @staticmethod
    def _apply_cap(request: RiskRequest, proposals: list[tuple[float, str, str]]) -> RiskDecision:
        if proposals:
            cap, source, reason = min(proposals, key=lambda item: item[0])
            if cap < request.target.size_ratio:
                capped_target = TargetPosition("flat", 0.0) if cap == 0.0 else replace(request.target, size_ratio=cap)
                capped_plan = replace(request.plan, target=capped_target)
                return RiskDecision(
                    passed=True,
                    plan=capped_plan,
                    reason=reason or f"target size ratio capped at {cap:g}",
                    cap_source=source,
                )
        return RiskDecision(passed=True, plan=request.plan)
