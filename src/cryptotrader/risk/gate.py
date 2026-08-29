"""依次运行全部目标仓位风险检查并聚合最严格 cap。"""

from __future__ import annotations

from decimal import Decimal

from cryptotrader.execution.allocation import WeightedAllocationPolicy
from cryptotrader.risk.models import (
    BookRiskDecision,
    BookRiskLimits,
    BookRiskRequest,
    ConnectionRiskDecision,
    ConnectionRiskLimits,
    ConnectionRiskRequest,
)


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
