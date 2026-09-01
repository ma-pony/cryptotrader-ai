"""依次运行全部目标仓位风险检查并聚合最严格 cap。"""
# ruff: noqa: RUF001 - Chinese public risk reasons.

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from cryptotrader.execution.allocation import WeightedAllocationPolicy
from cryptotrader.risk.models import (
    BookRiskDecision,
    BookRiskLimits,
    BookRiskRequest,
    ConnectionRiskDecision,
    ConnectionRiskLimits,
    ConnectionRiskRequest,
    risk_increase,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


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
        state = request.state
        equity = state.equity
        portfolio = request.portfolio
        if equity is None or equity <= 0 or portfolio.total_equity is None:
            if requested != 0:
                return BookRiskDecision(
                    False,
                    requested,
                    requested,
                    (),
                    (),
                    "incomplete_account",
                    "账户估值或完整性不足，仅允许可证明的平仓",
                    state=state,
                )
            return self._decision(request, Decimal("0"), "")
        targets = self._allocation_policy.allocate_exposure(requested, request.book, portfolio)
        positions = {item.connection_id: item.position.signed_notional for item in portfolio.connections}
        reducing = all(not risk_increase(positions[t.connection_id], t.target_signed_notional) for t in targets)
        if reducing:
            return self._decision(request, requested, "")
        if state.completeness:
            return BookRiskDecision(
                False, requested, requested, (), (), "incomplete_account", "账户风险事实不完整，不能增仓", state=state
            )
        absolute_cap = Decimal("1")
        cap_source = ""
        other_gross = sum(
            (
                abs(p.signed_notional.amount)
                for s in state.snapshots
                for p in s.positions
                if p.instrument.pair != request.pair or p.instrument.market_type != request.pair.market_type
            ),
            Decimal("0"),
        )
        other_net = sum(
            (
                p.signed_notional.amount
                for s in state.snapshots
                for p in s.positions
                if p.instrument.pair != request.pair or p.instrument.market_type != request.pair.market_type
            ),
            Decimal("0"),
        )
        pending = state.pending_increase_notional
        if pending is None:
            return BookRiskDecision(
                False, requested, requested, (), (), "incomplete_account", "待成交占用未知", state=state
            )
        gross_room = max(Decimal("0"), equity * self._limits.max_gross_exposure - other_gross - pending)
        net_room = max(
            Decimal("0"),
            equity * self._limits.max_net_exposure - (other_net if requested >= 0 else -other_net) - pending,
        )
        candidates = (
            (gross_room / equity, "max_gross_exposure"),
            (net_room / equity, "max_net_exposure"),
            (self._concentration_cap(request), "max_connection_concentration"),
        )
        for candidate, source in candidates:
            if candidate < absolute_cap:
                absolute_cap = candidate
                cap_source = source

        drawdown = max(
            Decimal("0"),
            (state.peak_equity - equity) / state.peak_equity if state.peak_equity else Decimal("0"),
        )
        if drawdown > self._limits.max_drawdown:
            absolute_cap = Decimal("0")
            cap_source = "max_drawdown"

        capped = self._signed_cap(requested, absolute_cap)
        return self._decision(request, capped, cap_source)

    def _decision(self, request, capped, cap_source):
        requested = request.target_exposure
        connection_targets = self._allocation_policy.allocate_exposure(
            capped,
            request.book,
            request.portfolio,
        )
        weights = tuple(item.weight for item in connection_targets)
        rejection = self.validate_targets(
            request, {target.connection_id: target.target_signed_notional for target in connection_targets}
        )
        return BookRiskDecision(
            passed=not rejection,
            requested_target_exposure=requested,
            capped_target_exposure=capped,
            connection_weights=weights,
            connection_targets=connection_targets,
            rejected_by=rejection,
            reason=rejection or ("" if capped == requested else f"book target exposure capped by {cap_source}"),
            cap_source=cap_source if capped != requested else "",
            state=request.state,
        )

    def validate_targets(self, request: BookRiskRequest, targets: Mapping[str, Decimal]) -> str:
        """Validate actual per-account targets; never reallocate frozen approval amounts."""
        current = {item.connection_id: item.position.signed_notional for item in request.portfolio.connections}
        if set(targets) != set(current) or set(current) != {s.connection_id for s in request.state.snapshots}:
            raise ValueError("actual targets must include every book account")
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in targets.values()):
            raise ValueError("actual targets must be finite Decimal amounts")
        state = request.state
        equity = state.equity
        if equity is None or equity <= 0 or request.portfolio.total_equity is None:
            return "incomplete_account" if any(targets.values()) else ""
        if not any(risk_increase(current[key], value) for key, value in targets.items()):
            return ""
        if state.completeness or state.pending_increase_notional is None:
            return "incomplete_account"
        other = {
            account.connection_id: tuple(
                position.signed_notional.amount
                for position in account.positions
                if position.instrument.pair != request.pair
                or position.instrument.market_type != request.pair.market_type
            )
            for account in state.snapshots
        }
        gross = sum((abs(value) for values in other.values() for value in values), Decimal("0"))
        net = sum((value for values in other.values() for value in values), Decimal("0"))
        pending = state.pending_increase_notional
        if gross + sum(map(abs, targets.values())) + pending > equity * self._limits.max_gross_exposure:
            return "max_gross_exposure"
        if abs(net + sum(targets.values())) + pending > equity * self._limits.max_net_exposure:
            return "max_net_exposure"
        if self._exceeds_concentration(request, targets, other):
            return "max_connection_concentration"
        if state.peak_equity and (state.peak_equity - equity) / state.peak_equity > self._limits.max_drawdown:
            return "max_drawdown"
        return ""

    def _exceeds_concentration(self, request, targets, other):
        for account in request.state.snapshots:
            own_pending = sum(
                (
                    order.remaining_notional.amount
                    for order in account.orders
                    if not order.reduce_only and order.status in {"open", "partially_filled"}
                ),
                Decimal("0"),
            )
            occupied = abs(targets[account.connection_id]) + sum(map(abs, other[account.connection_id])) + own_pending
            if occupied > request.state.equity * self._limits.max_connection_concentration:
                return True
        return False

    def _concentration_cap(self, request: BookRiskRequest) -> Decimal:
        caps = []
        state = request.state
        for allocation in request.book.allocations:
            if not allocation.enabled or allocation.weight <= 0:
                continue
            account = next(s for s in state.snapshots if s.connection_id == allocation.connection_id)
            other = sum(
                (
                    abs(p.signed_notional.amount)
                    for p in account.positions
                    if p.instrument.pair != request.pair or p.instrument.market_type != request.pair.market_type
                ),
                Decimal("0"),
            )
            pending = sum(
                (
                    o.remaining_notional.amount
                    for o in account.orders
                    if not o.reduce_only and o.status in {"open", "partially_filled"}
                ),
                Decimal("0"),
            )
            room = max(Decimal("0"), state.equity * self._limits.max_connection_concentration - other - pending)
            caps.append(room / state.equity / Decimal(str(allocation.weight)))
        return min(caps, default=Decimal("0"))

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

        return self.evaluate_target(request.portfolio, request.target.target_signed_notional, request.capabilities)

    def evaluate_target(self, portfolio, target_notional, capabilities) -> ConnectionRiskDecision:
        """Apply the same connection rules to allocated or frozen actual amounts."""
        increase = risk_increase(portfolio.position.signed_notional, target_notional)
        reason = self._available_rejection_reason(portfolio, target_notional, capabilities)
        return ConnectionRiskDecision(portfolio.connection_id, reason == "", increase, reason, "risk" if reason else "")

    def _available_rejection_reason(self, portfolio, target_notional, capabilities) -> str:
        increase = risk_increase(portfolio.position.signed_notional, target_notional)
        pair = portfolio.position.pair
        assert capabilities is not None
        if pair.market_type not in capabilities.market_types:
            return "market type unsupported by connection"
        if "market" not in capabilities.supported_order_types:
            return "market order unsupported by connection"

        if pair.market_type == "spot" and target_notional < 0:
            return "spot markets do not support short exposure"

        if not increase:
            current_notional = portfolio.position.signed_notional
            reducing_derivative = pair.market_type != "spot" and abs(target_notional) < abs(current_notional)
            if reducing_derivative and not capabilities.reduce_only:
                return "reduce-only capability required for derivative risk reduction"
            return ""

        if pair.market_type != "spot" and not capabilities.native_protection:
            return "native protection required for derivative risk increase"
        return self._margin_rejection_reason(portfolio, target_notional)

    def _margin_rejection_reason(self, portfolio, target_notional) -> str:
        pair = portfolio.position.pair
        equity = portfolio.equity
        if equity is None or equity <= 0:
            return "positive equity required for risk increase"
        account = portfolio.account_snapshot
        if account is None or account.completeness:
            return "complete account margin facts required for risk increase"
        currency = pair.settle or pair.quote
        if any(m.amount is None or m.currency != currency for m in (account.used_margin, account.available_margin)):
            return "same-currency margin facts required for risk increase"
        # No unproven leverage assumption: require incremental notional as collateral.
        additional = max(Decimal("0"), abs(target_notional) - abs(portfolio.position.signed_notional))
        if (
            additional > account.available_margin.amount
            or account.used_margin.amount + additional > equity * self._limits.max_margin_fraction
        ):
            return "insufficient margin for risk increase"
        return ""
