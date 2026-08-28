"""目标仓位风控使用的不可变模型。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from cryptotrader.execution.models import ConnectionTarget, ExecutionBook
from cryptotrader.portfolio.models import BookPortfolioSnapshot, ConnectionPortfolioSnapshot
from cryptotrader.venues.models import OpenVenueState, VenueCapabilities, VenueQuote

if TYPE_CHECKING:
    from cryptotrader.decision.models import TargetPosition, TradePlan
    from cryptotrader.signals.models import SignalContext


@dataclass(frozen=True)
class RiskRequest:
    context: SignalContext
    plan: TradePlan

    @property
    def target(self) -> TargetPosition:
        return self.plan.target

    @property
    def reduces_exposure(self) -> bool:
        current = self.context.current_position.signed_ratio
        target = self.target.signed_ratio
        return target == 0.0 or (current * target > 0.0 and abs(target) <= abs(current))


@dataclass(frozen=True)
class RiskCheckResult:
    passed: bool
    reason: str = ""
    size_ratio_cap: float | None = None


@dataclass(frozen=True)
class RiskDecision:
    passed: bool
    plan: TradePlan
    rejected_by: str = ""
    reason: str = ""
    cap_source: str = ""


def _require_decimal(
    value: object,
    field_name: str,
    *,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> None:
    if not isinstance(value, Decimal):
        raise ValueError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be a finite Decimal")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}")


@dataclass(frozen=True)
class BookRiskLimits:
    max_net_exposure: Decimal
    max_gross_exposure: Decimal
    max_drawdown: Decimal
    max_connection_concentration: Decimal

    def __post_init__(self) -> None:
        for field_name in (
            "max_net_exposure",
            "max_gross_exposure",
            "max_drawdown",
            "max_connection_concentration",
        ):
            _require_decimal(getattr(self, field_name), field_name, minimum=Decimal("0"), maximum=Decimal("1"))


@dataclass(frozen=True)
class BookRiskRequest:
    book: ExecutionBook
    portfolio: BookPortfolioSnapshot
    target_exposure: Decimal
    peak_equity: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.book, ExecutionBook):
            raise ValueError("book must be an ExecutionBook")
        if not isinstance(self.portfolio, BookPortfolioSnapshot):
            raise ValueError("portfolio must be a BookPortfolioSnapshot")
        if self.portfolio.book_id != self.book.id or self.portfolio.capital_scope != self.book.capital_scope:
            raise ValueError("portfolio must belong to the requested execution book")
        _require_decimal(
            self.target_exposure,
            "target_exposure",
            minimum=Decimal("-1"),
            maximum=Decimal("1"),
        )
        _require_decimal(self.peak_equity, "peak_equity", minimum=Decimal("0"))
        if self.peak_equity == 0:
            raise ValueError("peak_equity must be positive")


@dataclass(frozen=True)
class BookRiskDecision:
    passed: bool
    requested_target_exposure: Decimal
    capped_target_exposure: Decimal
    connection_weights: tuple[Decimal, ...]
    connection_targets: tuple[ConnectionTarget, ...]
    rejected_by: str = ""
    reason: str = ""
    cap_source: str = ""

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise ValueError("passed must be a boolean")
        _require_decimal(
            self.requested_target_exposure,
            "requested_target_exposure",
            minimum=Decimal("-1"),
            maximum=Decimal("1"),
        )
        _require_decimal(
            self.capped_target_exposure,
            "capped_target_exposure",
            minimum=Decimal("-1"),
            maximum=Decimal("1"),
        )
        self._validate_targets()
        self._validate_diagnostics()

    def _validate_targets(self) -> None:
        if type(self.connection_weights) is not tuple:
            raise ValueError("connection_weights must be a tuple")
        for weight in self.connection_weights:
            _require_decimal(weight, "connection weight", minimum=Decimal("0"), maximum=Decimal("1"))
        if type(self.connection_targets) is not tuple or not all(
            isinstance(target, ConnectionTarget) for target in self.connection_targets
        ):
            raise ValueError("connection_targets must be a tuple of ConnectionTarget")
        if tuple(target.weight for target in self.connection_targets) != self.connection_weights:
            raise ValueError("connection target weights must match the book decision")
        if any(target.target_exposure != self.capped_target_exposure for target in self.connection_targets):
            raise ValueError("connection targets must use the capped book exposure")
        connection_ids = tuple(target.connection_id for target in self.connection_targets)
        if len(connection_ids) != len(set(connection_ids)):
            raise ValueError("connection targets must have unique connection IDs")
        book_ids = {target.book_id for target in self.connection_targets}
        if len(book_ids) > 1:
            raise ValueError("connection targets must belong to one execution book")

    def _validate_diagnostics(self) -> None:
        for field_name in ("rejected_by", "reason", "cap_source"):
            if type(getattr(self, field_name)) is not str:
                raise ValueError(f"{field_name} must be a string")


@dataclass(frozen=True)
class ConnectionRiskLimits:
    max_margin_fraction: Decimal

    def __post_init__(self) -> None:
        _require_decimal(
            self.max_margin_fraction,
            "max_margin_fraction",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )


@dataclass(frozen=True)
class ConnectionRiskRequest:
    target: ConnectionTarget
    portfolio: ConnectionPortfolioSnapshot
    available: bool
    quote: VenueQuote | None
    open_state: OpenVenueState | None
    capabilities: VenueCapabilities | None

    def __post_init__(self) -> None:
        if not isinstance(self.target, ConnectionTarget):
            raise ValueError("target must be a ConnectionTarget")
        if not isinstance(self.portfolio, ConnectionPortfolioSnapshot):
            raise ValueError("portfolio must be a ConnectionPortfolioSnapshot")
        if self.portfolio.connection_id != self.target.connection_id:
            raise ValueError("portfolio connection_id must match the connection target")
        if type(self.available) is not bool:
            raise ValueError("available must be a boolean")
        if not self.available:
            if any(item is not None for item in (self.quote, self.open_state, self.capabilities)):
                raise ValueError("unavailable requests must not carry partial venue state")
            return
        self._validate_available_state()

    def _validate_available_state(self) -> None:
        if not isinstance(self.quote, VenueQuote):
            raise ValueError("available request requires a VenueQuote")
        if not isinstance(self.open_state, OpenVenueState):
            raise ValueError("available request requires an OpenVenueState")
        if not isinstance(self.capabilities, VenueCapabilities):
            raise ValueError("available request requires VenueCapabilities")
        pair = self.portfolio.position.pair
        if self.quote.pair != pair or self.open_state.position.pair != pair:
            raise ValueError("venue state pair must match the portfolio position")
        if any(order.pair != pair for order in self.open_state.open_orders):
            raise ValueError("open order pairs must match the portfolio position")
        if any(protection.pair != pair for protection in self.open_state.protections):
            raise ValueError("protection pairs must match the portfolio position")
        if self.open_state.position != self.portfolio.position:
            raise ValueError("open state position must match the portfolio snapshot")

    @property
    def risk_increase(self) -> bool:
        current = self.portfolio.position.signed_notional
        target = self.target.target_signed_notional
        if target == 0:
            return False
        if current == 0 or current * target < 0:
            return True
        return abs(target) > abs(current)


@dataclass(frozen=True)
class ConnectionRiskDecision:
    connection_id: str
    passed: bool
    risk_increase: bool
    reason: str = ""
    operation: str = ""

    def __post_init__(self) -> None:
        if type(self.connection_id) is not str or not self.connection_id.strip():
            raise ValueError("connection_id must be a non-empty string")
        if type(self.passed) is not bool or type(self.risk_increase) is not bool:
            raise ValueError("connection risk flags must be booleans")
        for field_name in ("reason", "operation"):
            if type(getattr(self, field_name)) is not str:
                raise ValueError(f"{field_name} must be a string")
        if self.passed and (self.reason or self.operation):
            raise ValueError("passed connection risk decisions must not carry failure diagnostics")
        if not self.passed and (not self.reason.strip() or not self.operation.strip()):
            raise ValueError("failed connection risk decisions require reason and operation")
