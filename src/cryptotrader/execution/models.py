"""Immutable execution-book configuration domain objects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from cryptotrader.pair import Pair
from cryptotrader.venues.models import VenueCapabilities, VenueQuote

if TYPE_CHECKING:
    from cryptotrader.risk.models import BookRiskDecision, ConnectionRiskDecision

CapitalScope = Literal["simulated", "real"]


@dataclass(frozen=True)
class ConnectionAllocation:
    """The configured share of one connection within an execution book."""

    connection_id: str
    enabled: bool
    weight: float

    def __post_init__(self) -> None:
        if type(self.connection_id) is not str or not self.connection_id.strip():
            raise ValueError("connection_id must not be empty")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(self.weight) is not float or not math.isfinite(self.weight) or not 0.0 <= self.weight <= 1.0:
            raise ValueError("allocation weight must be a finite float in [0, 1]")


@dataclass(frozen=True)
class ExecutionBook:
    """An isolated capital, risk, and approval scope."""

    id: str
    label: str
    capital_scope: CapitalScope
    enabled: bool
    hitl_required: bool
    allocations: tuple[ConnectionAllocation, ...]

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id.strip():
            raise ValueError("execution book id must be a non-empty string")
        if type(self.label) is not str or not self.label.strip():
            raise ValueError("execution book id and label must not be empty")
        if type(self.capital_scope) is not str or self.capital_scope not in {"simulated", "real"}:
            raise ValueError("unsupported capital_scope")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(self.hitl_required) is not bool:
            raise ValueError("hitl_required must be a boolean")
        if type(self.allocations) is not tuple or not all(
            isinstance(allocation, ConnectionAllocation) for allocation in self.allocations
        ):
            raise ValueError("allocations must be a tuple of ConnectionAllocation")


@dataclass(frozen=True)
class ConnectionTarget:
    """One connection's deterministic share of a platform-neutral book target."""

    book_id: str
    connection_id: str
    weight: Decimal
    book_equity: Decimal
    target_exposure: Decimal
    target_signed_notional: Decimal

    def __post_init__(self) -> None:
        if type(self.book_id) is not str or not self.book_id.strip():
            raise ValueError("book_id must be a non-empty string")
        if type(self.connection_id) is not str or not self.connection_id.strip():
            raise ValueError("connection_id must be a non-empty string")
        if not isinstance(self.weight, Decimal) or not self.weight.is_finite() or not Decimal("0") <= self.weight <= 1:
            raise ValueError("weight must be a finite Decimal in [0, 1]")
        if not isinstance(self.book_equity, Decimal) or not self.book_equity.is_finite() or self.book_equity < 0:
            raise ValueError("book_equity must be a non-negative finite Decimal")
        if (
            not isinstance(self.target_exposure, Decimal)
            or not self.target_exposure.is_finite()
            or not Decimal("-1") <= self.target_exposure <= 1
        ):
            raise ValueError("target_exposure must be a finite Decimal in [-1, 1]")
        if not isinstance(self.target_signed_notional, Decimal) or not self.target_signed_notional.is_finite():
            raise ValueError("target_signed_notional must be a finite Decimal")
        expected = self.book_equity * self.target_exposure * self.weight
        if self.target_signed_notional != expected:
            raise ValueError("target_signed_notional must equal book_equity * target_exposure * weight")


def _require_decimal(value: object, field_name: str, *, positive: bool = False) -> None:
    if not isinstance(value, Decimal):
        raise ValueError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be a finite Decimal")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")


@dataclass(frozen=True)
class ConnectionExecutionPlan:
    """One exact connection delta using only normalized venue values."""

    book_id: str
    connection_id: str
    pair: Pair
    current_signed_notional: Decimal
    target_signed_notional: Decimal
    delta_signed_notional: Decimal
    quote: VenueQuote
    execution_price: Decimal
    amount: Decimal
    side: Literal["buy", "sell"]
    reduce_only: bool
    market_type: str
    stop_loss: Decimal | None
    take_profit: Decimal | None
    old_protection_ids: tuple[str, ...]
    capabilities: VenueCapabilities

    def __post_init__(self) -> None:
        self._validate_identity()
        self._validate_delta()
        self._validate_execution()
        self._validate_protection()

    def _validate_identity(self) -> None:
        for field_name in ("book_id", "connection_id"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.pair, Pair):
            raise ValueError("pair must be a Pair")

    def _validate_delta(self) -> None:
        for field_name in (
            "current_signed_notional",
            "target_signed_notional",
            "delta_signed_notional",
        ):
            _require_decimal(getattr(self, field_name), field_name)
        if self.delta_signed_notional == 0:
            raise ValueError("delta_signed_notional must not be zero")
        if self.delta_signed_notional != self.target_signed_notional - self.current_signed_notional:
            raise ValueError("delta_signed_notional must equal target minus current")

    def _validate_execution(self) -> None:
        if not isinstance(self.quote, VenueQuote) or self.quote.pair != self.pair:
            raise ValueError("quote must match the execution pair")
        _require_decimal(self.execution_price, "execution_price", positive=True)
        _require_decimal(self.amount, "amount", positive=True)
        if self.side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        if (self.delta_signed_notional > 0) != (self.side == "buy"):
            raise ValueError("side must match delta_signed_notional")
        if type(self.reduce_only) is not bool:
            raise ValueError("reduce_only must be a boolean")
        if type(self.market_type) is not str or self.market_type != self.pair.market_type:
            raise ValueError("market_type must match the execution pair")

    def _validate_protection(self) -> None:
        for field_name in ("stop_loss", "take_profit"):
            value = getattr(self, field_name)
            if value is not None:
                _require_decimal(value, field_name, positive=True)
        if type(self.old_protection_ids) is not tuple or not all(
            type(protection_id) is str and bool(protection_id.strip()) for protection_id in self.old_protection_ids
        ):
            raise ValueError("old_protection_ids must be a tuple of non-empty strings")
        if len(self.old_protection_ids) != len(set(self.old_protection_ids)):
            raise ValueError("old_protection_ids must be unique")
        if not isinstance(self.capabilities, VenueCapabilities):
            raise ValueError("capabilities must be VenueCapabilities")


@dataclass(frozen=True)
class BookExecutionProposal:
    """An immutable whole-book proposal with explicit unavailable connections."""

    book_id: str
    capital_scope: CapitalScope
    config_revision: int
    pair: Pair
    requested_target_exposure: Decimal
    target_exposure: Decimal
    risk: BookRiskDecision
    connection_risks: tuple[ConnectionRiskDecision, ...]
    connection_plans: tuple[ConnectionExecutionPlan, ...]
    unavailable_connections: tuple[str, ...]
    errors: tuple[str, ...]
    ready: bool

    def __post_init__(self) -> None:
        self._validate_identity()
        self._validate_risk()
        self._validate_connection_results()
        self._validate_status()

    def _validate_identity(self) -> None:
        if type(self.book_id) is not str or not self.book_id.strip():
            raise ValueError("book_id must be a non-empty string")
        if self.capital_scope not in {"simulated", "real"}:
            raise ValueError("unsupported capital_scope")
        if type(self.config_revision) is not int or self.config_revision < 0:
            raise ValueError("config_revision must be a non-negative integer")
        if not isinstance(self.pair, Pair):
            raise ValueError("pair must be a Pair")

    def _validate_risk(self) -> None:
        from cryptotrader.risk.models import BookRiskDecision, ConnectionRiskDecision

        _require_decimal(self.requested_target_exposure, "requested_target_exposure")
        _require_decimal(self.target_exposure, "target_exposure")
        if not isinstance(self.risk, BookRiskDecision):
            raise ValueError("risk must be a BookRiskDecision")
        if self.requested_target_exposure != self.risk.requested_target_exposure:
            raise ValueError("requested_target_exposure must match the risk decision")
        if self.target_exposure != self.risk.capped_target_exposure:
            raise ValueError("target_exposure must match the risk decision")
        if type(self.connection_risks) is not tuple or not all(
            isinstance(item, ConnectionRiskDecision) for item in self.connection_risks
        ):
            raise ValueError("connection_risks must be a tuple of ConnectionRiskDecision")

    def _validate_connection_results(self) -> None:
        if type(self.connection_plans) is not tuple or not all(
            isinstance(item, ConnectionExecutionPlan) for item in self.connection_plans
        ):
            raise ValueError("connection_plans must be a tuple of ConnectionExecutionPlan")
        if any(plan.book_id != self.book_id or plan.pair != self.pair for plan in self.connection_plans):
            raise ValueError("connection plans must belong to the proposal")
        for field_name in ("unavailable_connections", "errors"):
            values = getattr(self, field_name)
            if type(values) is not tuple or not all(type(value) is str and bool(value.strip()) for value in values):
                raise ValueError(f"{field_name} must be a tuple of non-empty strings")
        if len(self.unavailable_connections) != len(set(self.unavailable_connections)):
            raise ValueError("unavailable_connections must be unique")

    def _validate_status(self) -> None:
        if type(self.ready) is not bool:
            raise ValueError("ready must be a boolean")
        if self.ready and not self.risk.passed:
            raise ValueError("a ready proposal requires passed book risk")

    @property
    def weights(self) -> tuple[Decimal, ...]:
        return self.risk.connection_weights
