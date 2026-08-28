"""Immutable execution-book configuration domain objects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from cryptotrader.pair import Pair
from cryptotrader.venues.models import ProtectionSpec, VenueCapabilities, VenueQuote

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
    current_signed_amount: Decimal
    target_signed_amount: Decimal
    delta_signed_amount: Decimal
    post_fill_signed_amount: Decimal
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
            "current_signed_amount",
            "target_signed_amount",
            "delta_signed_amount",
            "post_fill_signed_amount",
        ):
            _require_decimal(getattr(self, field_name), field_name)
        if self.delta_signed_notional == 0:
            raise ValueError("delta_signed_notional must not be zero")
        if self.delta_signed_notional != self.target_signed_notional - self.current_signed_notional:
            raise ValueError("delta_signed_notional must equal target minus current")
        if self.delta_signed_amount == 0:
            raise ValueError("delta_signed_amount must not be zero")
        if self.delta_signed_amount != self.target_signed_amount - self.current_signed_amount:
            raise ValueError("delta_signed_amount must equal target minus current")
        self._require_matching_sign(self.current_signed_notional, self.current_signed_amount, "current")
        self._require_matching_sign(self.target_signed_notional, self.target_signed_amount, "target")

    def _validate_execution(self) -> None:
        self._validate_quote_and_side()
        self._validate_amount_transition()
        self._validate_execution_capabilities()

    def _validate_quote_and_side(self) -> None:
        if not isinstance(self.quote, VenueQuote) or self.quote.pair != self.pair:
            raise ValueError("quote must match the execution pair")
        _require_decimal(self.execution_price, "execution_price", positive=True)
        _require_decimal(self.amount, "amount", positive=True)
        if self.side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        if (self.delta_signed_notional > 0) != (self.side == "buy"):
            raise ValueError("side must match delta_signed_notional")
        if (self.delta_signed_amount > 0) != (self.side == "buy"):
            raise ValueError("side must match delta_signed_amount")
        expected_price = self.quote.ask if self.side == "buy" else self.quote.bid
        if self.execution_price != expected_price:
            raise ValueError(f"{self.side} execution price must equal quote {'ask' if self.side == 'buy' else 'bid'}")

    def _validate_amount_transition(self) -> None:
        expected_target_amount = (
            Decimal("0") if self.target_signed_notional == 0 else self.target_signed_notional / self.execution_price
        )
        if self.target_signed_amount != expected_target_amount:
            raise ValueError("target_signed_amount must use target notional and execution price")
        if self.amount > abs(self.delta_signed_amount):
            raise ValueError("amount must not exceed the requested base delta")
        signed_fill = self.amount if self.side == "buy" else -self.amount
        if self.post_fill_signed_amount != self.current_signed_amount + signed_fill:
            raise ValueError("post_fill_signed_amount must equal current amount plus normalized order amount")
        if self.side == "buy" and not (
            self.current_signed_amount < self.post_fill_signed_amount <= self.target_signed_amount
        ):
            raise ValueError("buy amount must not cross the target signed amount")
        if self.side == "sell" and not (
            self.current_signed_amount > self.post_fill_signed_amount >= self.target_signed_amount
        ):
            raise ValueError("sell amount must not cross the target signed amount")
        if self.target_signed_amount == 0 and self.post_fill_signed_amount != 0:
            raise ValueError("flat target requires an exact base-amount close")
        if self.target_signed_amount != 0 and self.target_signed_amount * self.post_fill_signed_amount <= 0:
            raise ValueError("post-fill amount must reach the target position side")
        if type(self.reduce_only) is not bool:
            raise ValueError("reduce_only must be a boolean")
        expected_reduce_only = self._is_reduction(self.current_signed_amount, self.target_signed_amount)
        if self.reduce_only is not expected_reduce_only:
            raise ValueError("reduce_only must match the current-to-target transition")

    def _validate_execution_capabilities(self) -> None:
        if type(self.market_type) is not str or self.market_type != self.pair.market_type:
            raise ValueError("market_type must match the execution pair")
        if not isinstance(self.capabilities, VenueCapabilities):
            raise ValueError("capabilities must be VenueCapabilities")
        if self.market_type not in self.capabilities.market_types:
            raise ValueError("market type must be supported by capabilities")
        if "market" not in self.capabilities.supported_order_types:
            raise ValueError("market order must be supported by capabilities")
        if self.reduce_only and not self.capabilities.reduce_only:
            raise ValueError("reduce_only plan requires reduce-only capability")

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
        target_is_flat = self.target_signed_amount == 0
        if target_is_flat:
            if self.stop_loss is not None or self.take_profit is not None:
                raise ValueError("flat plan must not carry protection prices")
            return
        has_protection = self.stop_loss is not None or self.take_profit is not None
        if self.market_type != "spot" and not has_protection:
            raise ValueError("non-flat derivative plan requires protection")
        if has_protection:
            if not self.capabilities.native_protection:
                raise ValueError("protection plan requires native protection capability")
            spec = ProtectionSpec(
                self.pair,
                "long" if self.target_signed_amount > 0 else "short",
                abs(self.post_fill_signed_amount),
                self.stop_loss,
                self.take_profit,
            )
            spec.validate_geometry(self.execution_price)

    @staticmethod
    def _require_matching_sign(notional: Decimal, amount: Decimal, label: str) -> None:
        if (notional == 0) != (amount == 0) or notional * amount < 0:
            raise ValueError(f"{label} signed notional and amount must have matching signs")

    @staticmethod
    def _is_reduction(current: Decimal, target: Decimal) -> bool:
        return target == 0 or (current * target > 0 and abs(target) < abs(current))


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
        if any(target.book_id != self.book_id for target in self.risk.connection_targets):
            raise ValueError("connection risk targets must belong to the proposal book")
        if self.requested_target_exposure != self.risk.requested_target_exposure:
            raise ValueError("requested_target_exposure must match the risk decision")
        if self.target_exposure != self.risk.capped_target_exposure:
            raise ValueError("target_exposure must match the risk decision")
        if type(self.connection_risks) is not tuple or not all(
            isinstance(item, ConnectionRiskDecision) for item in self.connection_risks
        ):
            raise ValueError("connection_risks must be a tuple of ConnectionRiskDecision")

    def _validate_connection_results(self) -> None:
        self._validate_result_shapes()
        self._validate_result_id_sets()
        self._validate_result_diagnostics()

    def _validate_result_shapes(self) -> None:
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

    def _validate_result_id_sets(self) -> None:
        expected_ids = tuple(target.connection_id for target in self.risk.connection_targets)
        risk_ids = tuple(item.connection_id for item in self.connection_risks)
        if (self.risk.passed or self.connection_risks) and risk_ids != expected_ids:
            raise ValueError("connection risks must match configured order")
        plan_ids = tuple(plan.connection_id for plan in self.connection_plans)
        if len(plan_ids) != len(set(plan_ids)) or not self._is_ordered_subset(plan_ids, expected_ids):
            raise ValueError("connection plans must be unique and preserve configured order")
        targets_by_id = {target.connection_id: target for target in self.risk.connection_targets}
        if any(
            plan.target_signed_notional != targets_by_id[plan.connection_id].target_signed_notional
            for plan in self.connection_plans
        ):
            raise ValueError("connection plan target notional must match its risk target")
        if not self._is_ordered_subset(self.unavailable_connections, expected_ids):
            raise ValueError("unavailable connections must preserve configured order")
        if set(plan_ids).intersection(self.unavailable_connections):
            raise ValueError("planned and unavailable connection IDs must be disjoint")
        passed_ids = {item.connection_id for item in self.connection_risks if item.passed}
        if not set(plan_ids) <= passed_ids:
            raise ValueError("connection plans require passed connection risk decisions")

    def _validate_result_diagnostics(self) -> None:
        failed = tuple(item for item in self.connection_risks if not item.passed)
        expected_errors = tuple(f"connection {item.connection_id}: {item.operation} failed" for item in failed)
        if self.errors != expected_errors:
            raise ValueError("errors must match failed connection decisions in configured order")
        failed_by_id = {item.connection_id: item for item in failed}
        venue_failure_operations = {"session", "fetch_quote", "list_open_state", "normalize_amount"}
        if any(
            connection_id not in failed_by_id or failed_by_id[connection_id].operation not in venue_failure_operations
            for connection_id in self.unavailable_connections
        ):
            raise ValueError("unavailable connections must identify venue operation failures")

    def _validate_status(self) -> None:
        if type(self.ready) is not bool:
            raise ValueError("ready must be a boolean")
        if self.ready and not self.risk.passed:
            raise ValueError("a ready proposal requires passed book risk")
        if not self.ready and self.connection_plans:
            raise ValueError("a not-ready proposal must have zero connection plans")

    @staticmethod
    def _is_ordered_subset(values: tuple[str, ...], expected: tuple[str, ...]) -> bool:
        indices = [expected.index(value) for value in values if value in expected]
        return len(indices) == len(values) and indices == sorted(indices)

    @property
    def weights(self) -> tuple[Decimal, ...]:
        return self.risk.connection_weights
