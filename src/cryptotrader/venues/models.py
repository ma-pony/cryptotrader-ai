"""Immutable, platform-neutral venue domain values."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from cryptotrader.pair import MarketType, Pair

ConnectionEnvironment = Literal["paper", "demo", "testnet", "live"]
MarginMode = Literal["isolated", "cross"]
OrderSide = Literal["buy", "sell"]
PositionSide = Literal["long", "short"]

_ENVIRONMENTS = frozenset({"paper", "demo", "testnet", "live"})
_MARGIN_MODES = frozenset({"isolated", "cross"})
_MARKET_TYPES = frozenset({"spot", "swap", "future", "option"})


def _require_decimal(value: object, field_name: str, *, positive: bool = False) -> None:
    if not isinstance(value, Decimal):
        raise ValueError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be a finite Decimal")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _require_optional_decimal(value: object, field_name: str, *, positive: bool = False) -> None:
    if value is not None:
        _require_decimal(value, field_name, positive=positive)


@dataclass(frozen=True)
class VenueConnection:
    """One independently auditable account on a venue adapter."""

    id: str
    label: str
    adapter_id: str
    environment: ConnectionEnvironment
    enabled: bool
    credential_ref: str | None
    leverage: int
    margin_mode: MarginMode

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id.strip():
            raise ValueError("venue connection id must be a non-empty string")
        if type(self.label) is not str or not self.label.strip():
            raise ValueError("venue connection label must be a non-empty string")
        if type(self.adapter_id) is not str or not self.adapter_id.strip():
            raise ValueError("venue connection id, label, and adapter_id must not be empty")
        if type(self.environment) is not str or self.environment not in _ENVIRONMENTS:
            raise ValueError("unsupported connection environment")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if self.credential_ref is not None and (
            type(self.credential_ref) is not str or not self.credential_ref.strip()
        ):
            raise ValueError("credential_ref must be a non-empty string or None")
        if self.environment == "live" and self.credential_ref is None:
            raise ValueError("live connection requires credential_ref")
        if type(self.leverage) is not int or self.leverage < 1:
            raise ValueError("leverage must be at least one")
        if type(self.margin_mode) is not str or self.margin_mode not in _MARGIN_MODES:
            raise ValueError("unsupported margin_mode")


@dataclass(frozen=True)
class VenueCapabilities:
    """Execution features declared by an adapter for one environment."""

    market_types: frozenset[MarketType]
    native_protection: bool
    hedge_mode: bool
    reduce_only: bool
    supported_order_types: frozenset[str]

    def __post_init__(self) -> None:
        if type(self.market_types) is not frozenset or not self.market_types <= _MARKET_TYPES:
            raise ValueError("market_types must be a frozenset of declared market types")
        if any(type(value) is not bool for value in (self.native_protection, self.hedge_mode, self.reduce_only)):
            raise ValueError("venue capability flags must be booleans")
        if type(self.supported_order_types) is not frozenset or not all(
            type(order_type) is str and bool(order_type.strip()) for order_type in self.supported_order_types
        ):
            raise ValueError("supported_order_types must be a frozenset of non-empty strings")


@dataclass(frozen=True)
class VenueQuote:
    """One normalized venue quote whose prices retain decimal precision."""

    pair: Pair
    bid: Decimal
    ask: Decimal
    last: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.pair, Pair):
            raise ValueError("pair must be a Pair")
        _require_decimal(self.bid, "bid", positive=True)
        _require_decimal(self.ask, "ask", positive=True)
        _require_decimal(self.last, "last", positive=True)
        if self.bid > self.ask:
            raise ValueError("bid must not exceed ask")


@dataclass(frozen=True)
class ConnectionPosition:
    """A position normalized into signed amount and signed quote notional."""

    pair: Pair
    signed_amount: Decimal
    signed_notional: Decimal
    entry_price: Decimal | None

    def __post_init__(self) -> None:
        if not isinstance(self.pair, Pair):
            raise ValueError("pair must be a Pair")
        _require_decimal(self.signed_amount, "signed_amount")
        _require_decimal(self.signed_notional, "signed_notional")
        _require_optional_decimal(self.entry_price, "entry_price", positive=True)


@dataclass(frozen=True)
class OrderIntent:
    """A venue-ready order request with no platform-specific parameters."""

    pair: Pair
    side: OrderSide
    amount: Decimal
    order_type: str
    price: Decimal | None
    reduce_only: bool

    def __post_init__(self) -> None:
        if not isinstance(self.pair, Pair):
            raise ValueError("pair must be a Pair")
        if type(self.side) is not str or self.side not in {"buy", "sell"}:
            raise ValueError("unsupported order side")
        _require_decimal(self.amount, "amount", positive=True)
        if type(self.order_type) is not str or not self.order_type.strip():
            raise ValueError("order_type must be a non-empty string")
        _require_optional_decimal(self.price, "price", positive=True)
        if type(self.reduce_only) is not bool:
            raise ValueError("reduce_only must be a boolean")


@dataclass(frozen=True)
class NormalizedOrder:
    """Adapter-independent order state returned by a venue session."""

    id: str
    pair: Pair
    side: OrderSide
    order_type: str
    amount: Decimal
    filled_amount: Decimal
    average_price: Decimal | None
    status: str
    reduce_only: bool

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id.strip():
            raise ValueError("order id must be a non-empty string")
        if not isinstance(self.pair, Pair):
            raise ValueError("pair must be a Pair")
        if type(self.side) is not str or self.side not in {"buy", "sell"}:
            raise ValueError("unsupported order side")
        if type(self.order_type) is not str or not self.order_type.strip():
            raise ValueError("order_type must be a non-empty string")
        _require_decimal(self.amount, "amount", positive=True)
        _require_decimal(self.filled_amount, "filled_amount")
        if self.filled_amount < 0 or self.filled_amount > self.amount:
            raise ValueError("filled_amount must be within the order amount")
        _require_optional_decimal(self.average_price, "average_price", positive=True)
        if type(self.status) is not str or not self.status.strip():
            raise ValueError("order status must be a non-empty string")
        if type(self.reduce_only) is not bool:
            raise ValueError("reduce_only must be a boolean")


@dataclass(frozen=True)
class ProtectionSpec:
    """Desired native stop-loss and take-profit protection."""

    pair: Pair
    position_side: PositionSide
    amount: Decimal
    stop_loss: Decimal | None
    take_profit: Decimal | None

    def __post_init__(self) -> None:
        if not isinstance(self.pair, Pair):
            raise ValueError("pair must be a Pair")
        if type(self.position_side) is not str or self.position_side not in {"long", "short"}:
            raise ValueError("unsupported position_side")
        _require_decimal(self.amount, "amount", positive=True)
        _require_optional_decimal(self.stop_loss, "stop_loss", positive=True)
        _require_optional_decimal(self.take_profit, "take_profit", positive=True)
        if self.stop_loss is None and self.take_profit is None:
            raise ValueError("protection requires stop_loss or take_profit")


@dataclass(frozen=True)
class ProtectionState:
    """Normalized state for one logical native protection group."""

    protection_ids: tuple[str, ...]
    pair: Pair
    position_side: PositionSide
    amount: Decimal
    stop_loss: Decimal | None
    take_profit: Decimal | None
    active: bool
    triggered: bool

    def __post_init__(self) -> None:
        if (
            type(self.protection_ids) is not tuple
            or not self.protection_ids
            or not all(
                type(protection_id) is str and bool(protection_id.strip()) for protection_id in self.protection_ids
            )
        ):
            raise ValueError("protection_ids must be a non-empty tuple of strings")
        if not isinstance(self.pair, Pair):
            raise ValueError("pair must be a Pair")
        if type(self.position_side) is not str or self.position_side not in {"long", "short"}:
            raise ValueError("unsupported position_side")
        _require_decimal(self.amount, "amount", positive=True)
        _require_optional_decimal(self.stop_loss, "stop_loss", positive=True)
        _require_optional_decimal(self.take_profit, "take_profit", positive=True)
        if self.stop_loss is None and self.take_profit is None:
            raise ValueError("protection requires stop_loss or take_profit")
        if type(self.active) is not bool or type(self.triggered) is not bool:
            raise ValueError("protection state flags must be booleans")


@dataclass(frozen=True)
class OpenVenueState:
    """The open position, orders, and native protections on one connection."""

    position: ConnectionPosition
    open_orders: tuple[NormalizedOrder, ...]
    protections: tuple[ProtectionState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.position, ConnectionPosition):
            raise ValueError("position must be a ConnectionPosition")
        if type(self.open_orders) is not tuple or not all(
            isinstance(order, NormalizedOrder) for order in self.open_orders
        ):
            raise ValueError("open_orders must be a tuple of NormalizedOrder")
        if type(self.protections) is not tuple or not all(
            isinstance(protection, ProtectionState) for protection in self.protections
        ):
            raise ValueError("protections must be a tuple of ProtectionState")

    @property
    def triggered_protections(self) -> tuple[ProtectionState, ...]:
        """Return protections triggered while reading state, if any."""
        return tuple(protection for protection in self.protections if protection.triggered)
