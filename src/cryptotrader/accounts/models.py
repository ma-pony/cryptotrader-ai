"""Immutable account facts. Unknown values are never represented by zero."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime  # noqa: TC003 -- Pydantic resolves operation timestamps at runtime.
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from cryptotrader.pair import Pair


class ExitPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation_id: str
    version: int
    connection_id: str
    book_id: str | None
    capital_scope: Literal["simulated", "real"]
    pair: str
    kind: Literal["cancel_orders", "flatten"]
    stopped_scope: list[str]
    ordinary_order_ids: list[str]
    position_amount: Decimal
    close_amount: Decimal
    protection_ids: list[str]
    snapshot_time: datetime


class OperationOrderOut(BaseModel):
    id: str
    pair: str
    side: str
    order_type: str
    amount: Decimal
    filled_amount: Decimal
    average_price: Decimal | None
    status: str
    reduce_only: bool
    client_order_id: str | None


class AccountOperationResult(BaseModel):
    canceled_order_ids: list[str] = Field(default_factory=list)
    canceled_protection_ids: list[str] = Field(default_factory=list)
    orders: list[OperationOrderOut] = Field(default_factory=list)
    remaining_position: Decimal | None = None
    remaining_order_ids: list[str] = Field(default_factory=list)
    remaining_protection_ids: list[str] = Field(default_factory=list)
    observed_at: datetime | None = None
    failure_reason: str | None = None
    reconciliation_required: bool = False


class AccountOperationOut(BaseModel):
    operation_id: str
    connection_id: str
    pair: str
    kind: Literal["cancel_orders", "flatten"]
    status: Literal["preparing", "awaiting_confirmation", "executing", "completed", "failed", "invalidated"]
    created_at: datetime
    updated_at: datetime
    plan: ExitPlan | None = None
    result: AccountOperationResult = Field(default_factory=AccountOperationResult)


@dataclass(frozen=True)
class Money:
    amount: Decimal | None
    currency: str
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.currency:
            raise ValueError("money currency is required")
        if self.amount is None:
            if not self.unavailable_reason:
                raise ValueError("unknown money requires an unavailable reason")
        elif not isinstance(self.amount, Decimal) or not self.amount.is_finite():
            raise ValueError("money amount must be a finite Decimal")
        elif self.unavailable_reason is not None:
            raise ValueError("known money cannot have an unavailable reason")


@dataclass(frozen=True)
class Instrument:
    venue_symbol: str
    pair: Pair | None
    market_type: str
    tradable: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.venue_symbol or (self.tradable and self.pair is None):
            raise ValueError("tradable instruments require a normalized pair")
        if not self.tradable and not self.reason:
            raise ValueError("unavailable instrument requires a reason")


@dataclass(frozen=True)
class AccountPosition:
    instrument: Instrument
    signed_amount: Decimal
    available_amount: Decimal | None
    signed_notional: Money
    entry_price: Decimal | None
    unrealized_pnl: Money


@dataclass(frozen=True)
class AccountOrder:
    connection_id: str
    venue_order_id: str
    instrument: Instrument
    side: str
    order_type: str
    amount: Decimal
    filled_amount: Decimal
    average_price: Decimal | None
    status: str
    reduce_only: bool
    protection: bool
    client_order_id: str | None
    observed_at: datetime
    remaining_notional: Money

    def __post_init__(self) -> None:
        if self.remaining_notional.amount is not None and self.remaining_notional.amount < 0:
            raise ValueError("remaining order notional cannot be negative")


@dataclass(frozen=True)
class AccountSnapshot:
    connection_id: str
    observed_at: datetime
    capital_scope: Literal["simulated", "real"]
    equity: Money
    balances: tuple[Money, ...]
    positions: tuple[AccountPosition, ...]
    orders: tuple[AccountOrder, ...]
    used_margin: Money
    available_margin: Money
    completeness: tuple[str, ...]
    valuation_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for key in ("balances", "positions", "orders", "completeness", "valuation_notes"):
            object.__setattr__(self, key, tuple(getattr(self, key)))


@dataclass(frozen=True)
class Fill:
    connection_id: str
    venue_fill_id: str
    venue_order_id: str
    instrument: Instrument
    side: str
    amount: Decimal
    price: Decimal
    occurred_at: datetime
    fee: Money  # Positive expense; negative rebate.
    realized_pnl: Money  # Gross trading P&L; never includes fee/funding twice.
    source: Literal["platform", "local_calculation"]
    client_order_id: str | None = None


@dataclass(frozen=True)
class FundingEntry:
    connection_id: str
    venue_entry_id: str
    instrument: Instrument
    amount: Money  # Positive income; negative expense.
    occurred_at: datetime


@dataclass(frozen=True)
class FillPage:
    items: tuple[Fill, ...]
    next_cursor: str | None
    complete: bool
    coverage_start: datetime | None = None
    coverage_end: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))


@dataclass(frozen=True)
class FundingPage:
    items: tuple[FundingEntry, ...]
    next_cursor: str | None
    complete: bool
    coverage_start: datetime | None = None
    coverage_end: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))


def external_order_source(client_order_id: str | None, owned_ids: set[str]) -> str:
    """Only exact ownership evidence can classify an order as strategy-owned."""
    return "strategy" if client_order_id in owned_ids else "external"
