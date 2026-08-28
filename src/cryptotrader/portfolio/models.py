"""Immutable portfolio snapshots shared by venue sessions and aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

from cryptotrader.venues.models import ConnectionPosition

_CAPITAL_SCOPES = {"simulated", "real"}


@dataclass(frozen=True)
class ConnectionPortfolioSnapshot:
    """Normalized equity and position for one venue connection."""

    connection_id: str
    equity: Decimal
    balances: Mapping[str, Decimal]
    position: ConnectionPosition

    def __post_init__(self) -> None:
        if type(self.connection_id) is not str or not self.connection_id.strip():
            raise ValueError("connection_id must be a non-empty string")
        if not isinstance(self.equity, Decimal) or not self.equity.is_finite() or self.equity < 0:
            raise ValueError("equity must be a non-negative finite Decimal")
        if not isinstance(self.balances, Mapping):
            raise ValueError("balances must be a mapping")
        normalized: dict[str, Decimal] = {}
        for asset, amount in self.balances.items():
            if type(asset) is not str or not asset.strip():
                raise ValueError("balance asset must be a non-empty string")
            if not isinstance(amount, Decimal) or not amount.is_finite():
                raise ValueError("balance amount must be a finite Decimal")
            normalized[asset] = amount
        object.__setattr__(self, "balances", MappingProxyType(normalized))
        if not isinstance(self.position, ConnectionPosition):
            raise ValueError("position must be a ConnectionPosition")


@dataclass(frozen=True)
class BookPortfolioSnapshot:
    """An exact aggregate of the enabled connections in one execution book."""

    book_id: str
    capital_scope: str
    total_equity: Decimal
    total_signed_notional: Decimal
    connections: tuple[ConnectionPortfolioSnapshot, ...]

    def __post_init__(self) -> None:
        if type(self.book_id) is not str or not self.book_id.strip():
            raise ValueError("book_id must be a non-empty string")
        if type(self.capital_scope) is not str or self.capital_scope not in _CAPITAL_SCOPES:
            raise ValueError("capital_scope must be simulated or real")
        if not isinstance(self.total_equity, Decimal) or not self.total_equity.is_finite():
            raise ValueError("total_equity must be a finite Decimal")
        if self.total_equity < 0:
            raise ValueError("total_equity must be non-negative")
        if not isinstance(self.total_signed_notional, Decimal) or not self.total_signed_notional.is_finite():
            raise ValueError("total_signed_notional must be a finite Decimal")
        if type(self.connections) is not tuple or not all(
            isinstance(connection, ConnectionPortfolioSnapshot) for connection in self.connections
        ):
            raise ValueError("connections must be a tuple of ConnectionPortfolioSnapshot")
        connection_ids = tuple(connection.connection_id for connection in self.connections)
        if len(connection_ids) != len(set(connection_ids)):
            raise ValueError("connections must have unique connection IDs")
        equity = sum((connection.equity for connection in self.connections), Decimal("0"))
        if self.total_equity != equity:
            raise ValueError("total_equity must equal the sum of connection equity")
        signed_notional = sum(
            (connection.position.signed_notional for connection in self.connections),
            Decimal("0"),
        )
        if self.total_signed_notional != signed_notional:
            raise ValueError("total_signed_notional must equal the sum of connection signed notional")
