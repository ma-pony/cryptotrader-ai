"""Immutable portfolio snapshots shared by venue sessions and aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

from cryptotrader.venues.models import ConnectionPosition


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
