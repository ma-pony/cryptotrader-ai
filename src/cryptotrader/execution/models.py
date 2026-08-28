"""Immutable execution-book configuration domain objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CapitalScope = Literal["simulated", "real"]


@dataclass(frozen=True)
class ConnectionAllocation:
    """The configured share of one connection within an execution book."""

    connection_id: str
    enabled: bool
    weight: float

    def __post_init__(self) -> None:
        if not self.connection_id.strip():
            raise ValueError("connection_id must not be empty")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("allocation weight must be in [0, 1]")


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
        if not self.id.strip() or not self.label.strip():
            raise ValueError("execution book id and label must not be empty")
        if self.capital_scope not in {"simulated", "real"}:
            raise ValueError("unsupported capital_scope")
