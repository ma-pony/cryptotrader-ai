"""Immutable execution-book configuration domain objects."""

from __future__ import annotations

import math
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
