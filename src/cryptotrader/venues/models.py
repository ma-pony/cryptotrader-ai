"""Immutable configuration for executable venue connections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ConnectionEnvironment = Literal["paper", "demo", "testnet", "live"]
MarginMode = Literal["isolated", "cross"]


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
        if not self.id.strip() or not self.label.strip() or not self.adapter_id.strip():
            raise ValueError("venue connection id, label, and adapter_id must not be empty")
        if self.environment not in {"paper", "demo", "testnet", "live"}:
            raise ValueError("unsupported connection environment")
        if self.leverage < 1:
            raise ValueError("leverage must be at least one")
        if self.margin_mode not in {"isolated", "cross"}:
            raise ValueError("unsupported margin_mode")
