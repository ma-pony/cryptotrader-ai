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
        if type(self.id) is not str or not self.id.strip():
            raise ValueError("venue connection id must be a non-empty string")
        if type(self.label) is not str or not self.label.strip():
            raise ValueError("venue connection label must be a non-empty string")
        if type(self.adapter_id) is not str or not self.adapter_id.strip():
            raise ValueError("venue connection id, label, and adapter_id must not be empty")
        if type(self.environment) is not str or self.environment not in {"paper", "demo", "testnet", "live"}:
            raise ValueError("unsupported connection environment")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if self.credential_ref is not None and type(self.credential_ref) is not str:
            raise ValueError("credential_ref must be a string or None")
        if type(self.leverage) is not int or self.leverage < 1:
            raise ValueError("leverage must be at least one")
        if type(self.margin_mode) is not str or self.margin_mode not in {"isolated", "cross"}:
            raise ValueError("unsupported margin_mode")
