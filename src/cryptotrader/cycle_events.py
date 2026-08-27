"""Typed business events emitted by the shared trading cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class CycleEvent:
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("cycle event name must not be empty")


class CycleEventSink(Protocol):
    async def publish(self, event: CycleEvent) -> None: ...


class NullCycleEventSink:
    async def publish(self, event: CycleEvent) -> None:
        return None
