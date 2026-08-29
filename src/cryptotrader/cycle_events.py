"""Typed business events emitted by the shared trading cycle."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
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


class MultiplexedCycleEventSink:
    """Publish globally and to context-local subscribers inherited by child tasks."""

    def __init__(self, base: CycleEventSink) -> None:
        self.base = base
        self._subscribers: ContextVar[tuple[CycleEventSink, ...]] = ContextVar(
            f"cycle_event_subscribers_{id(self)}",
            default=(),
        )

    @contextmanager
    def route(self, sink: CycleEventSink):
        current = self._subscribers.get()
        token = self._subscribers.set((*current, sink))
        try:
            yield
        finally:
            self._subscribers.reset(token)

    async def publish(self, event: CycleEvent) -> None:
        sinks = (self.base, *self._subscribers.get())
        unique = tuple(sink for index, sink in enumerate(sinks) if all(sink is not prior for prior in sinks[:index]))
        await asyncio.gather(*(sink.publish(event) for sink in unique))
