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
        self._observers: list[CycleEventSink] = []
        self._subscribers: ContextVar[tuple[CycleEventSink, ...]] = ContextVar(
            f"cycle_event_subscribers_{id(self)}",
            default=(),
        )
        self._identity: ContextVar[tuple[str, int] | None] = ContextVar(
            f"cycle_event_identity_{id(self)}",
            default=None,
        )

    @contextmanager
    def cycle(self, cycle_id: str, config_revision: int):
        token = self._identity.set((cycle_id, config_revision))
        try:
            yield
        finally:
            self._identity.reset(token)

    def observe(self, sink: CycleEventSink) -> None:
        """Process-local observers only wake independent owners; no trading dependency."""
        self._observers.append(sink)

    @contextmanager
    def route(self, sink: CycleEventSink):
        current = self._subscribers.get()
        token = self._subscribers.set((*current, sink))
        try:
            yield
        finally:
            self._subscribers.reset(token)

    async def publish(self, event: CycleEvent) -> None:
        identity = self._identity.get()
        if identity is not None:
            cycle_id, config_revision = identity
            event = CycleEvent(
                event.name,
                {**event.data, "cycle_id": cycle_id, "config_revision": config_revision},
                event.timestamp,
            )
        sinks = (self.base, *self._subscribers.get(), *self._observers)
        unique = tuple(sink for index, sink in enumerate(sinks) if all(sink is not prior for prior in sinks[:index]))
        results = await asyncio.gather(*(sink.publish(event) for sink in unique), return_exceptions=True)
        base_result = results[0]
        if isinstance(base_result, BaseException):
            raise base_result
        routed_control_flow = next(
            (
                result
                for result in results[1:]
                if isinstance(result, BaseException) and not isinstance(result, Exception)
            ),
            None,
        )
        if routed_control_flow is not None:
            raise routed_control_flow
