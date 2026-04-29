"""SSE envelope and event bus — publish/subscribe for real-time analysis events."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from cryptotrader._compat import UTC

if TYPE_CHECKING:
    from cryptotrader.chat.event_buffer import EventBuffer

logger = logging.getLogger(__name__)


@dataclass
class SSEEnvelope:
    event_id: int
    type: str
    ts: str
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "ts": self.ts,
            "session_id": self.session_id,
            "data": self.data,
        }

    @staticmethod
    def now_iso() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def to_sse_frame(envelope: SSEEnvelope) -> str:
        payload = json.dumps(envelope.to_dict(), ensure_ascii=False)
        return f"id: {envelope.event_id}\nevent: {envelope.type}\ndata: {payload}\n\n"


class EventBus:
    """Publishes SSE events to both a persistent buffer and a live queue."""

    def __init__(self, session_id: str, buffer: EventBuffer) -> None:
        self._session_id = session_id
        self._buffer = buffer
        self._subscribers: list[asyncio.Queue[SSEEnvelope]] = []

    async def publish(self, event_type: str, data: dict[str, Any] | None = None) -> SSEEnvelope:
        eid = await self._buffer.next_event_id()
        envelope = SSEEnvelope(
            event_id=eid,
            type=event_type,
            ts=SSEEnvelope.now_iso(),
            session_id=self._session_id,
            data=data or {},
        )
        await self._buffer.push(envelope)
        for q in self._subscribers:
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue full for session %s, dropping event", self._session_id)
        return envelope

    def subscribe(self) -> asyncio.Queue[SSEEnvelope]:
        q: asyncio.Queue[SSEEnvelope] = asyncio.Queue(maxsize=200)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[SSEEnvelope]) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers.remove(q)
