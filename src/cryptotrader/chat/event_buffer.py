"""Event buffer — persistent ring buffer backed by Redis List (or in-memory fallback)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.chat.event_bus import SSEEnvelope
    from cryptotrader.risk.state import RedisStateManager

logger = logging.getLogger(__name__)


class EventBuffer:
    """Append-only event buffer with TTL and overflow protection."""

    def __init__(
        self,
        session_id: str,
        state_mgr: RedisStateManager,
        ttl_s: int = 300,
        max_size: int = 500,
    ) -> None:
        self._session_id = session_id
        self._state = state_mgr
        self._ttl_s = ttl_s
        self._max_size = max_size
        self._key = f"analysis:events:{session_id}"
        self._seq_key = f"analysis:event_seq:{session_id}"

    async def next_event_id(self) -> int:
        val = await self._state.incr(self._seq_key)
        await self._state.expire(self._seq_key, self._ttl_s)
        return val if val is not None else 1

    async def push(self, envelope: SSEEnvelope) -> None:
        data = json.dumps(envelope.to_dict(), ensure_ascii=False)
        current_len = await self._state.buffer_len(self._key)
        if current_len >= self._max_size:
            logger.warning(
                "EventBuffer overflow for session %s (size=%d, max=%d)",
                self._session_id,
                current_len,
                self._max_size,
            )
        await self._state.buffer_push(self._key, data, self._max_size, self._ttl_s)

    async def range_after(self, last_event_id: int) -> list[SSEEnvelope]:
        from cryptotrader.chat.event_bus import SSEEnvelope as Env

        raw = await self._state.buffer_range(self._key, 0, -1)
        result: list[SSEEnvelope] = []
        for item in raw:
            try:
                d = json.loads(item)
                env = Env(**d)
                if env.event_id > last_event_id:
                    result.append(env)
            except (json.JSONDecodeError, TypeError):
                logger.debug("Skipping malformed event in buffer for %s", self._session_id)
        return result

    async def set_ttl(self, ttl_s: int) -> None:
        self._ttl_s = ttl_s
        await self._state.buffer_set_ttl(self._key, ttl_s)
        await self._state.expire(self._seq_key, ttl_s)

    async def mark_done(self, event_type: str = "stream_done") -> None:
        from cryptotrader.chat.event_bus import SSEEnvelope as Env

        eid = await self.next_event_id()
        env = Env(
            event_id=eid,
            type=event_type,
            ts=Env.now_iso(),
            session_id=self._session_id,
            data={},
        )
        await self.push(env)

    async def exists(self) -> bool:
        length = await self._state.buffer_len(self._key)
        return length > 0
