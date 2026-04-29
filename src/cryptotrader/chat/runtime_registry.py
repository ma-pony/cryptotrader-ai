"""Per-session runtime object registry.

LangGraph's MemorySaver checkpointer serializes the full state via msgpack
on every node transition. Live runtime objects (``EventBus``,
``RedisStateManager``) are not msgpack-serializable, so they cannot live in
``state["metadata"]``. This registry lets ``analysis_runner`` stash them by
session_id and lets nodes look them up.

Entries are removed automatically by ``analysis_runner`` in its ``finally``
block. As a safety net we also support manual ``unregister``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cryptotrader.chat.event_bus import EventBus
    from cryptotrader.risk.state import RedisStateManager


_runtime: dict[str, dict[str, Any]] = {}


def register(
    session_id: str, *, event_bus: EventBus | None = None, redis_state_manager: RedisStateManager | None = None
) -> None:
    bag = _runtime.setdefault(session_id, {})
    if event_bus is not None:
        bag["event_bus"] = event_bus
    if redis_state_manager is not None:
        bag["redis_state_manager"] = redis_state_manager


def unregister(session_id: str) -> None:
    _runtime.pop(session_id, None)


def get_event_bus(session_id: str | None) -> EventBus | None:
    if not session_id:
        return None
    return _runtime.get(session_id, {}).get("event_bus")


def get_redis_state_manager(session_id: str | None) -> RedisStateManager | None:
    if not session_id:
        return None
    return _runtime.get(session_id, {}).get("redis_state_manager")
