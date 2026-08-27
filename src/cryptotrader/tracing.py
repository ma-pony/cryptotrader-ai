"""Trace identifiers and timing for request and component boundaries."""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import structlog

_F = TypeVar("_F", bound=Callable[..., Coroutine[Any, Any, Any]])


def set_trace_id(trace_id: str | None = None) -> str:
    """Bind a trace ID to the current structured-logging context."""
    value = trace_id or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(trace_id=value)
    return value


def get_trace_id() -> str | None:
    """Return the trace ID bound to the current structured-logging context."""
    return structlog.contextvars.get_contextvars().get("trace_id")


def node_logger() -> Callable[[_F], _F]:
    """Time an async component boundary and emit structured entry/exit logs."""

    def decorator(function: _F) -> _F:
        node_name = function.__name__

        @functools.wraps(function)
        async def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
            trace_id = metadata.get("trace_id") or get_trace_id()
            log = structlog.get_logger()
            log.info("node_entry", node=node_name, trace_id=trace_id)
            started_at = time.monotonic()
            try:
                return await function(state, *args, **kwargs)
            finally:
                duration_ms = int((time.monotonic() - started_at) * 1000)
                log.info("node_exit", node=node_name, duration_ms=duration_ms, trace_id=trace_id)

        return wrapper  # type: ignore[return-value]

    return decorator
