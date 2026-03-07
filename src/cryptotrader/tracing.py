"""Request tracing with structlog."""

import uuid

import structlog


def set_trace_id(trace_id: str | None = None) -> str:
    """Set trace ID for current context."""
    tid = trace_id or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(trace_id=tid)
    return tid


def get_trace_id() -> str | None:
    """Get current trace ID from structlog context."""
    return structlog.contextvars.get_contextvars().get("trace_id")
