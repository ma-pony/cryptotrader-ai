"""Graph tracing — capture per-node outputs via LangGraph streaming."""

from __future__ import annotations

import functools
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import structlog

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Coroutine[Any, Any, Any]])


# Per-trace_id node-trace registry.
# `run_graph_traced` and `analysis_runner` populate this as nodes complete;
# the journal's `record_trade` / `record_rejection` nodes read from here
# because LangGraph node return-deltas can't carry the per-node timing list
# back to a sibling node within the same graph execution.
_node_trace_registry: dict[str, list[dict]] = {}


def trace_register(trace_id: str) -> None:
    """Initialize a fresh trace bucket for the given trace_id."""
    if trace_id:
        _node_trace_registry[trace_id] = []


def trace_append(trace_id: str | None, entry: dict) -> None:
    """Append a node trace entry. No-op if trace_id is unset or unregistered."""
    if not trace_id:
        return
    bucket = _node_trace_registry.get(trace_id)
    if bucket is not None:
        bucket.append(entry)


def trace_get(trace_id: str | None) -> list[dict]:
    """Return current trace entries; empty list if unset/unregistered."""
    if not trace_id:
        return []
    return list(_node_trace_registry.get(trace_id, []))


def trace_unregister(trace_id: str | None) -> None:
    """Drop the trace bucket. Call in finally after the graph completes."""
    if trace_id:
        _node_trace_registry.pop(trace_id, None)


def set_trace_id(trace_id: str | None = None) -> str:
    """Set trace ID for current context."""
    tid = trace_id or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(trace_id=tid)
    return tid


def get_trace_id() -> str | None:
    """Get current trace ID from structlog context."""
    return structlog.contextvars.get_contextvars().get("trace_id")


def node_logger() -> Callable[[_F], _F]:
    """Decorator factory: wrap an async node function to emit node_entry/node_exit log events.

    node_entry event fields: node, trace_id.
    node_exit event fields: node, duration_ms, trace_id.

    trace_id is resolved from state["metadata"].get("trace_id") first,
    then from structlog contextvars.

    Uses functools.wraps to preserve function metadata (__name__, __doc__, __annotations__, etc.)
    so LangGraph's get_type_hints() resolves type annotations correctly.
    """

    def decorator(fn: _F) -> _F:
        node_name = fn.__name__

        @functools.wraps(fn)
        async def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            # Resolve trace_id: prefer state metadata, fall back to structlog context
            metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
            trace_id = metadata.get("trace_id") or get_trace_id()

            _log = structlog.get_logger()
            _log.info("node_entry", node=node_name, trace_id=trace_id)

            t0 = time.monotonic()
            try:
                result = await fn(state, *args, **kwargs)
            finally:
                duration_ms = int((time.monotonic() - t0) * 1000)
                _log.info("node_exit", node=node_name, duration_ms=duration_ms, trace_id=trace_id)

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


async def run_graph_traced(
    graph,
    initial_state: dict,
    event_bus: Any = None,
) -> tuple[dict, list[dict]]:
    """Run a LangGraph graph with per-node tracing.

    Uses ``graph.astream(stream_mode="updates")`` to capture the output
    of every node as it completes.

    Args:
        event_bus: Optional EventBus for publishing structured SSE events.

    Returns:
        (final_state, node_trace) where node_trace is a list of dicts:
        [{"node": str, "output": dict, "duration_ms": int}, ...]
    """
    node_trace: list[dict] = []
    final_state: dict[str, Any] = {}

    import uuid

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Share the live trace via the registry so journal nodes (which run *inside*
    # the graph) can read what's accumulated so far. Keyed by state.metadata.trace_id.
    trace_id = (initial_state.get("metadata") or {}).get("trace_id")
    trace_register(trace_id) if trace_id else None
    last_chunk_t = time.monotonic()
    try:
        async for chunk in graph.astream(initial_state, config=config, stream_mode="updates"):
            now = time.monotonic()
            duration_ms = int((now - last_chunk_t) * 1000)
            last_chunk_t = now
            for node_name, update in chunk.items():
                entry = _build_trace_entry(node_name, update)
                entry["duration_ms"] = duration_ms
                node_trace.append(entry)
                trace_append(trace_id, entry)
                logger.debug("Node %s completed in %dms", node_name, duration_ms)
                if event_bus is not None:
                    await event_bus.publish(
                        "node_done",
                        {
                            "node_name": node_name,
                            "duration_ms": duration_ms,
                        },
                    )

        # Build final state from initial + all updates
        final_state = dict(initial_state)
        for entry in node_trace:
            _merge_update(final_state, entry["output"])

        return final_state, node_trace
    finally:
        trace_unregister(trace_id)


def _build_trace_entry(node_name: str, update: Any) -> dict:
    """Build a trace entry from a node's output."""
    # Extract meaningful data from the state update
    output_summary = _summarize_node_output(node_name, update)
    return {
        "node": node_name,
        "output": update if isinstance(update, dict) else {},
        "summary": output_summary,
        "duration_ms": 0,  # astream doesn't provide timing; we add it below
        "ts": time.time(),
    }


_SUMMARY_HANDLERS: list[tuple[str, Any]] = []  # populated below


def _summarize_node_output(node_name: str, update: Any) -> str:
    """Create a human-readable summary of a node's output."""
    if not isinstance(update, dict):
        return str(update)[:200]
    data = update.get("data", {})
    for key, handler in _SUMMARY_HANDLERS:
        if data.get(key):
            return handler(data[key])
    # Check debate gate (flag-style key)
    if "debate_skipped" in data:
        reason = data.get("debate_skip_reason", "")
        return f"{'SKIPPED' if data['debate_skipped'] else 'DEBATE'}: {reason}"
    keys = list(data.keys())[:5]
    return f"keys={keys}" if keys else "(empty)"


def _summarize_analyses(analyses: dict) -> str:
    parts = []
    for aid, a in analyses.items():
        if isinstance(a, dict):
            parts.append(f"{aid}: {a.get('direction', '?')} {a.get('confidence', 0):.0%}")
    return " | ".join(parts)


def _summarize_verdict(verdict: dict) -> str:
    action = verdict.get("action", "?")
    conf = verdict.get("confidence", 0)
    scale = verdict.get("position_scale", 0)
    thesis = verdict.get("thesis", "")[:100]
    return f"{action} conf={conf:.0%} scale={scale:.0%} | {thesis}"


def _summarize_risk_gate(rg: dict) -> str:
    if rg.get("passed", True):
        return "PASSED"
    return f"REJECTED: {rg.get('rejected_by', '')} — {rg.get('reason', '')}"


def _summarize_snapshot(s: dict) -> str:
    return f"price=${s.get('price', 0):,.0f} vol={s.get('volatility', 0):.4f}"


def _summarize_regime(tags: list) -> str:
    return f"regime={tags}"


_SUMMARY_HANDLERS = [
    ("analyses", _summarize_analyses),
    ("verdict", _summarize_verdict),
    ("risk_gate", _summarize_risk_gate),
    ("snapshot_summary", _summarize_snapshot),
    ("regime_tags", _summarize_regime),
]


def _merge_update(state: dict, update: dict) -> None:
    """Merge a node's state update into the accumulated state (in-place)."""
    for k, v in update.items():
        if k in state and isinstance(state[k], dict) and isinstance(v, dict):
            _merge_update(state[k], v)
        else:
            state[k] = v


def add_timing_to_trace(trace: list[dict]) -> None:
    """Post-process trace to compute per-node durations from timestamps."""
    for i, entry in enumerate(trace):
        if i == 0:
            entry["duration_ms"] = 0
        else:
            entry["duration_ms"] = int((entry["ts"] - trace[i - 1]["ts"]) * 1000)
