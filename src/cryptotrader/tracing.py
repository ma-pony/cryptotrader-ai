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


async def run_graph_traced(graph, initial_state: dict) -> tuple[dict, list[dict]]:
    """Run a LangGraph graph with per-node tracing.

    Uses ``graph.astream(stream_mode="updates")`` to capture the output
    of every node as it completes.

    Returns:
        (final_state, node_trace) where node_trace is a list of dicts:
        [{"node": str, "output": dict, "duration_ms": int}, ...]
    """
    node_trace: list[dict] = []
    final_state: dict[str, Any] = {}

    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        # chunk is a dict: {node_name: state_update}
        for node_name, update in chunk.items():
            entry = _build_trace_entry(node_name, update)
            node_trace.append(entry)
            logger.debug("Node %s completed in %dms", node_name, entry["duration_ms"])

    # Build final state from initial + all updates
    final_state = dict(initial_state)
    for entry in node_trace:
        _merge_update(final_state, entry["output"])

    return final_state, node_trace


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
