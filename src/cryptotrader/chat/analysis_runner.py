"""Background analysis coroutine — runs the trading graph and buffers events."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cryptotrader.chat.event_bus import EventBus
    from cryptotrader.risk.state import RedisStateManager

logger = logging.getLogger(__name__)


async def run_analysis_and_buffer(
    pair: str,
    session_id: str,
    event_bus: EventBus,
    interrupt_event: asyncio.Event,
    state_mgr: RedisStateManager,
    graph: Any,
    trigger_source: str = "chat",
) -> None:
    """Run the full trading analysis graph, publishing events to the bus.

    This coroutine is designed to outlive the HTTP connection that spawned it.
    The ``graph`` parameter must be a compiled LangGraph (obtained from
    ``build_trading_graph()``); this module does not import ``graph.py``
    directly to respect the TID251 module boundary.
    """
    from cryptotrader.chat import runtime_registry
    from cryptotrader.config import load_config
    from cryptotrader.state import build_initial_state

    # Stash live runtime objects in a session-keyed registry so they don't end
    # up in state["metadata"] (state is msgpack-serialized by LangGraph's
    # MemorySaver checkpointer and EventBus / RedisStateManager are not
    # serializable). Nodes resolve them via session_id lookup.
    runtime_registry.register(session_id, event_bus=event_bus, redis_state_manager=state_mgr)

    try:
        await event_bus.publish(
            "session_start",
            {
                "session_id": session_id,
                "pair": pair,
                "trigger_source": trigger_source,
            },
        )
        await state_mgr.set(f"analysis:status:{session_id}", "running", ex=600)

        config = load_config()
        # Use session_id as the trace_id so the journal node can pull the
        # accumulated node_trace from the registry by trace_id.
        trace_id = session_id
        initial_state = build_initial_state(
            pair,
            engine=config.engine,
            exchange_id=config.exchange_id,
            config=config,
            extra_metadata={
                # Only serializable primitives go into state.
                "session_id": session_id,
                "trace_id": trace_id,
            },
        )
        final_state: dict[str, Any] = {}

        # LangGraph MemorySaver checkpointer requires configurable.thread_id.
        # Reuse session_id so checkpoints are addressable per chat session.
        graph_config = {"configurable": {"thread_id": session_id}}

        import time as _time

        from cryptotrader.metrics import get_metrics_collector
        from cryptotrader.tracing import (
            trace_append,
            trace_register,
            trace_unregister,
        )

        pipeline_t0 = _time.monotonic()
        trace_register(trace_id)
        # Bind the per-decision token ledger in *this* coroutine context so
        # every subtask LangGraph spawns inherits it (see comment in
        # ``tracing.run_graph_traced`` for the same fix on the scheduler path).
        from cryptotrader.llm.token_tracker import start_ledger

        start_ledger()
        # Track previous chunk timestamp so each node's duration_ms reflects
        # the wall-clock gap between the prior node_exit and this node_exit
        # (LangGraph astream doesn't provide per-node timings directly).
        last_chunk_t = pipeline_t0

        try:
            async for chunk in graph.astream(initial_state, config=graph_config, stream_mode="updates"):
                if interrupt_event.is_set():
                    await _handle_interrupt(event_bus, session_id, final_state)
                    return

                from cryptotrader.tracing import _summarize_node_output

                now = _time.monotonic()
                duration_ms = int((now - last_chunk_t) * 1000)
                last_chunk_t = now
                for node_name, update in chunk.items():
                    # Record into shared trace registry so journal node can persist
                    # node_timeline + latency_breakdown to the commit.
                    # Compute human-readable summary the same way run_graph_traced
                    # (scheduler path) does — keeps both code paths producing
                    # equivalent journal data.
                    trace_append(
                        trace_id,
                        {
                            "node": node_name,
                            "duration_ms": duration_ms,
                            "ts": _time.time(),
                            "summary": _summarize_node_output(node_name, update),
                        },
                    )
                    await event_bus.publish(
                        "node_done",
                        {
                            "node_name": node_name,
                            "duration_ms": duration_ms,
                        },
                    )
                    if isinstance(update, dict):
                        final_state.update(update)
        finally:
            trace_unregister(trace_id)

        # Record pipeline duration in Prometheus histogram so /api/metrics/summary
        # p50/p95 actually populate. Same observation used by scheduler runs.
        get_metrics_collector().observe_pipeline_duration(ms=(_time.monotonic() - pipeline_t0) * 1000.0)

        await event_bus.publish(
            "stream_done",
            {
                "session_id": session_id,
            },
        )
        await state_mgr.set(f"analysis:status:{session_id}", "done", ex=600)

    except asyncio.CancelledError:
        logger.info("Analysis task cancelled: session_id=%s", session_id)
        await event_bus.publish("stream_error", {"error": "Task cancelled"})
    except Exception:
        logger.exception("Analysis task failed: session_id=%s", session_id)
        await event_bus.publish("stream_error", {"error": "Internal analysis error"})
        await state_mgr.set(f"analysis:status:{session_id}", "error", ex=600)
    finally:
        runtime_registry.unregister(session_id)


async def _handle_interrupt(
    event_bus: EventBus,
    session_id: str,
    final_state: dict[str, Any],
) -> None:
    """Handle soft interrupt — produce partial verdict if possible."""
    from cryptotrader.chat.partial_verdict import make_partial_verdict
    from cryptotrader.chat.task_manager import BackgroundTaskManager

    mgr = BackgroundTaskManager.get_instance()
    task = mgr.get(session_id)
    completed_agents = task.completed_agents if task else []

    if not completed_agents:
        await event_bus.publish(
            "interrupt_rejected",
            {
                "reason": "尚无可用 Agent 分析结果",
            },
        )
        return

    analyses = (final_state.get("data") or {}).get("analyses", {})
    available = {k: v for k, v in analyses.items() if k in completed_agents}
    partial = make_partial_verdict(available)

    await event_bus.publish("checkpoint_saved", {"session_id": session_id})
    await event_bus.publish("verdict_partial", partial)
    await event_bus.publish("stream_done", {"session_id": session_id, "interrupted": True})
