"""POST /api/chat/stream — SSE streaming chat endpoint (FR-600~619, FR-001~010)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from cryptotrader._compat import UTC

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat")


class AdditionalContextRequest(BaseModel):
    """Chart capture payloads for multimodal context injection."""

    payloads: list[dict] = []
    model: str = ""


class ChatStreamRequest(BaseModel):
    """Request body for POST /api/chat/stream."""

    session_id: str = ""
    message: str = ""
    model: str = ""
    additional_context: AdditionalContextRequest | None = None
    last_event_id: int | None = None


async def _sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_AGENT_KIND_MAP: dict[str, str] = {
    "tech_agent": "tech",
    "chain_agent": "chain",
    "news_agent": "news",
    "macro_agent": "macro",
}


def _agent_kind_from_name(name: str) -> str:
    """Normalize agent id to the 4-tone palette used by the frontend."""
    if name in _AGENT_KIND_MAP:
        return _AGENT_KIND_MAP[name]
    low = name.lower()
    if "tech" in low or "indicator" in low:
        return "tech"
    if "chain" in low or "whale" in low or "onchain" in low:
        return "chain"
    if "news" in low or "sentiment" in low:
        return "news"
    if "macro" in low or "fed" in low or "dxy" in low:
        return "macro"
    return "other"


async def _run_chat_stream_legacy(req: ChatStreamRequest):
    """Legacy generator — runs the trading graph inline and streams SSE events.

    Kept for backward compatibility (FR-023). Used when the chat subsystem
    infrastructure is not available.
    """
    session_id = req.session_id or str(uuid.uuid4())
    msg_id = str(uuid.uuid4())
    ts = datetime.now(UTC).isoformat()

    yield await _sse_frame("session", {"session_id": session_id})
    yield await _sse_frame("message_start", {"id": msg_id, "role": "assistant", "ts": ts})

    try:
        from cryptotrader.config import load_config
        from cryptotrader.graph import build_trading_graph
        from cryptotrader.state import build_initial_state
        from cryptotrader.tracing import run_graph_traced

        config = load_config()

        if req.additional_context and req.additional_context.payloads:
            from api.context_builder import build_multimodal_messages

            ctx_model = req.additional_context.model or req.model or config.models.analysis
            _ctx_msgs, degraded = build_multimodal_messages(
                req.additional_context.payloads,
                ctx_model,
                config,
            )
            if degraded:
                yield await _sse_frame("context_notice", {"type": "image_too_large"})

        pair = (
            req.message.strip().upper()
            if "/" in req.message
            else (config.scheduler.pairs[0].canonical() if config.scheduler.pairs else "BTC/USDT")
        )

        graph = build_trading_graph()
        state = build_initial_state(pair, engine="paper", exchange_id=config.exchange_id, config=config)
        yield await _sse_frame("content_delta", {"id": msg_id, "delta": f"Analyzing {pair}...\n\n"})

        result, node_trace = await run_graph_traced(graph, state)

        for entry in node_trace:
            yield await _sse_frame(
                "tool_call",
                {
                    "id": str(uuid.uuid4()),
                    "name": entry.get("node", ""),
                    "args": {"duration_ms": entry.get("duration_ms", 0)},
                },
            )
            await asyncio.sleep(0)

        verdict = result.get("data", {}).get("verdict", {})
        analyses = result.get("data", {}).get("analyses", {})

        # Structured agent_message events so the frontend can render each analyst
        # as a distinct bubble (matches prototype design: per-agent color + direction + conf).
        for agent_name, analysis in (analyses or {}).items():
            direction = analysis.get("direction", "neutral")
            conf = float(analysis.get("confidence", 0.0) or 0.0)
            reasoning = analysis.get("reasoning", "") or ""
            kind = _agent_kind_from_name(agent_name)
            yield await _sse_frame(
                "agent_message",
                {
                    "id": str(uuid.uuid4()),
                    "agent": kind,
                    "agent_name": agent_name,
                    "direction": direction,
                    "confidence": conf,
                    "reasoning": reasoning,
                },
            )
            await asyncio.sleep(0)

        yield await _sse_frame(
            "verdict",
            {
                "id": str(uuid.uuid4()),
                "action": verdict.get("action", "hold"),
                "confidence": float(verdict.get("confidence", 0.0) or 0.0),
                "position_scale": float(verdict.get("position_scale", 0.0) or 0.0),
                "reasoning": verdict.get("reasoning", "") or "",
            },
        )

        # Legacy markdown summary — kept for backwards compatibility; the frontend
        # that knows about agent_message can ignore the redundant text.
        summary_lines = [
            f"**{pair} Analysis Complete**\n\n",
            f"- Direction: **{verdict.get('action', 'hold')}**\n",
            f"- Confidence: {verdict.get('confidence', 0):.0%}\n",
            f"- Position Scale: {verdict.get('position_scale', 0):.1%}\n\n",
        ]
        if analyses:
            summary_lines.append("**Agent Analyses:**\n\n")
            for agent_name, analysis in analyses.items():
                direction = analysis.get("direction", "hold")
                conf = analysis.get("confidence", 0)
                summary_lines.append(f"- {agent_name}: {direction} ({conf:.0%})\n")

        yield await _sse_frame("content_delta", {"id": msg_id, "delta": "".join(summary_lines)})

    except Exception as exc:
        logger.exception("Chat stream error")
        error_detail = _classify_stream_error(exc)
        yield await _sse_frame("stream_error", error_detail)
        yield await _sse_frame(
            "content_delta",
            {"id": msg_id, "delta": f"\n\n**Error**: {error_detail['message']}\n"},
        )

    yield await _sse_frame("message_end", {"id": msg_id})
    yield await _sse_frame("done", {"session_id": session_id})


async def _sse_consumer_gen(session_id: str, last_event_id: int | None = None):
    """Consume events from an EventBus and yield SSE frames.

    New analysis (last_event_id=None): subscribe and stream live events.
    Reconnect (last_event_id set): subscribe first, replay missed events,
    then seamlessly continue with live events (deduplicating overlap).
    """
    from cryptotrader.chat.event_bus import SSEEnvelope
    from cryptotrader.chat.task_manager import BackgroundTaskManager

    mgr = BackgroundTaskManager.get_instance()
    task = mgr.get(session_id)
    if task is None:
        yield await _sse_frame("stream_error", {"error": "Session not found"})
        return

    bus = task.event_bus
    q = bus.subscribe()
    try:
        if last_event_id is not None:
            yield await _sse_frame(
                "stream_resume",
                {
                    "session_id": session_id,
                    "last_event_id": last_event_id,
                },
            )

            buffer = bus._buffer
            replayed = await buffer.range_after(last_event_id)
            max_replayed_id = last_event_id
            for env in replayed:
                yield SSEEnvelope.to_sse_frame(env)
                max_replayed_id = max(max_replayed_id, env.event_id)
                if env.type in ("stream_done", "stream_error"):
                    return

            if task.completed:
                return

            last_event_id = max_replayed_id

        while True:
            try:
                envelope = await asyncio.wait_for(q.get(), timeout=30.0)
            except TimeoutError:
                # Py 3.10: asyncio.TimeoutError is NOT builtins.TimeoutError —
                # use the asyncio variant explicitly so keepalives actually fire.
                yield ": keepalive\n\n"
                continue

            if last_event_id is not None and envelope.event_id <= last_event_id:
                continue

            yield SSEEnvelope.to_sse_frame(envelope)

            if envelope.type in ("stream_done", "stream_error"):
                return
    finally:
        bus.unsubscribe(q)


@router.post("/stream")
async def chat_stream(req: ChatStreamRequest):
    """Stream a multi-agent analysis as SSE events.

    New analysis: spawns a background task and consumes events via EventBus.
    Reconnect (last_event_id set): replays missed events then attaches to live stream.
    Falls back to legacy inline mode if chat subsystem is unavailable.
    """
    session_id = req.session_id or str(uuid.uuid4())

    if req.last_event_id is not None:
        return await _handle_reconnect(session_id, req.last_event_id)

    try:
        return await _handle_new_analysis(session_id, req)
    except Exception:
        logger.debug("Background task mode unavailable, falling back to legacy", exc_info=True)
        return StreamingResponse(
            _run_chat_stream_legacy(req),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

_ERROR_MESSAGES: dict[str, str] = {
    "rate_limit": "Rate limited by LLM provider. Retrying automatically.",
    "auth_error": "Authentication failed. Check API key configuration.",
    "timeout": "Analysis timed out. Try again.",
    "bad_request": "Invalid request to LLM provider.",
    "server_error": "LLM provider server error. Try again later.",
    "connection_error": "Cannot reach LLM provider. Check network.",
    "providers_exhausted": "All LLM providers failed. Check configuration.",
    "unknown": "Analysis failed. Check server logs for details.",
}


def _classify_stream_error(exc: Exception) -> dict:
    """Classify an exception into a structured SSE error event payload."""
    from cryptotrader.llm.errors import classify_error

    category, retryable = classify_error(exc)
    return {
        "category": category,
        "retryable": retryable,
        "message": _ERROR_MESSAGES.get(category, _ERROR_MESSAGES["unknown"]),
    }


async def _handle_new_analysis(session_id: str, req: ChatStreamRequest) -> StreamingResponse:
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer
    from cryptotrader.chat.event_buffer import EventBuffer
    from cryptotrader.chat.event_bus import EventBus
    from cryptotrader.chat.task_manager import BackgroundTaskManager, TooManyTasksError
    from cryptotrader.config import load_config
    from cryptotrader.graph import build_trading_graph
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    pair = (
        req.message.strip().upper()
        if "/" in req.message
        else (config.scheduler.pairs[0].canonical() if config.scheduler.pairs else "BTC/USDT")
    )

    state_mgr = RedisStateManager(config.infrastructure.redis_url or None)
    buffer = EventBuffer(session_id, state_mgr, config.chat.event_buffer_ttl_seconds, config.chat.event_buffer_max_size)
    bus = EventBus(session_id, buffer)
    graph = build_trading_graph()

    coro = run_analysis_and_buffer(
        pair=pair,
        session_id=session_id,
        event_bus=bus,
        interrupt_event=asyncio.Event(),
        state_mgr=state_mgr,
        graph=graph,
        trigger_source="chat",
    )

    mgr = BackgroundTaskManager.get_instance(config.chat)
    try:
        mgr.create(session_id, pair, coro, "chat", bus)
    except TooManyTasksError as exc:
        raise HTTPException(status_code=429, detail="Too many concurrent analyses") from exc

    return StreamingResponse(
        _sse_consumer_gen(session_id),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


async def _handle_reconnect(session_id: str, last_event_id: int) -> StreamingResponse | JSONResponse:
    from cryptotrader.chat.task_manager import BackgroundTaskManager

    mgr = BackgroundTaskManager.get_instance()
    task = mgr.get(session_id)

    if task is None:
        raise HTTPException(status_code=410, detail="Session expired or not found")

    buffer = task.event_bus._buffer
    has_data = await buffer.exists()
    if not has_data and task.completed:
        raise HTTPException(status_code=410, detail="Session buffer expired")

    return StreamingResponse(
        _sse_consumer_gen(session_id, last_event_id),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
