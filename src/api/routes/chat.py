"""SSE chat surface backed by the shared TradingCycle."""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/chat")


class AdditionalContextRequest(BaseModel):
    payloads: list[dict] = []
    model: str = ""


class ChatStreamRequest(BaseModel):
    session_id: str = ""
    message: str = ""
    model: str = ""
    additional_context: AdditionalContextRequest | None = None
    last_event_id: int | None = None


async def _sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def _sse_consumer_gen(session_id: str, last_event_id: int | None = None):
    from cryptotrader.chat.event_bus import SSEEnvelope
    from cryptotrader.chat.task_manager import BackgroundTaskManager

    task = BackgroundTaskManager.get_instance().get(session_id)
    if task is None:
        yield await _sse_frame("stream_error", {"error": "Session not found"})
        return

    bus = task.event_bus
    queue = bus.subscribe()
    try:
        if last_event_id is not None:
            yield await _sse_frame(
                "stream_resume",
                {"session_id": session_id, "last_event_id": last_event_id},
            )
            replayed = await bus._buffer.range_after(last_event_id)
            for envelope in replayed:
                yield SSEEnvelope.to_sse_frame(envelope)
                last_event_id = max(last_event_id, envelope.event_id)
                if envelope.type in ("stream_done", "stream_error"):
                    return
            if task.completed:
                return

        while True:
            try:
                envelope = await asyncio.wait_for(queue.get(), timeout=30.0)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            if last_event_id is not None and envelope.event_id <= last_event_id:
                continue
            yield SSEEnvelope.to_sse_frame(envelope)
            if envelope.type in ("stream_done", "stream_error"):
                return
    finally:
        bus.unsubscribe(queue)


@router.post("/stream")
async def chat_stream(request: ChatStreamRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if request.last_event_id is not None:
        return await _handle_reconnect(session_id, request.last_event_id)
    return await _handle_new_analysis(session_id, request)


async def _handle_new_analysis(session_id: str, request: ChatStreamRequest) -> StreamingResponse:
    from cryptotrader.bootstrap import build_trading_cycle
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer
    from cryptotrader.chat.event_buffer import EventBuffer
    from cryptotrader.chat.event_bus import EventBus, EventBusCycleSink
    from cryptotrader.chat.task_manager import BackgroundTaskManager, TooManyTasksError
    from cryptotrader.config import load_config
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    pair = (
        request.message.strip().upper()
        if "/" in request.message
        else (config.scheduler.pairs[0].canonical() if config.scheduler.pairs else "BTC/USDT")
    )
    state = RedisStateManager(config.infrastructure.redis_url or None)
    buffer = EventBuffer(
        session_id,
        state,
        config.chat.event_buffer_ttl_seconds,
        config.chat.event_buffer_max_size,
    )
    bus = EventBus(session_id, buffer)
    cycle = build_trading_cycle(config, "paper", EventBusCycleSink(bus))

    async def run_cycle(interrupt_event: asyncio.Event) -> None:
        await run_analysis_and_buffer(
            pair=pair,
            session_id=session_id,
            event_bus=bus,
            interrupt_event=interrupt_event,
            state_mgr=state,
            cycle=cycle,
            trigger_source="chat",
        )

    try:
        BackgroundTaskManager.get_instance(config.chat).create(
            session_id,
            pair,
            run_cycle,
            "chat",
            bus,
        )
    except TooManyTasksError as error:
        raise HTTPException(status_code=429, detail="Too many concurrent analyses") from error
    return StreamingResponse(
        _sse_consumer_gen(session_id),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


async def _handle_reconnect(session_id: str, last_event_id: int) -> StreamingResponse | JSONResponse:
    from cryptotrader.chat.task_manager import BackgroundTaskManager

    task = BackgroundTaskManager.get_instance().get(session_id)
    if task is None:
        raise HTTPException(status_code=410, detail="Session expired or not found")
    if not await task.event_bus._buffer.exists() and task.completed:
        raise HTTPException(status_code=410, detail="Session buffer expired")
    return StreamingResponse(
        _sse_consumer_gen(session_id, last_event_id),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
