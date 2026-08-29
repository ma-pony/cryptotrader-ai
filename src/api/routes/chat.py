"""SSE chat surface backed by the shared TradingCycle."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from cryptotrader.decision.models import CycleOutcome

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
async def chat_stream(payload: ChatStreamRequest, request: Request):
    session_id = payload.session_id or str(uuid.uuid4())
    if payload.last_event_id is not None:
        return await _handle_reconnect(session_id, payload.last_event_id)
    return await _handle_new_analysis(session_id, payload, request)


async def _handle_new_analysis(
    session_id: str,
    payload: ChatStreamRequest,
    request: Request,
) -> StreamingResponse:
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer
    from cryptotrader.chat.event_buffer import EventBuffer
    from cryptotrader.chat.event_bus import EventBus
    from cryptotrader.chat.task_manager import (
        BackgroundTaskManager,
        ExecutionInProgressError,
        TooManyTasksError,
    )
    from cryptotrader.risk.state import RedisStateManager

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or runtime.cycle is None:
        raise HTTPException(status_code=503, detail="Trading runtime is not active")
    manager = BackgroundTaskManager.get_instance(workflow_publisher=None)
    existing = manager.get(session_id)
    if existing is not None and not existing.completed:
        if existing.event_bus.execution_started:
            raise HTTPException(status_code=409, detail="Analysis execution is already in progress")
        interrupted = manager.interrupt(session_id)
        pending = interrupted or existing
        with contextlib.suppress(asyncio.CancelledError):
            await pending.task
    config = runtime.snapshot.document
    pair = (
        payload.message.strip().upper()
        if "/" in payload.message
        else (config.scheduler.pairs[0] if config.scheduler.pairs else "BTC/USDT")
    )
    state = RedisStateManager(config.infrastructure.redis_url or None)
    buffer = EventBuffer(
        session_id,
        state,
        600,
        1000,
    )
    bus = EventBus(session_id, buffer)

    async def run_cycle(interrupt_event: asyncio.Event) -> CycleOutcome | None:
        return await run_analysis_and_buffer(
            pair=pair,
            session_id=session_id,
            event_bus=bus,
            interrupt_event=interrupt_event,
            state_mgr=state,
            runtime=runtime,
            trigger_source="chat",
        )

    try:
        BackgroundTaskManager.get_instance(workflow_publisher=state.publish).create(
            session_id,
            pair,
            run_cycle,
            "chat",
            bus,
        )
    except ExecutionInProgressError as error:
        raise HTTPException(status_code=409, detail="Analysis execution is already in progress") from error
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
