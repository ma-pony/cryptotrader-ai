"""TradingCycle chat cancellation and workflow watch endpoints."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/chat")


class InterruptResponse(BaseModel):
    type: str
    session_id: str


@router.post("/interrupt/{session_id}")
async def interrupt_analysis(session_id: str) -> InterruptResponse:
    from cryptotrader.chat.task_manager import BackgroundTaskManager

    manager = BackgroundTaskManager.get_instance()
    task = manager.get(session_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if task.completed or task.interrupt_event.is_set():
        return InterruptResponse(type="interrupt_noop", session_id=session_id)
    interrupted_task = manager.interrupt(session_id)
    if interrupted_task is None:
        return InterruptResponse(type="interrupt_noop", session_id=session_id)
    with suppress(asyncio.CancelledError):
        await interrupted_task.task
    return InterruptResponse(type="interrupt_received", session_id=session_id)


@router.get("/watch")
async def watch_workflows(request: Request):
    state = _get_state_manager(request)

    async def generate():
        async for message in state.subscribe_iter("analysis:new_workflow"):
            yield f"event: new_workflow\ndata: {message}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _get_state_manager(request: Request):
    from cryptotrader.risk.state import RedisStateManager

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Trading runtime is not initialized")
    return RedisStateManager(runtime.snapshot.document.infrastructure.redis_url or None)
