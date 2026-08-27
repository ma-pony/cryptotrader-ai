"""TradingCycle chat cancellation and workflow watch endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
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
    manager.interrupt(session_id)
    return InterruptResponse(type="interrupt_received", session_id=session_id)


@router.get("/watch")
async def watch_workflows():
    state = _get_state_manager()

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


def _get_state_manager():
    from cryptotrader.config import load_config
    from cryptotrader.risk.state import RedisStateManager

    return RedisStateManager(load_config().infrastructure.redis_url or None)
