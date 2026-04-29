"""Chat control API — interrupt, steer, watch endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
# Auth applied at registration site in api/main.py to keep protection visible.
router = APIRouter(prefix="/api/chat")


class InterruptResponse(BaseModel):
    type: str
    session_id: str


class SteerRequest(BaseModel):
    target: str
    instruction: str


class SteerResponse(BaseModel):
    type: str
    target: str
    queue_position: int = 0


VALID_AGENTS = {"tech_agent", "chain_agent", "news_agent", "macro_agent"}


@router.post("/interrupt/{session_id}")
async def interrupt_analysis(session_id: str) -> InterruptResponse:
    from cryptotrader.chat.task_manager import BackgroundTaskManager

    mgr = BackgroundTaskManager.get_instance()
    task = mgr.get(session_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if task.completed or task.interrupt_event.is_set():
        return InterruptResponse(type="interrupt_noop", session_id=session_id)

    mgr.interrupt(session_id)
    return InterruptResponse(type="interrupt_received", session_id=session_id)


@router.post("/steer/{session_id}")
async def steer_agent(session_id: str, req: SteerRequest) -> SteerResponse:
    from cryptotrader.chat.task_manager import BackgroundTaskManager
    from cryptotrader.config import load_config

    if req.target not in VALID_AGENTS:
        raise HTTPException(status_code=422, detail=f"Invalid agent: {req.target}")

    mgr = BackgroundTaskManager.get_instance()
    task = mgr.get(session_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if req.target in task.completed_agents:
        return SteerResponse(type="steer_too_late", target=req.target)

    config = load_config()
    instruction = req.instruction
    if len(instruction) > config.chat.max_steering_instruction_chars:
        instruction = instruction[: config.chat.max_steering_instruction_chars]
        await task.event_bus.publish(
            "steer_truncated",
            {
                "target": req.target,
                "original_length": len(req.instruction),
            },
        )

    steer_key = f"steering:{session_id}:{req.target}"
    state_mgr = _get_state_manager()
    await state_mgr.buffer_push(steer_key, instruction, max_size=10, ttl=300)
    queue_len = await state_mgr.buffer_len(steer_key)

    await task.event_bus.publish(
        "steer_queued",
        {
            "target": req.target,
            "queue_position": queue_len,
        },
    )

    return SteerResponse(type="steer_queued", target=req.target, queue_position=queue_len)


@router.get("/watch")
async def watch_workflows():
    state_mgr = _get_state_manager()

    async def _generate():
        async for msg in state_mgr.subscribe_iter("analysis:new_workflow"):
            envelope_str = f"event: new_workflow\ndata: {msg}\n\n"
            yield envelope_str

    return StreamingResponse(
        _generate(),
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

    config = load_config()
    return RedisStateManager(config.infrastructure.redis_url or None)
