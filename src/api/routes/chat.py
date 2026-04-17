"""POST /api/chat/stream — SSE streaming chat endpoint (FR-600~619)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat")


class ChatStreamRequest(BaseModel):
    """Request body for POST /api/chat/stream."""

    session_id: str = ""
    message: str = ""
    model: str = ""


async def _sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _run_chat_stream(req: ChatStreamRequest):
    """Generator that runs the trading graph and streams SSE events."""
    session_id = req.session_id or str(uuid.uuid4())
    msg_id = str(uuid.uuid4())
    ts = datetime.now(UTC).isoformat()

    yield await _sse_frame("session", {"session_id": session_id})

    yield await _sse_frame(
        "message_start",
        {
            "id": msg_id,
            "role": "assistant",
            "ts": ts,
        },
    )

    try:
        from cryptotrader.config import load_config
        from cryptotrader.graph import build_trading_graph
        from cryptotrader.state import build_initial_state
        from cryptotrader.tracing import run_graph_traced

        config = load_config()
        pair = (
            req.message.strip().upper()
            if "/" in req.message
            else (config.scheduler.pairs[0] if config.scheduler.pairs else "BTC/USDT")
        )

        graph = build_trading_graph()
        state = build_initial_state(
            pair,
            engine="paper",
            exchange_id=config.exchange_id,
            config=config,
        )

        yield await _sse_frame(
            "content_delta",
            {
                "id": msg_id,
                "delta": f"Analyzing {pair}...\n\n",
            },
        )

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

        summary = "".join(summary_lines)
        yield await _sse_frame("content_delta", {"id": msg_id, "delta": summary})

    except Exception:
        logger.exception("Chat stream error")
        yield await _sse_frame(
            "content_delta",
            {
                "id": msg_id,
                "delta": "\n\n**Error**: Analysis failed. Check server logs for details.\n",
            },
        )

    yield await _sse_frame("message_end", {"id": msg_id})
    yield await _sse_frame("done", {"session_id": session_id})


@router.post("/stream")
async def chat_stream(req: ChatStreamRequest):
    """Stream a multi-agent analysis as SSE events."""
    return StreamingResponse(
        _run_chat_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
