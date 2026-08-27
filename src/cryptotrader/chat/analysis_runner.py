"""Run one TradingCycle in the background and buffer its business events."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cryptotrader.decision.models import CycleRequest
from cryptotrader.pair import Pair

if TYPE_CHECKING:
    from cryptotrader.chat.event_bus import EventBus
    from cryptotrader.risk.state import RedisStateManager
    from cryptotrader.trading_cycle import TradingCycle

logger = logging.getLogger(__name__)


async def run_analysis_and_buffer(
    pair: str,
    session_id: str,
    event_bus: EventBus,
    interrupt_event: asyncio.Event,
    state_mgr: RedisStateManager,
    cycle: TradingCycle,
    trigger_source: str = "chat",
) -> None:
    await event_bus.publish(
        "session_start",
        {"session_id": session_id, "pair": pair, "trigger_source": trigger_source},
    )
    if interrupt_event.is_set():
        await event_bus.publish("cycle_cancelled", {"session_id": session_id, "status": "cancelled"})
        await event_bus.publish("stream_done", {"session_id": session_id, "interrupted": True})
        await state_mgr.set(f"analysis:status:{session_id}", "cancelled", ex=600)
        return

    await state_mgr.set(f"analysis:status:{session_id}", "running", ex=600)
    try:
        outcome = await cycle.run(CycleRequest(Pair.parse(pair), "paper"))
        await event_bus.publish(
            "stream_done",
            {"session_id": session_id, "cycle_id": outcome.cycle_id, "status": outcome.status},
        )
        await state_mgr.set(f"analysis:status:{session_id}", "done", ex=600)
    except asyncio.CancelledError:
        logger.info("Trading cycle cancelled: session_id=%s", session_id)
        await event_bus.publish("stream_done", {"session_id": session_id, "interrupted": True})
        await state_mgr.set(f"analysis:status:{session_id}", "cancelled", ex=600)
        raise
    except Exception:
        logger.exception("Trading cycle failed: session_id=%s", session_id)
        await event_bus.publish("stream_error", {"error": "Internal analysis error"})
        await state_mgr.set(f"analysis:status:{session_id}", "error", ex=600)
