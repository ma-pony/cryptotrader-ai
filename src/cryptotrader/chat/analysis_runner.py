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
    from cryptotrader.runtime import Runtime

logger = logging.getLogger(__name__)


async def run_analysis_and_buffer(
    pair: str,
    session_id: str,
    event_bus: EventBus,
    interrupt_event: asyncio.Event,
    state_mgr: RedisStateManager,
    runtime: Runtime,
    trigger_source: str = "chat",
) -> None:
    def published_count(event_type: str) -> int:
        counter = getattr(event_bus, "published_count", None)
        if callable(counter):
            return int(counter(event_type))
        events = getattr(event_bus, "events", ())
        return sum(1 for name, _data in events if name == event_type)

    cancelled_count_at_start = published_count("cycle_cancelled")
    done_count_at_start = published_count("stream_done")

    async def finish_cancelled() -> None:
        done_already_published = published_count("stream_done") > done_count_at_start
        if not done_already_published and published_count("cycle_cancelled") == cancelled_count_at_start:
            await event_bus.publish(
                "cycle_cancelled",
                {"session_id": session_id, "status": "cancelled"},
            )
        if not done_already_published:
            await event_bus.publish("stream_done", {"session_id": session_id, "interrupted": True})
        await state_mgr.set(f"analysis:status:{session_id}", "cancelled", ex=600)

    try:
        await event_bus.publish(
            "session_start",
            {"session_id": session_id, "pair": pair, "trigger_source": trigger_source},
        )
        if interrupt_event.is_set():
            await finish_cancelled()
            return

        await state_mgr.set(f"analysis:status:{session_id}", "running", ex=600)
        from cryptotrader.chat.event_bus import EventBusCycleSink

        if runtime.cycle is None:
            raise RuntimeError("Trading runtime is not active")
        with runtime.events.route(EventBusCycleSink(event_bus)):
            outcome = await runtime.cycle.run(CycleRequest(Pair.parse(pair)))
        await event_bus.publish(
            "stream_done",
            {"session_id": session_id, "cycle_id": outcome.cycle_id, "status": outcome.status},
        )
        await state_mgr.set(f"analysis:status:{session_id}", "done", ex=600)
    except asyncio.CancelledError:
        logger.info("Trading cycle cancelled: session_id=%s", session_id)
        await finish_cancelled()
        raise
    except Exception:
        logger.exception("Trading cycle failed: session_id=%s", session_id)
        await event_bus.publish("stream_error", {"error": "Internal analysis error"})
        await state_mgr.set(f"analysis:status:{session_id}", "error", ex=600)
