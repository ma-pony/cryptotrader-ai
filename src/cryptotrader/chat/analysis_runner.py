"""Run one TradingCycle in the background and buffer its business events."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cryptotrader.decision.models import CycleRequest
from cryptotrader.pair import Pair

if TYPE_CHECKING:
    from cryptotrader.chat.event_bus import EventBus
    from cryptotrader.decision.models import CycleOutcome
    from cryptotrader.risk.state import RedisStateManager
    from cryptotrader.runtime import Runtime

logger = logging.getLogger(__name__)


def _published_count(event_bus: EventBus, event_type: str) -> int:
    counter = getattr(event_bus, "published_count", None)
    if callable(counter):
        return int(counter(event_type))
    events = getattr(event_bus, "events", ())
    return sum(1 for name, _data in events if name == event_type)


async def _safe_publish(event_bus: EventBus, event_type: str, data: dict, *, log_message: str) -> None:
    try:
        await event_bus.publish(event_type, data)
    except Exception:
        logger.warning(log_message)


async def _safe_status(state_mgr: RedisStateManager, session_id: str, status: str) -> None:
    try:
        await state_mgr.set(f"analysis:status:{session_id}", status, ex=600)
    except Exception:
        logger.warning("Failed to publish terminal analysis status: session_id=%s", session_id)


async def _finish_cancelled(
    event_bus: EventBus,
    state_mgr: RedisStateManager,
    session_id: str,
    cancelled_count_at_start: int,
    done_count_at_start: int,
) -> None:
    done_already_published = _published_count(event_bus, "stream_done") > done_count_at_start
    if not done_already_published and _published_count(event_bus, "cycle_cancelled") == cancelled_count_at_start:
        await _safe_publish(
            event_bus,
            "cycle_cancelled",
            {"session_id": session_id, "status": "cancelled"},
            log_message=f"Failed to publish analysis cancellation: session_id={session_id}",
        )
    if not done_already_published:
        await _safe_publish(
            event_bus,
            "stream_done",
            {"session_id": session_id, "interrupted": True},
            log_message=f"Failed to publish cancelled stream state: session_id={session_id}",
        )
    await _safe_status(state_mgr, session_id, "cancelled")


async def _publish_outcome(event_bus: EventBus, session_id: str, outcome: CycleOutcome) -> None:
    for book in outcome.books:
        execution = book.execution
        try:
            await event_bus.publish(
                "book_result",
                {
                    "book_id": book.book_id,
                    "capital_scope": book.capital_scope,
                    "status": book.status,
                    "execution_status": execution.status if execution is not None else "not_started",
                    "requires_attention": execution.requires_attention if execution is not None else False,
                },
            )
        except Exception:
            logger.warning("Failed to publish terminal book result: session_id=%s", session_id)
    try:
        await event_bus.publish(
            "stream_done",
            {
                "session_id": session_id,
                "cycle_id": outcome.cycle_id,
                "config_revision": outcome.config_revision,
                "status": outcome.status,
                "execution_status": outcome.execution_status,
                "requires_attention": outcome.requires_attention,
            },
        )
    except Exception:
        logger.warning("Failed to publish terminal stream state: session_id=%s", session_id)


async def run_analysis_and_buffer(
    pair: str,
    session_id: str,
    event_bus: EventBus,
    interrupt_event: asyncio.Event,
    state_mgr: RedisStateManager,
    runtime: Runtime,
    trigger_source: str = "chat",
) -> CycleOutcome | None:
    cancelled_count_at_start = _published_count(event_bus, "cycle_cancelled")
    done_count_at_start = _published_count(event_bus, "stream_done")

    try:
        await event_bus.publish(
            "session_start",
            {"session_id": session_id, "pair": pair, "trigger_source": trigger_source},
        )
        if interrupt_event.is_set():
            await _finish_cancelled(event_bus, state_mgr, session_id, cancelled_count_at_start, done_count_at_start)
            return None

        await state_mgr.set(f"analysis:status:{session_id}", "running", ex=600)
        from cryptotrader.chat.event_bus import EventBusCycleSink

        async with runtime.execution_lease(pair) as cycle:
            with runtime.events.route(EventBusCycleSink(event_bus)):
                outcome = await cycle.run(CycleRequest(Pair.parse(pair)))
    except asyncio.CancelledError:
        logger.info("Trading cycle cancelled: session_id=%s", session_id)
        await _finish_cancelled(event_bus, state_mgr, session_id, cancelled_count_at_start, done_count_at_start)
        raise
    except Exception:
        logger.error("Trading cycle failed: session_id=%s", session_id)
        await _safe_publish(
            event_bus,
            "stream_error",
            {"error": "Internal analysis error"},
            log_message=f"Failed to publish analysis error: session_id={session_id}",
        )
        await _safe_status(state_mgr, session_id, "error")
        return None

    await _publish_outcome(event_bus, session_id, outcome)
    await _safe_status(state_mgr, session_id, "done")
    return outcome
