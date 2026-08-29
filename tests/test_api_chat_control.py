"""Tests for chat control API endpoints — interrupt and watch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from cryptotrader.chat.event_buffer import EventBuffer
from cryptotrader.chat.event_bus import EventBus
from cryptotrader.chat.task_manager import BackgroundTaskManager
from cryptotrader.config import ChatConfig
from cryptotrader.risk.state import RedisStateManager


@pytest.fixture(autouse=True)
def _reset_singleton():
    BackgroundTaskManager.reset()
    yield
    BackgroundTaskManager.reset()


@pytest.fixture
def state_mgr():
    return RedisStateManager(None)


def _make_bus(session_id: str, state_mgr: RedisStateManager) -> EventBus:
    buf = EventBuffer(session_id, state_mgr)
    return EventBus(session_id, buf)


async def _long_coro(_interrupt_event):
    await asyncio.sleep(10)


class _BlockingCancellationCycle:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.entered = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.cleanup_release = asyncio.Event()

    async def run(self, _cycle_request):
        self.entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cleanup_started.set()
            await self.cleanup_release.wait()
            await self.bus.publish("cycle_cancelled", {"run": "old", "status": "cancelled"})
            raise


@pytest.mark.asyncio
async def test_interrupt_returns_received(state_mgr):
    from api.routes.chat_control import interrupt_analysis

    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus = _make_bus("s1", state_mgr)
    mgr.create("s1", "BTC/USDT", _long_coro, "chat", bus)

    resp = await interrupt_analysis("s1")
    assert resp.type == "interrupt_received"
    assert resp.session_id == "s1"


@pytest.mark.asyncio
async def test_interrupt_noop_when_already_interrupted(state_mgr):
    from api.routes.chat_control import interrupt_analysis

    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus = _make_bus("s1", state_mgr)
    mgr.create("s1", "BTC/USDT", _long_coro, "chat", bus)

    await interrupt_analysis("s1")
    resp = await interrupt_analysis("s1")
    assert resp.type == "interrupt_noop"


@pytest.mark.asyncio
async def test_interrupt_404_for_unknown():
    from fastapi import HTTPException

    from api.routes.chat_control import interrupt_analysis

    config = ChatConfig(max_concurrent_tasks=5)
    BackgroundTaskManager.get_instance(config)
    with pytest.raises(HTTPException) as exc_info:
        await interrupt_analysis("nonexistent")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_interrupt_response_waits_for_exact_old_task_cleanup_before_same_session_reuse(state_mgr):
    from api.routes.chat_control import interrupt_analysis
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer

    config = ChatConfig(max_concurrent_tasks=5)
    manager = BackgroundTaskManager.get_instance(config)
    shared_buffer = EventBuffer("s1", state_mgr)
    old_bus = EventBus("s1", shared_buffer)
    old_cycle = _BlockingCancellationCycle(old_bus)
    from cryptotrader.cycle_events import MultiplexedCycleEventSink, NullCycleEventSink

    old_runtime = SimpleNamespace(
        cycle=old_cycle,
        events=MultiplexedCycleEventSink(NullCycleEventSink()),
    )

    async def run_old(interrupt_event):
        await run_analysis_and_buffer(
            pair="BTC/USDT:USDT",
            session_id="s1",
            event_bus=old_bus,
            interrupt_event=interrupt_event,
            state_mgr=state_mgr,
            runtime=old_runtime,
        )

    old_analysis = manager.create("s1", "BTC/USDT:USDT", run_old, "chat", old_bus)
    await old_cycle.entered.wait()

    interrupt_response = asyncio.create_task(interrupt_analysis("s1"))
    await old_cycle.cleanup_started.wait()
    await asyncio.sleep(0)

    assert not interrupt_response.done()
    assert await state_mgr.get("analysis:status:s1") == "running"

    old_cycle.cleanup_release.set()
    response = await asyncio.wait_for(interrupt_response, timeout=1)

    assert response.type == "interrupt_received"
    assert old_analysis.task.done()
    assert await state_mgr.get("analysis:status:s1") == "cancelled"
    old_events = await shared_buffer.range_after(0)
    assert [event.type for event in old_events][-2:] == ["cycle_cancelled", "stream_done"]
    old_terminal_id = old_events[-1].event_id

    new_buffer = EventBuffer("s1", state_mgr)
    new_bus = EventBus("s1", new_buffer)
    new_started = asyncio.Event()
    new_session_start_id: asyncio.Future[int] = asyncio.Future()

    async def run_new(_interrupt_event):
        session_start = await new_bus.publish("session_start", {"run": "new"})
        new_session_start_id.set_result(session_start.event_id)
        await new_bus.publish("cycle_started", {"run": "new"})
        new_started.set()
        await asyncio.Future()

    new_analysis = manager.create("s1", "BTC/USDT:USDT", run_new, "chat", new_bus)
    await new_started.wait()
    await asyncio.sleep(0)

    assert manager.get("s1") is new_analysis
    assert not new_analysis.completed
    start_id = await new_session_start_id
    assert old_terminal_id < start_id
    replayed = await new_buffer.range_after(start_id)
    assert [(event.type, event.data.get("run")) for event in replayed] == [("cycle_started", "new")]

    new_analysis.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await new_analysis.task
