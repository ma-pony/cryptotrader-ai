"""Chat 只转发 CycleEvent。取消不产生部分裁决。"""

from __future__ import annotations

import asyncio

import pytest

from cryptotrader.cycle_events import CycleEvent
from cryptotrader.decision.models import CycleOutcome


class _Bus:
    def __init__(self) -> None:
        self.events = []

    async def publish(self, name, data=None):
        self.events.append((name, data or {}))


class _State:
    def __init__(self) -> None:
        self.values = {}

    async def set(self, key, value, ex=None):
        self.values[key] = value


class _Cycle:
    def __init__(self, sink) -> None:
        self.events = sink
        self.requests = []

    async def run(self, cycle_request):
        self.requests.append(cycle_request)
        await self.events.publish(CycleEvent("component_completed", {"component_id": "kronos"}))
        return CycleOutcome("cycle-1", "no_change", 1)


class _BlockingBus(_Bus):
    def __init__(self, blocked_event: str) -> None:
        super().__init__()
        self.blocked_event = blocked_event
        self.entered = asyncio.Event()

    async def publish(self, name, data=None):
        self.events.append((name, data or {}))
        if name == self.blocked_event:
            self.entered.set()
            await asyncio.Future()


class _BlockingState(_State):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()

    async def set(self, key, value, ex=None):
        if value == "running":
            self.entered.set()
            await asyncio.Future()
        self.values[key] = value


class _CancellationAwareCycle:
    def __init__(self, bus: _Bus) -> None:
        self.bus = bus
        self.entered = asyncio.Event()

    async def run(self, _cycle_request):
        self.entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await self.bus.publish("cycle_cancelled", {"status": "cancelled"})
            raise


class _EarlyCancelledCycle:
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def run(self, _cycle_request):
        self.entered.set()
        await asyncio.Future()


@pytest.mark.asyncio
async def test_chat_runner_forwards_cycle_events():
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer
    from cryptotrader.chat.event_bus import EventBusCycleSink

    bus = _Bus()
    cycle = _Cycle(EventBusCycleSink(bus))

    await run_analysis_and_buffer(
        pair="BTC/USDT:USDT",
        session_id="s1",
        event_bus=bus,
        interrupt_event=asyncio.Event(),
        state_mgr=_State(),
        cycle=cycle,
    )

    assert any(name == "component_completed" for name, _ in bus.events)
    assert cycle.requests[0].pair.canonical() == "BTC/USDT:USDT"


@pytest.mark.asyncio
async def test_chat_cancel_does_not_publish_partial_verdict():
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer
    from cryptotrader.chat.event_bus import EventBusCycleSink

    bus = _Bus()
    interrupt = asyncio.Event()
    interrupt.set()

    await run_analysis_and_buffer(
        pair="BTC/USDT:USDT",
        session_id="s1",
        event_bus=bus,
        interrupt_event=interrupt,
        state_mgr=_State(),
        cycle=_Cycle(EventBusCycleSink(bus)),
    )

    names = [name for name, _ in bus.events]
    assert "verdict_partial" not in names
    assert "cycle_cancelled" in names


@pytest.mark.asyncio
async def test_task_cancellation_during_session_start_publishes_one_terminal_sequence():
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer

    bus = _BlockingBus("session_start")
    state = _State()
    task = asyncio.create_task(
        run_analysis_and_buffer(
            pair="BTC/USDT:USDT",
            session_id="s1",
            event_bus=bus,
            interrupt_event=asyncio.Event(),
            state_mgr=state,
            cycle=_Cycle(bus),
        )
    )
    await bus.entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    names = [name for name, _ in bus.events]
    assert names == ["session_start", "cycle_cancelled", "stream_done"]
    assert state.values["analysis:status:s1"] == "cancelled"


@pytest.mark.asyncio
async def test_task_cancellation_during_running_status_write_publishes_one_terminal_sequence():
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer

    bus = _Bus()
    state = _BlockingState()
    task = asyncio.create_task(
        run_analysis_and_buffer(
            pair="BTC/USDT:USDT",
            session_id="s1",
            event_bus=bus,
            interrupt_event=asyncio.Event(),
            state_mgr=state,
            cycle=_Cycle(bus),
        )
    )
    await state.entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    names = [name for name, _ in bus.events]
    assert names == ["session_start", "cycle_cancelled", "stream_done"]
    assert state.values["analysis:status:s1"] == "cancelled"


@pytest.mark.asyncio
async def test_cycle_cancellation_does_not_duplicate_existing_cancelled_terminal():
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer

    bus = _Bus()
    state = _State()
    cycle = _CancellationAwareCycle(bus)
    task = asyncio.create_task(
        run_analysis_and_buffer(
            pair="BTC/USDT:USDT",
            session_id="s1",
            event_bus=bus,
            interrupt_event=asyncio.Event(),
            state_mgr=state,
            cycle=cycle,
        )
    )
    await cycle.entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    names = [name for name, _ in bus.events]
    assert names.count("cycle_cancelled") == 1
    assert names[-2:] == ["cycle_cancelled", "stream_done"]
    assert state.values["analysis:status:s1"] == "cancelled"


@pytest.mark.asyncio
async def test_cycle_cancellation_before_cycle_terminal_still_publishes_cancelled_terminal():
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer

    bus = _Bus()
    state = _State()
    cycle = _EarlyCancelledCycle()
    task = asyncio.create_task(
        run_analysis_and_buffer(
            pair="BTC/USDT:USDT",
            session_id="s1",
            event_bus=bus,
            interrupt_event=asyncio.Event(),
            state_mgr=state,
            cycle=cycle,
        )
    )
    await cycle.entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    names = [name for name, _ in bus.events]
    assert names.count("cycle_cancelled") == 1
    assert names[-2:] == ["cycle_cancelled", "stream_done"]
    assert state.values["analysis:status:s1"] == "cancelled"
