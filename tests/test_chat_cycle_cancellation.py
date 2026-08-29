"""Chat 只转发 CycleEvent。取消不产生部分裁决。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

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


class _EventState(_State):
    def __init__(self) -> None:
        super().__init__()
        self.sequence = {}
        self.buffers = {}

    async def incr(self, key):
        self.sequence[key] = self.sequence.get(key, 0) + 1
        return self.sequence[key]

    async def expire(self, key, ttl):
        return True

    async def buffer_len(self, key):
        return len(self.buffers.get(key, ()))

    async def buffer_push(self, key, value, max_size, ttl):
        self.buffers.setdefault(key, []).append(value)
        self.buffers[key] = self.buffers[key][-max_size:]

    async def buffer_range(self, key, start, end):
        return list(self.buffers.get(key, ()))


class _Cycle:
    def __init__(self, sink) -> None:
        self.events = sink
        self.requests = []

    async def run(self, cycle_request):
        self.requests.append(cycle_request)
        await self.events.publish(CycleEvent("component_completed", {"component_id": "kronos"}))
        return CycleOutcome("cycle-1", 1, None, (), "no_change", "not_started", False)


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


class _SharedRuntimeCycle:
    def __init__(self, sink) -> None:
        self.events = sink

    async def run(self, cycle_request):
        pair = cycle_request.pair.canonical()
        await asyncio.create_task(self.events.publish(CycleEvent("internal_debate", {"pair": pair})))
        return CycleOutcome(f"cycle-{pair}", 1, None, (), "completed", "not_started", False)


class _BaseSink:
    def __init__(self) -> None:
        self.events = []

    async def publish(self, event):
        self.events.append(event)


def _runtime_for(cycle):
    from cryptotrader.cycle_events import MultiplexedCycleEventSink, NullCycleEventSink

    events = MultiplexedCycleEventSink(NullCycleEventSink())
    if hasattr(cycle, "events"):
        cycle.events = events
    return SimpleNamespace(cycle=cycle, events=events)


@pytest.mark.asyncio
async def test_shared_runtime_routes_concurrent_component_events_to_the_correct_real_event_bus():
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer
    from cryptotrader.chat.event_buffer import EventBuffer
    from cryptotrader.chat.event_bus import EventBus
    from cryptotrader.cycle_events import MultiplexedCycleEventSink
    from cryptotrader.runtime import Runtime

    base = _BaseSink()
    routed = MultiplexedCycleEventSink(base)
    cycle = _SharedRuntimeCycle(routed)
    runtime = Runtime(
        snapshot=SimpleNamespace(),
        repository=object(),
        cycle=cycle,
        sessions={},
        signal_registry=object(),
        market_registry=object(),
        venue_registry=object(),
        events=routed,
    )
    first_state = _EventState()
    second_state = _EventState()
    first_bus = EventBus("first-session", EventBuffer("first-session", first_state))
    second_bus = EventBus("second-session", EventBuffer("second-session", second_state))

    await asyncio.gather(
        run_analysis_and_buffer(
            pair="BTC/USDT:USDT",
            session_id="first-session",
            event_bus=first_bus,
            interrupt_event=asyncio.Event(),
            state_mgr=first_state,
            runtime=runtime,
        ),
        run_analysis_and_buffer(
            pair="ETH/USDT:USDT",
            session_id="second-session",
            event_bus=second_bus,
            interrupt_event=asyncio.Event(),
            state_mgr=second_state,
            runtime=runtime,
        ),
    )

    first_events = await first_bus._buffer.range_after(0)
    second_events = await second_bus._buffer.range_after(0)
    first_debate = next(event for event in first_events if event.type == "internal_debate")
    second_debate = next(event for event in second_events if event.type == "internal_debate")
    assert first_debate.data["pair"] == "BTC/USDT:USDT"
    assert second_debate.data["pair"] == "ETH/USDT:USDT"
    assert [event.data["pair"] for event in base.events if event.name == "internal_debate"] == [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
    ]


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
        runtime=_runtime_for(cycle),
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
        runtime=_runtime_for(_Cycle(EventBusCycleSink(bus))),
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
            runtime=_runtime_for(_Cycle(bus)),
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
            runtime=_runtime_for(_Cycle(bus)),
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
            runtime=_runtime_for(cycle),
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
            runtime=_runtime_for(cycle),
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
