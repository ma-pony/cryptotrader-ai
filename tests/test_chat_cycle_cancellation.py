"""Chat 只转发 CycleEvent。取消不产生部分裁决。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

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
async def test_mounted_chat_handler_never_reaches_legacy_load_config():
    from api.routes.chat import ChatStreamRequest, _handle_new_analysis
    from cryptotrader.chat.task_manager import BackgroundTaskManager
    from cryptotrader.cycle_events import MultiplexedCycleEventSink, NullCycleEventSink

    class RouteCycle:
        async def run(self, cycle_request):
            return CycleOutcome("cycle-chat", 4, None, (), "completed", "not_started", False)

    runtime = SimpleNamespace(
        cycle=RouteCycle(),
        events=MultiplexedCycleEventSink(NullCycleEventSink()),
        snapshot=SimpleNamespace(
            document=SimpleNamespace(
                scheduler=SimpleNamespace(pairs=()),
                infrastructure=SimpleNamespace(redis_url=""),
            )
        ),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime=runtime)))

    with patch("cryptotrader.config.load_config", side_effect=AssertionError("legacy load_config reached")):
        await _handle_new_analysis(
            "mounted-chat",
            ChatStreamRequest(message="BTC/USDT:USDT"),
            request,
        )
        task = BackgroundTaskManager.get_instance().get("mounted-chat")
        assert task is not None
        await task.task


@pytest.mark.asyncio
async def test_real_runtime_cycle_routes_concurrent_component_events_to_the_correct_event_bus():
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer
    from cryptotrader.chat.event_buffer import EventBuffer
    from cryptotrader.chat.event_bus import EventBus
    from cryptotrader.cycle_events import MultiplexedCycleEventSink
    from cryptotrader.runtime import Runtime
    from cryptotrader.signals.models import ComponentSignal, DataRequirements
    from cryptotrader.signals.runner import ComponentRunner
    from tests.test_multi_book_cycle import _book, _cycle, _snapshot

    class RuntimeComponent:
        id = "fixture"
        display_name = "Fixture"
        description = "runtime fixture"

        @staticmethod
        def requirements():
            return DataRequirements()

        async def evaluate(self, signal_context):
            await asyncio.sleep(0)
            return ComponentSignal(self.id, "long", 1.0, "fixture")

    class RuntimeRegistry:
        @staticmethod
        def enabled(profile):
            return (RuntimeComponent(),)

    base = _BaseSink()
    routed = MultiplexedCycleEventSink(base)
    snapshot = _snapshot(_book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False))
    cycle, _, _, _, _ = _cycle(snapshot)
    cycle.events = routed
    cycle.runner = ComponentRunner(routed)
    cycle.registry = RuntimeRegistry()
    runtime = Runtime(
        snapshot=snapshot,
        repository=cycle.repository,
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
            pair="BTC/USDT:USDT",
            session_id="second-session",
            event_bus=second_bus,
            interrupt_event=asyncio.Event(),
            state_mgr=second_state,
            runtime=runtime,
        ),
    )

    first_events = await first_bus._buffer.range_after(0)
    second_events = await second_bus._buffer.range_after(0)
    first_component = next(event for event in first_events if event.type == "component_completed")
    second_component = next(event for event in second_events if event.type == "component_completed")
    assert first_component.data["config_revision"] == snapshot.revision
    assert second_component.data["config_revision"] == snapshot.revision
    assert first_component.data["cycle_id"] != second_component.data["cycle_id"]
    base_cycle_ids = {event.data["cycle_id"] for event in base.events if event.name == "component_completed"}
    assert base_cycle_ids == {
        first_component.data["cycle_id"],
        second_component.data["cycle_id"],
    }
    assert not any(event.data.get("cycle_id") == second_component.data["cycle_id"] for event in first_events)
    assert not any(event.data.get("cycle_id") == first_component.data["cycle_id"] for event in second_events)


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
