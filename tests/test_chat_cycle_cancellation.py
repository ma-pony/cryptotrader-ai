"""Chat 只转发 CycleEvent。取消不产生部分裁决。"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cryptotrader.cycle_events import CycleEvent
from cryptotrader.decision.models import CycleOutcome
from tests.runtime_lease import static_cycle_lease


@pytest.fixture(autouse=True)
def _reset_task_manager():
    from cryptotrader.chat.task_manager import BackgroundTaskManager

    BackgroundTaskManager.reset()
    yield
    BackgroundTaskManager.reset()


class _Bus:
    def __init__(self) -> None:
        self.events = []
        self._execution_started = False

    @property
    def execution_started(self):
        return self._execution_started

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

    return SimpleNamespace(
        cycle=cycle,
        events=events,
        snapshot=SimpleNamespace(revision=1),
        cycle_lease=static_cycle_lease(cycle),
    )


@pytest.mark.asyncio
async def test_chat_reloads_once_reports_each_book_and_publishes_strict_terminal() -> None:
    from cryptotrader.chat.analysis_runner import run_analysis_and_buffer
    from cryptotrader.cycle_events import MultiplexedCycleEventSink, NullCycleEventSink

    books = (
        SimpleNamespace(
            book_id="simulation",
            capital_scope="simulated",
            status="completed",
            execution=SimpleNamespace(status="completed", requires_attention=False),
            raw_error="must never escape",
        ),
        SimpleNamespace(
            book_id="live",
            capital_scope="real",
            status="partial",
            execution=SimpleNamespace(status="partial", requires_attention=True),
            raw_error="credential and adapter detail",
        ),
    )

    class Cycle:
        def __init__(self) -> None:
            self.requests = []

        async def run(self, request):
            self.requests.append(request)
            return CycleOutcome("chat-cycle", 12, None, books, "partial", "partial", True)

    cycle = Cycle()
    runtime = SimpleNamespace(
        cycle=cycle,
        events=MultiplexedCycleEventSink(NullCycleEventSink()),
        snapshot=SimpleNamespace(revision=11),
        cycle_lease=static_cycle_lease(cycle),
    )
    bus = _Bus()

    outcome = await run_analysis_and_buffer(
        pair="BTC/USDT",
        session_id="books",
        event_bus=bus,
        interrupt_event=asyncio.Event(),
        state_mgr=_State(),
        runtime=runtime,
    )

    assert outcome.cycle_id == "chat-cycle"
    assert len(cycle.requests) == 1
    book_events = [data for name, data in bus.events if name == "book_result"]
    assert book_events == [
        {
            "book_id": "simulation",
            "capital_scope": "simulated",
            "status": "completed",
            "execution_status": "completed",
            "requires_attention": False,
        },
        {
            "book_id": "live",
            "capital_scope": "real",
            "status": "partial",
            "execution_status": "partial",
            "requires_attention": True,
        },
    ]
    terminal = next(data for name, data in bus.events if name == "stream_done")
    assert terminal == {
        "session_id": "books",
        "cycle_id": "chat-cycle",
        "config_revision": 12,
        "status": "partial",
        "execution_status": "partial",
        "requires_attention": True,
    }
    assert "raw_error" not in repr(bus.events)


@pytest.mark.asyncio
async def test_interrupt_after_execution_started_waits_and_returns_exact_outcome() -> None:
    from api.routes.chat_control import interrupt_analysis
    from cryptotrader.chat.event_buffer import EventBuffer
    from cryptotrader.chat.event_bus import EventBus
    from cryptotrader.chat.task_manager import BackgroundTaskManager
    from cryptotrader.risk.state import RedisStateManager

    state = RedisStateManager(None)
    bus = EventBus("execution-session", EventBuffer("execution-session", state))
    release = asyncio.Event()

    async def runner(_interrupt_event):
        await bus.publish(
            "book_execution_started",
            {"cycle_id": "order-cycle", "config_revision": 14, "book_id": "live"},
        )
        await release.wait()
        return CycleOutcome("order-cycle", 14, None, (), "partial", "partial", True)

    manager = BackgroundTaskManager.get_instance()
    analysis = manager.create("execution-session", "BTC/USDT", runner, "chat", bus)
    await bus.wait_for_execution_started()

    response_task = asyncio.create_task(interrupt_analysis("execution-session"))
    await asyncio.sleep(0)

    assert not analysis.task.cancelled()
    assert not analysis.interrupt_event.is_set()
    assert not response_task.done()

    release.set()
    response = await response_task
    assert response.model_dump() == {
        "type": "execution_in_progress",
        "session_id": "execution-session",
        "cycle_id": "order-cycle",
        "status": "partial",
        "execution_status": "partial",
        "requires_attention": True,
    }


@pytest.mark.asyncio
async def test_same_session_replacement_after_execution_started_returns_safe_409() -> None:
    from fastapi import HTTPException

    from api.routes.chat import ChatStreamRequest, _handle_new_analysis
    from cryptotrader.chat.event_buffer import EventBuffer
    from cryptotrader.chat.event_bus import EventBus
    from cryptotrader.chat.task_manager import BackgroundTaskManager
    from cryptotrader.risk.state import RedisStateManager

    state = RedisStateManager(None)
    bus = EventBus("same-session", EventBuffer("same-session", state))
    release = asyncio.Event()

    async def old_runner(_interrupt_event):
        await bus.publish(
            "book_execution_started",
            {"cycle_id": "old-cycle", "config_revision": 15, "book_id": "live"},
        )
        await release.wait()
        return CycleOutcome("old-cycle", 15, None, (), "completed", "completed", False)

    manager = BackgroundTaskManager.get_instance()
    old = manager.create("same-session", "BTC/USDT", old_runner, "chat", bus)
    await bus.wait_for_execution_started()

    runtime = SimpleNamespace(
        cycle=object(),
        snapshot=SimpleNamespace(
            document=SimpleNamespace(
                scheduler=SimpleNamespace(pairs=("BTC/USDT",)),
                infrastructure=SimpleNamespace(redis_url=""),
            )
        ),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime=runtime)))

    with pytest.raises(HTTPException) as error:
        await _handle_new_analysis(
            "same-session",
            ChatStreamRequest(message="BTC/USDT"),
            request,
        )

    assert error.value.status_code == 409
    assert error.value.detail == "Analysis execution is already in progress"
    assert manager.get("same-session") is old
    assert not old.interrupt_event.is_set()
    assert not old.task.cancelled()

    release.set()
    await old.task


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
    runtime.cycle_lease = static_cycle_lease(runtime.cycle)
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
        assert task.outcome == CycleOutcome("cycle-chat", 4, None, (), "completed", "not_started", False)


@pytest.mark.asyncio
async def test_mounted_chat_execution_started_interrupt_returns_exact_cycle_outcome():
    from api.routes.chat import ChatStreamRequest, _handle_new_analysis
    from api.routes.chat_control import interrupt_analysis
    from cryptotrader.chat.task_manager import BackgroundTaskManager
    from cryptotrader.cycle_events import MultiplexedCycleEventSink, NullCycleEventSink

    terminal = asyncio.Event()
    expected = CycleOutcome("mounted-execution", 8, None, (), "completed", "completed", False)

    class RouteCycle:
        async def run(self, _request):
            await self.events.publish(CycleEvent("book_execution_started", {"book_id": "live"}))
            await terminal.wait()
            return expected

    cycle = RouteCycle()
    events = MultiplexedCycleEventSink(NullCycleEventSink())
    cycle.events = events
    runtime = SimpleNamespace(
        cycle=cycle,
        events=events,
        snapshot=SimpleNamespace(
            document=SimpleNamespace(
                scheduler=SimpleNamespace(pairs=()),
                infrastructure=SimpleNamespace(redis_url=""),
            )
        ),
        cycle_lease=static_cycle_lease(cycle),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime=runtime)))

    class FailingAfterExecutionBuffer:
        def __init__(self, *_args, **_kwargs):
            self.sequence = 0
            self.failed = False

        async def next_event_id(self):
            self.sequence += 1
            return self.sequence

        async def push(self, envelope):
            if self.failed:
                raise RuntimeError("terminal observer unavailable")
            if envelope.type == "book_execution_started":
                self.failed = True

    with patch("cryptotrader.chat.event_buffer.EventBuffer", FailingAfterExecutionBuffer):
        await _handle_new_analysis("mounted-execution", ChatStreamRequest(message="BTC/USDT"), request)
    managed = BackgroundTaskManager.get_instance().get("mounted-execution")
    assert managed is not None
    await managed.event_bus.wait_for_execution_started()
    interrupting = asyncio.create_task(interrupt_analysis("mounted-execution"))
    await asyncio.sleep(0)
    assert not interrupting.done()
    terminal.set()

    response = await interrupting
    assert managed.outcome == expected
    assert response.model_dump() == {
        "type": "execution_in_progress",
        "session_id": "mounted-execution",
        "cycle_id": "mounted-execution",
        "status": "completed",
        "execution_status": "completed",
        "requires_attention": False,
    }


@pytest.mark.asyncio
async def test_event_sink_latches_execution_before_buffer_failure_blocks_interrupt_and_replacement():
    from cryptotrader.chat.event_bus import EventBus, EventBusCycleSink
    from cryptotrader.chat.task_manager import BackgroundTaskManager, ExecutionInProgressError
    from cryptotrader.cycle_events import CycleEvent

    class FailingBuffer:
        async def next_event_id(self):
            return 1

        async def push(self, _event):
            raise RuntimeError("buffer unavailable")

    bus = EventBus("failed-observer", FailingBuffer())
    manager = BackgroundTaskManager.get_instance()
    terminal = asyncio.Event()

    with contextlib.suppress(RuntimeError):
        await EventBusCycleSink(bus).publish(CycleEvent("book_execution_started", {"book_id": "live"}))

    async def runner(_interrupt):
        await terminal.wait()
        return CycleOutcome("cycle", 1, None, (), "completed", "completed", False)

    manager.create("failed-observer", "BTC/USDT", runner, "chat", bus)
    assert bus.execution_started
    assert bus.published_count("book_execution_started") == 0
    assert manager.interrupt("failed-observer") is None
    with pytest.raises(ExecutionInProgressError):
        manager.create("failed-observer", "BTC/USDT", runner, "chat", bus)
    terminal.set()
    await manager.drain()


@pytest.mark.asyncio
async def test_background_manager_drain_cancels_pre_execution_and_waits_for_execution_terminal():
    from cryptotrader.chat.task_manager import BackgroundTaskManager

    manager = BackgroundTaskManager.get_instance()
    pre_bus = _Bus()
    execution_bus = _Bus()
    execution_bus._execution_started = True
    terminal = asyncio.Event()

    async def pending(_interrupt):
        await asyncio.Future()

    async def order_bearing(_interrupt):
        await terminal.wait()
        return CycleOutcome("terminal", 1, None, (), "completed", "completed", False)

    pre = manager.create("pre", "BTC/USDT", pending, "chat", pre_bus)
    bearing = manager.create("bearing", "BTC/USDT", order_bearing, "chat", execution_bus)
    draining = asyncio.create_task(manager.drain())
    await asyncio.sleep(0)
    assert pre.interrupt_event.is_set()
    assert not draining.done()
    terminal.set()
    await draining
    assert bearing.outcome.cycle_id == "terminal"


@pytest.mark.asyncio
async def test_cancelled_manager_drain_still_records_order_bearing_outcome():
    from cryptotrader.chat.task_manager import BackgroundTaskManager

    manager = BackgroundTaskManager.get_instance()
    bus = _Bus()
    bus._execution_started = True
    terminal = asyncio.Event()
    child_cancelled = False

    async def order_bearing(_interrupt):
        nonlocal child_cancelled
        try:
            await terminal.wait()
        except asyncio.CancelledError:
            child_cancelled = True
            raise
        return CycleOutcome("drained", 3, None, (), "partial", "partial", True)

    managed = manager.create("drained", "BTC/USDT", order_bearing, "chat", bus)
    draining = asyncio.create_task(manager.drain())
    await asyncio.sleep(0)
    draining.cancel()
    await asyncio.sleep(0)
    draining.cancel()
    assert draining.done() is False
    assert child_cancelled is False

    terminal.set()
    with pytest.raises(asyncio.CancelledError):
        await draining
    assert managed.task.cancelled() is False
    assert managed.outcome == CycleOutcome("drained", 3, None, (), "partial", "partial", True)


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
    runtime.cycle_lease = static_cycle_lease(cycle)
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
