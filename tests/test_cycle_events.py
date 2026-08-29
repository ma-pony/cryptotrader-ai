"""TradingCycle 业务事件协议。"""

from __future__ import annotations

import asyncio
from datetime import UTC

import pytest


def test_cycle_event_has_utc_timestamp_and_payload():
    from cryptotrader.cycle_events import CycleEvent

    event = CycleEvent("component_started", {"component_id": "kronos"})

    assert event.name == "component_started"
    assert event.data == {"component_id": "kronos"}
    assert event.timestamp.tzinfo is UTC


@pytest.mark.asyncio
async def test_null_sink_accepts_events():
    from cryptotrader.cycle_events import CycleEvent, NullCycleEventSink

    await NullCycleEventSink().publish(CycleEvent("cycle_started"))


class _RecordingSink:
    def __init__(self) -> None:
        self.events = []

    async def publish(self, event) -> None:
        self.events.append(event)


class _FailingSink:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def publish(self, event) -> None:
        raise self.error


class _ControlFlow(BaseException):
    pass


@pytest.mark.asyncio
async def test_multiplexed_sink_attributes_concurrent_events_without_cross_talk():
    from cryptotrader.cycle_events import CycleEvent, MultiplexedCycleEventSink

    base = _RecordingSink()
    first = _RecordingSink()
    second = _RecordingSink()
    sink = MultiplexedCycleEventSink(base)

    async def publish(cycle_id, revision, routed):
        with sink.cycle(cycle_id, revision), sink.route(routed):
            await asyncio.create_task(sink.publish(CycleEvent("component_started", {"component_id": "kronos"})))

    await asyncio.gather(
        publish("cycle-1", 11, first),
        publish("cycle-2", 12, second),
    )

    assert {(event.data["cycle_id"], event.data["config_revision"]) for event in base.events} == {
        ("cycle-1", 11),
        ("cycle-2", 12),
    }
    assert [event.data["cycle_id"] for event in first.events] == ["cycle-1"]
    assert [event.data["cycle_id"] for event in second.events] == ["cycle-2"]


@pytest.mark.asyncio
async def test_routed_subscriber_exception_does_not_abort_base_or_cycle():
    from cryptotrader.cycle_events import CycleEvent, MultiplexedCycleEventSink

    base = _RecordingSink()
    healthy = _RecordingSink()
    sink = MultiplexedCycleEventSink(base)

    with sink.route(_FailingSink(RuntimeError("observer failed"))), sink.route(healthy):
        await sink.publish(CycleEvent("component_started"))

    assert [event.name for event in base.events] == ["component_started"]
    assert [event.name for event in healthy.events] == ["component_started"]


@pytest.mark.asyncio
async def test_routed_subscriber_cancellation_propagates_after_base_observes_event():
    from cryptotrader.cycle_events import CycleEvent, MultiplexedCycleEventSink

    base = _RecordingSink()
    sink = MultiplexedCycleEventSink(base)

    with sink.route(_FailingSink(asyncio.CancelledError())), pytest.raises(asyncio.CancelledError):
        await sink.publish(CycleEvent("component_started"))

    assert [event.name for event in base.events] == ["component_started"]


@pytest.mark.asyncio
async def test_routed_subscriber_control_flow_base_exception_propagates():
    from cryptotrader.cycle_events import CycleEvent, MultiplexedCycleEventSink

    sink = MultiplexedCycleEventSink(_RecordingSink())

    with sink.route(_FailingSink(_ControlFlow())), pytest.raises(_ControlFlow):
        await sink.publish(CycleEvent("component_started"))


@pytest.mark.asyncio
async def test_base_sink_failure_remains_a_cycle_failure():
    from cryptotrader.cycle_events import CycleEvent, MultiplexedCycleEventSink

    sink = MultiplexedCycleEventSink(_FailingSink(RuntimeError("base failed")))

    with pytest.raises(RuntimeError, match="base failed"):
        await sink.publish(CycleEvent("component_started"))


def test_cycle_event_rejects_empty_name():
    from cryptotrader.cycle_events import CycleEvent

    with pytest.raises(ValueError, match="name"):
        CycleEvent("  ")
