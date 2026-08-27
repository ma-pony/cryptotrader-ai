"""TradingCycle 业务事件协议。"""

from __future__ import annotations

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


def test_cycle_event_rejects_empty_name():
    from cryptotrader.cycle_events import CycleEvent

    with pytest.raises(ValueError, match="name"):
        CycleEvent("  ")
