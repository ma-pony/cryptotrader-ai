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
