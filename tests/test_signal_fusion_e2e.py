"""信号组件事件、融合与多资金池目标的端到端契约。"""

from __future__ import annotations

import pytest

from cryptotrader.decision.models import CycleRequest
from cryptotrader.signals.models import ComponentSignal, DataRequirements
from cryptotrader.signals.runner import ComponentRunner
from tests.test_multi_book_cycle import PAIR, _book, _cycle, _snapshot


class _Sink:
    def __init__(self) -> None:
        self.events = []

    async def publish(self, event) -> None:
        self.events.append(event)


class _Component:
    id = "fixture"
    display_name = "Fixture"
    description = "Fixture signal"

    def requirements(self):
        return DataRequirements()

    async def evaluate(self, context):
        return ComponentSignal(self.id, "long", 1.0, "market evidence")


class _Registry:
    def enabled(self, profile):
        return (_Component(),)


@pytest.mark.asyncio
async def test_component_events_and_fused_target_are_shared_by_all_books():
    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    live = _book("live", "real", ("live-first", "live-second"), hitl=False)
    cycle, _, coordinator, _, _ = _cycle(_snapshot(simulation, live))
    sink = _Sink()
    cycle.registry = _Registry()
    cycle.runner = ComponentRunner(sink)
    cycle.events = sink

    outcome = await cycle.run(CycleRequest(PAIR))

    assert outcome.target_position.side == "long"
    assert outcome.target_position.size_ratio == 1.0
    assert [proposal.book_id for proposal in coordinator.proposals] == ["simulation", "live"]
    assert [event.name for event in sink.events].count("component_started") == 1
    assert [event.name for event in sink.events].count("component_completed") == 1
    assert [event.name for event in sink.events].count("book_proposed") == 2
