"""并发组件执行必须全成全败且保持 Profile 顺序。"""

from __future__ import annotations

import asyncio

import pytest

from cryptotrader.signals.models import DataRequirements
from tests.factories.signal_fusion import context, signal


class RecordingSink:
    def __init__(self) -> None:
        self.events = []

    async def publish(self, event) -> None:
        self.events.append(event)


class FakeComponent:
    display_name = "Fake"
    description = "test component"

    def __init__(
        self,
        component_id: str,
        *,
        delay=0.0,
        error: BaseException | None = None,
        result=None,
    ) -> None:
        self.id = component_id
        self.delay = delay
        self.error = error
        self.result = result

    def requirements(self):
        return DataRequirements()

    async def evaluate(self, signal_context):
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return signal(self.id)


@pytest.mark.asyncio
async def test_runner_returns_signals_in_configured_order():
    from cryptotrader.signals.runner import ComponentRunner

    sink = RecordingSink()
    result = await ComponentRunner(sink).run(
        (FakeComponent("kronos", delay=0.02), FakeComponent("llm_committee")),
        context(),
    )

    assert [item.component_id for item in result] == ["kronos", "llm_committee"]
    assert [event.name for event in sink.events].count("component_started") == 2
    assert [event.name for event in sink.events].count("component_completed") == 2


@pytest.mark.asyncio
async def test_runner_raises_one_error_containing_all_component_failures():
    from cryptotrader.signals.runner import ComponentRunError, ComponentRunner

    sink = RecordingSink()
    with pytest.raises(ComponentRunError) as caught:
        await ComponentRunner(sink).run(
            (
                FakeComponent("kronos", error=RuntimeError("model down")),
                FakeComponent("llm_committee", error=ValueError("bad response")),
            ),
            context(),
        )

    assert set(caught.value.errors) == {"kronos", "llm_committee"}
    failed_events = [event for event in sink.events if event.name == "component_failed"]
    assert {event.data["component_id"] for event in failed_events} == {"kronos", "llm_committee"}


@pytest.mark.asyncio
async def test_one_failure_discards_other_successes():
    from cryptotrader.signals.runner import ComponentRunError, ComponentRunner

    with pytest.raises(ComponentRunError):
        await ComponentRunner(RecordingSink()).run(
            (FakeComponent("kronos"), FakeComponent("llm_committee", error=RuntimeError("down"))),
            context(),
        )


@pytest.mark.asyncio
async def test_component_cancellation_is_not_wrapped_as_failure():
    from cryptotrader.signals.runner import ComponentRunner

    sink = RecordingSink()
    with pytest.raises(asyncio.CancelledError):
        await ComponentRunner(sink).run(
            (FakeComponent("kronos", error=asyncio.CancelledError()),),
            context(),
        )

    assert not [event for event in sink.events if event.name == "component_failed"]


@pytest.mark.asyncio
async def test_event_payload_identifies_component_without_provider_payload():
    from cryptotrader.signals.runner import ComponentRunner

    sink = RecordingSink()
    await ComponentRunner(sink).run((FakeComponent("kronos"),), context())

    completed = next(event for event in sink.events if event.name == "component_completed")
    assert completed.data == {
        "component_id": "kronos",
        "direction": "long",
        "confidence": 0.8,
    }


@pytest.mark.asyncio
async def test_component_events_do_not_expose_signal_or_error_secrets():
    from cryptotrader.signals.runner import ComponentRunError, ComponentRunner

    sensitive_marker = "provider-sensitive-marker"
    sink = RecordingSink()
    with pytest.raises(ComponentRunError):
        await ComponentRunner(sink).run(
            (
                FakeComponent("safe", result=signal("safe", reasoning=sensitive_marker)),
                FakeComponent("failed", error=RuntimeError(sensitive_marker)),
            ),
            context(),
        )

    assert sensitive_marker not in repr(sink.events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result",
    [signal("wrong-id"), {"direction": "long", "confidence": 0.8}],
)
async def test_invalid_plugin_output_is_reported_as_component_failure(invalid_result):
    from cryptotrader.signals.runner import ComponentRunError, ComponentRunner

    sink = RecordingSink()
    with pytest.raises(ComponentRunError) as caught:
        await ComponentRunner(sink).run(
            (FakeComponent("custom", result=invalid_result),),
            context(),
        )

    assert set(caught.value.errors) == {"custom"}
    failed = next(event for event in sink.events if event.name == "component_failed")
    assert failed.data["component_id"] == "custom"


def test_component_signal_rejects_unknown_direction_at_runtime():
    from cryptotrader.signals.models import ComponentSignal

    with pytest.raises(ValueError, match="direction"):
        ComponentSignal("custom", "up", 0.8, "invalid")
