"""信号组件协议与启动时 Registry 契约。"""

from __future__ import annotations

import pytest

from cryptotrader.profiles.models import ComponentWeight, SignalProfile
from cryptotrader.signals.models import ComponentSignal, DataRequirements, SignalContext


class FakeComponent:
    id = "fake"
    display_name = "Fake"
    description = "test component"

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def evaluate(self, context: SignalContext) -> ComponentSignal:
        return ComponentSignal(self.id, "neutral", 0.0, context.pair.canonical())


class OtherComponent(FakeComponent):
    id = "other"
    display_name = "Other"


def _profile() -> SignalProfile:
    return SignalProfile(
        revision=1,
        components=(
            ComponentWeight("other", False, 0.0),
            ComponentWeight("fake", True, 1.0),
        ),
        neutral_threshold=0.2,
        max_target_ratio=1.0,
        atr_stop_multiplier=2.0,
        reward_ratio=2.0,
    )


def test_registry_rejects_duplicate_component_id():
    from cryptotrader.signals.registry import SignalComponentRegistry

    registry = SignalComponentRegistry()
    registry.register(FakeComponent())

    with pytest.raises(ValueError, match="duplicate"):
        registry.register(FakeComponent())


def test_registry_returns_enabled_components_in_profile_order():
    from cryptotrader.signals.registry import SignalComponentRegistry

    registry = SignalComponentRegistry((FakeComponent(), OtherComponent()))

    assert registry.ids() == ("fake", "other")
    assert registry.enabled(_profile()) == (registry.get("fake"),)


def test_registry_exposes_installed_component_metadata():
    from cryptotrader.signals.registry import ComponentMetadata, SignalComponentRegistry

    registry = SignalComponentRegistry((FakeComponent(),))

    assert registry.metadata() == (
        ComponentMetadata(
            component_id="fake",
            display_name="Fake",
            description="test component",
        ),
    )


def test_registry_rejects_factory_that_returns_non_component(monkeypatch):
    from dataclasses import replace

    from cryptotrader.configuration import registry
    from cryptotrader.cycle_events import NullCycleEventSink
    from cryptotrader.runtime_config.models import SignalComponentConfig
    from cryptotrader.signals.registry import SignalComponentRegistry
    from tests.factories.runtime_config import runtime_document, signal_config
    from tests.factories.workbench_extensions import sample_registry

    extensions, _ = sample_registry()
    extensions.components["sample_signal"] = replace(
        extensions.components["sample_signal"], factory=lambda context: object()
    )
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
    document = runtime_document(
        signals=signal_config(components=(SignalComponentConfig(component_id="sample_signal", enabled=True, weight=1),))
    )
    with pytest.raises(TypeError, match="did not return SignalComponent"):
        SignalComponentRegistry.discover(document, NullCycleEventSink())


def test_component_execution_error_keeps_component_and_cause():
    from cryptotrader.signals.component import ComponentExecutionError

    cause = RuntimeError("offline")
    error = ComponentExecutionError("fake", cause)

    assert error.component_id == "fake"
    assert error.cause is cause
    assert str(error) == "signal component fake failed: RuntimeError: offline"


@pytest.mark.asyncio
async def test_registered_sample_signal_uses_nondefault_configured_window(monkeypatch):
    from datetime import UTC, datetime

    from cryptotrader.configuration import registry
    from cryptotrader.cycle_events import NullCycleEventSink
    from cryptotrader.pair import Pair
    from cryptotrader.runtime_config.models import SignalComponentConfig
    from cryptotrader.signals.registry import SignalComponentRegistry
    from tests.factories.runtime_config import runtime_document, signal_config
    from tests.factories.workbench_extensions import sample_registry

    extensions, calls = sample_registry()
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
    document = runtime_document(
        signals=signal_config(
            components=(
                SignalComponentConfig(component_id="sample_signal", enabled=True, weight=1, parameters={"window": 37}),
            )
        )
    )
    discovered = SignalComponentRegistry.discover(document, NullCycleEventSink())
    pair = Pair.parse("BTC/USDT:USDT")
    result = await discovered.get("sample_signal").evaluate(
        SignalContext(
            pair=pair,
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
            market_data_source_id="fixture-market",
            market_type=pair.market_type,
            current_price=100.0,
            atr=5.0,
            snapshots={},
        )
    )

    assert calls == ["signal"]
    assert result.details["window"] == 37
    assert any(block.kind == "metrics" and block.metrics[0].value == 37 for block in result.blocks)
    assert any(block.kind == "table" and block.rows[0].cells[0].value == 37 for block in result.blocks)
