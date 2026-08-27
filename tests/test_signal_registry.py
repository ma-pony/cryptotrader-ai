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
        hitl_required=False,
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


def test_registry_loads_installed_python_factory():
    from cryptotrader.signals.registry import SignalComponentRegistry

    registry = SignalComponentRegistry()
    registry.load_factory("tests.factories.fake_signal_plugin:create_component")

    assert registry.ids() == ("factory_component",)


def test_registry_rejects_factory_that_returns_non_component():
    from cryptotrader.signals.registry import SignalComponentRegistry

    registry = SignalComponentRegistry()

    with pytest.raises(TypeError, match="did not return SignalComponent"):
        registry.load_factory("tests.factories.fake_signal_plugin:create_invalid_component")


def test_component_execution_error_keeps_component_and_cause():
    from cryptotrader.signals.component import ComponentExecutionError

    cause = RuntimeError("offline")
    error = ComponentExecutionError("fake", cause)

    assert error.component_id == "fake"
    assert error.cause is cause
    assert str(error) == "signal component fake failed: RuntimeError: offline"
