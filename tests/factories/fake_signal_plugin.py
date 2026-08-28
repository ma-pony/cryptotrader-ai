"""Registry 动态导入测试使用的真实 Python factory。"""

from __future__ import annotations

from typing import Any

from cryptotrader.signals.models import ComponentSignal, DataRequirements, SignalContext


class FactoryComponent:
    id = "factory_component"
    display_name = "Factory Component"
    description = "loaded from a Python factory"

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def evaluate(self, context: SignalContext) -> ComponentSignal:
        return ComponentSignal(self.id, "neutral", 0.0, context.pair.canonical())


def create_component() -> FactoryComponent:
    return FactoryComponent()


def create_invalid_component() -> object:
    return object()


class FixtureSignalComponent(FactoryComponent):
    id = "fixture_signal"
    display_name = "Fixture Signal"
    description = "loaded from an installed entry point"

    def __init__(self, document: Any) -> None:
        configured = next(item for item in document.signals.components if item.component_id == self.id)
        self.parameters = configured.parameters


def create_fixture_signal(document, sink) -> FixtureSignalComponent:
    del sink
    return FixtureSignalComponent(document)


def create_conflicting_kronos(document, sink) -> FactoryComponent:
    del document, sink
    component = FactoryComponent()
    component.id = "kronos"
    return component
