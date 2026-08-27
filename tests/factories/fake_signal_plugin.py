"""Registry 动态导入测试使用的真实 Python factory。"""

from __future__ import annotations

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
