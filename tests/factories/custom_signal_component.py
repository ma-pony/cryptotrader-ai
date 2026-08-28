"""Registry factory test component."""

from cryptotrader.signals.models import ComponentSignal, DataRequirements

factory_calls = 0


class FakeSignalComponent:
    id = "fake"
    display_name = "Fake"
    description = "Bootstrap test component"

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def evaluate(self, context) -> ComponentSignal:
        return ComponentSignal(self.id, "neutral", 0.0, "test")


def create() -> FakeSignalComponent:
    global factory_calls
    factory_calls += 1
    return FakeSignalComponent()
