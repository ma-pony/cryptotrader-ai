class StaticCycleLease:
    def __init__(self, cycle) -> None:
        self.cycle = cycle

    async def __aenter__(self):
        return self.cycle

    async def __aexit__(self, *_exc) -> None:
        return None


def static_cycle_lease(cycle):
    return lambda: StaticCycleLease(cycle)
