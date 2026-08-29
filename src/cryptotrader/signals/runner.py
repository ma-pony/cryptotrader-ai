"""Run enabled signal components concurrently with all-success semantics."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from cryptotrader.cycle_events import CycleEvent
from cryptotrader.signals.models import ComponentSignal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cryptotrader.cycle_events import CycleEventSink
    from cryptotrader.signals.component import SignalComponent
    from cryptotrader.signals.models import SignalContext


class ComponentRunError(RuntimeError):
    def __init__(self, errors: Mapping[str, BaseException]) -> None:
        self.errors = dict(errors)
        component_ids = ", ".join(self.errors)
        super().__init__(f"signal components failed: {component_ids}")


class ComponentRunner:
    def __init__(self, events: CycleEventSink) -> None:
        self.events = events

    async def run(
        self,
        components: Sequence[SignalComponent],
        context: SignalContext,
    ) -> tuple[ComponentSignal, ...]:
        results = await asyncio.gather(
            *(self._run_one(component, context) for component in components),
            return_exceptions=True,
        )
        cancellation = next((item for item in results if isinstance(item, asyncio.CancelledError)), None)
        if cancellation is not None:
            raise cancellation
        errors = {
            component.id: result
            for component, result in zip(components, results, strict=True)
            if isinstance(result, BaseException)
        }
        if errors:
            raise ComponentRunError(errors)
        return tuple(cast("ComponentSignal", item) for item in results)

    async def _run_one(
        self,
        component: SignalComponent,
        context: SignalContext,
    ) -> ComponentSignal:
        await self.events.publish(CycleEvent("component_started", {"component_id": component.id}))
        try:
            result = await component.evaluate(context)
            if not isinstance(result, ComponentSignal):
                raise TypeError("component must return ComponentSignal")
            if result.component_id != component.id:
                raise ValueError(
                    f"component {component.id!r} returned signal for {result.component_id!r}",
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.events.publish(
                CycleEvent(
                    "component_failed",
                    {
                        "component_id": component.id,
                        "stage": "evaluation",
                        "error_type": type(error).__name__,
                    },
                )
            )
            raise
        await self.events.publish(
            CycleEvent(
                "component_completed",
                {
                    "component_id": component.id,
                    "direction": result.direction,
                    "confidence": result.confidence,
                },
            )
        )
        return result
