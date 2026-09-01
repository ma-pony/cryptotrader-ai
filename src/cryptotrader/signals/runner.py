"""Run enabled signal components concurrently with all-success semantics."""
# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING, cast

from cryptotrader.cycle_events import CycleEvent
from cryptotrader.signals.component import ComponentExecutionError
from cryptotrader.signals.models import ComponentSignal
from cryptotrader.signals.presentation import TextBlock

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cryptotrader.cycle_events import CycleEventSink
    from cryptotrader.signals.component import SignalComponent
    from cryptotrader.signals.models import SignalContext


class ComponentRunError(RuntimeError):
    def __init__(
        self, errors: Mapping[str, BaseException], component_signals: tuple[ComponentSignal, ...] = ()
    ) -> None:
        self.errors = dict(errors)
        self.component_signals = component_signals
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
        signals = tuple(result.signal if isinstance(result, _FailedEvaluationError) else result for result in results)
        if errors:
            raise ComponentRunError(errors, cast("tuple[ComponentSignal, ...]", signals))
        return tuple(cast("ComponentSignal", item) for item in results)

    async def _run_one(
        self,
        component: SignalComponent,
        context: SignalContext,
    ) -> ComponentSignal:
        await self.events.publish(CycleEvent("component_started", {"component_id": component.id}))
        started = perf_counter()
        try:
            result = await component.evaluate(context)
            if not isinstance(result, ComponentSignal):
                raise TypeError("component must return ComponentSignal")
            if result.component_id != component.id:
                raise ValueError(
                    f"component {component.id!r} returned signal for {result.component_id!r}",
                )
            result = replace(
                result,
                duration_ms=int((perf_counter() - started) * 1000),
                evaluation_reference=context.evaluation_reference,
            )
            if result.status == "failed":
                raise _FailedEvaluationError(result)
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
            signal = (
                error.signal
                if isinstance(error, _FailedEvaluationError)
                else ComponentSignal(
                    component.id,
                    "neutral",
                    0.0,
                    "组件执行失败，未参与融合。",
                    status="failed",
                    blocks=(
                        TextBlock(
                            title="失败原因", body=f"组件执行失败（{type(error).__name__}）；原始错误正文未保存。"
                        ),
                    ),
                    evaluation_reference=context.evaluation_reference,
                    duration_ms=int((perf_counter() - started) * 1000),
                )
            )
            failure = _FailedEvaluationError(signal)
            failure.stage = error.stage if isinstance(error, ComponentExecutionError) else "evaluation"
            raise failure from error
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


class _FailedEvaluationError(RuntimeError):
    def __init__(self, signal: ComponentSignal):
        self.signal = signal
        super().__init__("component evaluation failed")
