"""The single account-free market → components → fusion → target pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from cryptotrader.decision.exit_policy import exit_candle_requirement
from cryptotrader.signals.models import CandleRequirement, ComponentSignal, DataRequirements, SignalContext
from cryptotrader.signals.presentation import freeze_evaluation_reference
from cryptotrader.signals.runner import ComponentRunError

if TYPE_CHECKING:
    from cryptotrader.decision.models import TargetPosition
    from cryptotrader.signals.fusion import FusedSignal


class AnalysisFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    stage: str
    message: str


@dataclass(frozen=True)
class AnalysisResult:
    context: SignalContext | None
    component_signals: tuple[ComponentSignal, ...]
    fused_signal: FusedSignal | None
    target_position: TargetPosition | None
    failure: AnalysisFailure | None


class SignalAnalysisService:
    """No account, approval, execution or venue capability is accepted here."""

    def __init__(self, *, market_source, registry, runner, fusion, decisions, clock=None):
        self.market_source = market_source
        self.registry = registry
        self.runner = runner
        self.fusion = fusion
        self.decisions = decisions
        self.clock = clock or (lambda: datetime.now(UTC))

    async def analyze(self, pair, snapshot, as_of: datetime) -> AnalysisResult:
        context = None
        signals = ()
        fused = target = None
        stage = "market"
        try:
            if as_of.tzinfo is None:
                raise ValueError("analysis time must be timezone-aware")
            document = snapshot.document
            profile = document.signals.to_profile(snapshot.revision)
            components = self.registry.enabled(profile)
            # Exit candles have an explicit configured timeframe; ordering does
            # not determine ATR, and evaluation only adds its reference candles.
            requirements = DataRequirements.merge(
                *(component.requirements() for component in components),
                DataRequirements(candles=(exit_candle_requirement(document.market_data),)),
                DataRequirements(candles=(CandleRequirement(document.market_data.timeframe, 2),)),
            )
            context = await self.market_source.collect(pair, as_of, requirements)
            if context.pair != pair or context.as_of != as_of or context.market_data_source_id != self.market_source.id:
                raise ValueError("market context identity mismatch")
            context = replace(
                context,
                evaluation_reference=freeze_evaluation_reference(
                    context,
                    document.market_data.timeframe,
                    document.signals.evaluation_interval,
                ),
            )
            stage = "components"
            signals = await self.runner.run(components, context)
            stage = "fusion"
            fused = self.fusion.fuse(signals, profile.components)
            stage = "target"
            target = self.decisions.target_for(fused, profile)
        except ComponentRunError as error:
            signals = error.component_signals
            stages = {getattr(failure, "stage", "evaluation") for failure in error.errors.values()}
            safe_stages = {
                "gate_loading",
                "feature_computation",
                "gate_classification",
                "input_data",
                "prediction",
                "prediction_output",
            }
            component_stage = next(iter(stages)) if len(stages) == 1 else "evaluation"
            failed_stage = f"components.{component_stage}" if component_stage in safe_stages else "components"
            return AnalysisResult(
                context,
                signals,
                None,
                None,
                AnalysisFailure(
                    code="component_failed",
                    stage=failed_stage,
                    message="组件分析失败。当次结果已保留。",
                ),
            )
        except Exception:
            return AnalysisResult(
                context,
                signals,
                fused,
                target,
                AnalysisFailure(
                    code=f"{stage}_failed",
                    stage=stage,
                    message="分析阶段失败。未执行任何交易。",
                ),
            )
        return AnalysisResult(context, signals, fused, target, None)
