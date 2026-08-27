"""不依赖 LLM 的确定性信号加权融合。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.profiles.models import ComponentWeight
    from cryptotrader.signals.models import ComponentSignal


@dataclass(frozen=True)
class ComponentContribution:
    component_id: str
    weight: float
    signed_score: float
    weighted_score: float


@dataclass(frozen=True)
class FusedSignal:
    score: float
    contributions: tuple[ComponentContribution, ...]
    reasoning: str


class WeightedSignalFusion:
    """按 Profile 原始权重融合全部启用组件; 运行时不归一化。"""

    def fuse(
        self,
        signals: tuple[ComponentSignal, ...],
        weights: tuple[ComponentWeight, ...],
    ) -> FusedSignal:
        signal_by_id = {item.component_id: item for item in signals}
        if len(signal_by_id) != len(signals):
            raise ValueError("duplicate component signals")

        contributions: list[ComponentContribution] = []
        for configured in (item for item in weights if item.enabled):
            signal = signal_by_id.get(configured.component_id)
            if signal is None:
                raise ValueError(f"missing signal for enabled component {configured.component_id}")
            signed_score = {
                "long": signal.confidence,
                "short": -signal.confidence,
                "neutral": 0.0,
            }[signal.direction]
            contributions.append(
                ComponentContribution(
                    component_id=configured.component_id,
                    weight=configured.weight,
                    signed_score=signed_score,
                    weighted_score=configured.weight * signed_score,
                )
            )

        score = sum(item.weighted_score for item in contributions)
        reasoning = " | ".join(f"{item.component_id}: {item.weighted_score:+.4f}" for item in contributions)
        return FusedSignal(score=score, contributions=tuple(contributions), reasoning=reasoning)
