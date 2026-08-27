"""融合分数到目标仓位的唯一映射。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.decision.models import TargetPosition

if TYPE_CHECKING:
    from cryptotrader.profiles.models import SignalProfile
    from cryptotrader.signals.fusion import FusedSignal


class DecisionEngine:
    def target_for(self, fused: FusedSignal, profile: SignalProfile) -> TargetPosition:
        magnitude = abs(fused.score)
        if magnitude <= profile.neutral_threshold:
            return TargetPosition("flat", 0.0)
        size_ratio = (magnitude - profile.neutral_threshold) / (1.0 - profile.neutral_threshold)
        size_ratio *= profile.max_target_ratio
        return TargetPosition("long" if fused.score > 0 else "short", size_ratio)
