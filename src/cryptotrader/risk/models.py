"""目标仓位风控使用的不可变模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.decision.models import TargetPosition, TradePlan
    from cryptotrader.signals.models import SignalContext


@dataclass(frozen=True)
class RiskRequest:
    context: SignalContext
    plan: TradePlan

    @property
    def target(self) -> TargetPosition:
        return self.plan.target

    @property
    def reduces_exposure(self) -> bool:
        current = self.context.current_position.signed_ratio
        target = self.target.signed_ratio
        return target == 0.0 or (current * target > 0.0 and abs(target) <= abs(current))


@dataclass(frozen=True)
class RiskCheckResult:
    passed: bool
    reason: str = ""
    size_ratio_cap: float | None = None


@dataclass(frozen=True)
class RiskDecision:
    passed: bool
    plan: TradePlan
    rejected_by: str = ""
    reason: str = ""
