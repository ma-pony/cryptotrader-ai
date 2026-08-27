"""融合信号之后的不可变决策模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime

    from cryptotrader.execution.service import ExecutionResult
    from cryptotrader.pair import Pair
    from cryptotrader.risk.models import RiskDecision
    from cryptotrader.signals.fusion import FusedSignal
    from cryptotrader.signals.models import ComponentSignal, TradingMode

CycleStatus = Literal[
    "completed",
    "no_change",
    "awaiting_approval",
    "approval_rejected",
    "component_failed",
    "risk_rejected",
    "execution_failed",
    "cancelled",
]


@dataclass(frozen=True)
class TargetPosition:
    """风险上限之内希望达到的最终仓位。"""

    side: Literal["long", "short", "flat"]
    size_ratio: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.size_ratio <= 1.0:
            raise ValueError("size_ratio must be in [0, 1]")
        if self.side == "flat" and self.size_ratio != 0.0:
            raise ValueError("flat target requires size_ratio=0")
        if self.side != "flat" and self.size_ratio == 0.0:
            raise ValueError("non-flat target requires positive size_ratio")

    @property
    def signed_ratio(self) -> float:
        return {"long": self.size_ratio, "short": -self.size_ratio, "flat": 0.0}[self.side]


@dataclass(frozen=True)
class TradePlan:
    """由统一退出策略补全后的最终交易计划。"""

    target: TargetPosition
    stop_loss: float | None
    take_profit: float | None
    component_signals: tuple[ComponentSignal, ...]
    fused_signal: FusedSignal


@dataclass(frozen=True)
class CycleRequest:
    """启动一次决策周期所需的输入。"""

    pair: Pair
    mode: TradingMode
    exchange_id: str = ""
    as_of: datetime | None = None


@dataclass(frozen=True)
class CycleOutcome:
    """所有决策周期共用的终止结果。"""

    cycle_id: str
    status: CycleStatus
    profile_revision: int
    component_signals: tuple[ComponentSignal, ...] = ()
    fused_signal: FusedSignal | None = None
    trade_plan: TradePlan | None = None
    risk_result: RiskDecision | None = None
    execution_result: ExecutionResult | None = None
    approval_id: str | None = None
    error: str | None = None
