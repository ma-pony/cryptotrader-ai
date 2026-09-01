"""融合信号之后的不可变决策模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cryptotrader.journal.models import BookCycleResult
    from cryptotrader.pair import Pair
    from cryptotrader.signals.fusion import FusedSignal
    from cryptotrader.signals.models import ComponentSignal

CycleStatus = Literal[
    "completed",
    "no_change",
    "awaiting_approval",
    "approval_rejected",
    "component_failed",
    "cycle_failed",
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
    mode: Literal["trading", "backtest"] = "trading"
    origin: Literal["manual", "scheduled", "trigger", "backtest"] = "manual"
    decision_id: str | None = None
    confirmed_book_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CycleOutcome:
    """一次平台无关信号周期的严格 Journal 投影。"""

    cycle_id: str
    config_revision: int
    target_position: TargetPosition | None
    books: tuple[BookCycleResult, ...]
    status: str
    execution_status: str
    requires_attention: bool

    def book(self, book_id: str) -> BookCycleResult:
        matches = tuple(item for item in self.books if item.book_id == book_id)
        if len(matches) != 1:
            raise LookupError(f"book {book_id!r} does not exist in cycle {self.cycle_id}")
        return matches[0]
