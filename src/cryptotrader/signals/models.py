"""信号组件共享的不可变领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from cryptotrader.models import DataSnapshot
    from cryptotrader.pair import MarketType, Pair

SignalDirection = Literal["long", "short", "neutral"]
TradingMode = Literal["paper", "live", "backtest"]


@dataclass(frozen=True, order=True)
class CandleRequirement:
    """组件需要的一个已闭合 K 线窗口。"""

    timeframe: str
    limit: int

    def __post_init__(self) -> None:
        if not self.timeframe.strip() or self.limit <= 0:
            raise ValueError("candle requirement requires timeframe and positive limit")


@dataclass(frozen=True)
class DataRequirements:
    """一个或多个信号组件合并后的数据需求。"""

    candles: tuple[CandleRequirement, ...] = ()
    onchain: bool = False
    news: bool = False
    macro: bool = False
    kronos_aux: bool = False

    @classmethod
    def merge(cls, *requirements: DataRequirements) -> DataRequirements:
        limits: dict[str, int] = {}
        timeframe_order: list[str] = []
        for requirement in requirements:
            for candle in requirement.candles:
                if candle.timeframe not in limits:
                    timeframe_order.append(candle.timeframe)
                limits[candle.timeframe] = max(limits.get(candle.timeframe, 0), candle.limit)
        return cls(
            candles=tuple(CandleRequirement(timeframe, limits[timeframe]) for timeframe in timeframe_order),
            onchain=any(item.onchain for item in requirements),
            news=any(item.news for item in requirements),
            macro=any(item.macro for item in requirements),
            kronos_aux=any(item.kronos_aux for item in requirements),
        )


@dataclass(frozen=True)
class PositionSnapshot:
    """决策周期开始时的单交易对仓位。"""

    side: Literal["long", "short", "flat"]
    amount: float
    size_ratio: float
    avg_price: float | None = None
    unrealized_pnl: float = 0.0

    @property
    def signed_amount(self) -> float:
        return {"long": self.amount, "short": -self.amount, "flat": 0.0}[self.side]

    @property
    def signed_ratio(self) -> float:
        return {"long": self.size_ratio, "short": -self.size_ratio, "flat": 0.0}[self.side]


@dataclass(frozen=True)
class SignalContext:
    """所有组件在同一个 ``as_of`` 下只读的市场上下文。"""

    pair: Pair
    as_of: datetime
    mode: TradingMode
    exchange_id: str
    market_type: MarketType
    equity: float
    current_price: float
    atr: float
    current_position: PositionSnapshot
    snapshots: Mapping[str, DataSnapshot]
    portfolio: Mapping[str, Any]


@dataclass(frozen=True)
class ComponentSignal:
    """组件对市场方向的唯一标准输出。"""

    component_id: str
    direction: SignalDirection
    confidence: float
    reasoning: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("component_id must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
