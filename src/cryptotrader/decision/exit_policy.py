"""所有组件共用的 ATR 止盈止损策略。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.configuration.parameters import DefaultMarketSourceParameters
from cryptotrader.decision.models import TargetPosition, TradePlan
from cryptotrader.signals.models import CandleRequirement

if TYPE_CHECKING:
    from cryptotrader.profiles.models import SignalProfile
    from cryptotrader.runtime_config.models import MarketDataConfig
    from cryptotrader.signals.fusion import FusedSignal
    from cryptotrader.signals.models import ComponentSignal, SignalContext


def exit_candle_requirement(market_data: MarketDataConfig) -> CandleRequirement:
    """Resolve the shared exit window separately from component and evaluation windows."""
    parameters = (
        DefaultMarketSourceParameters.model_validate(dict(market_data.parameters))
        if market_data.source_id == "default"
        else DefaultMarketSourceParameters()
    )
    return CandleRequirement(parameters.timeframe, parameters.limit)


class AtrExitPolicy:
    def build_plan(
        self,
        context: SignalContext,
        target: TargetPosition,
        signals: tuple[ComponentSignal, ...],
        fused: FusedSignal,
        profile: SignalProfile,
    ) -> TradePlan:
        stop_loss: float | None = None
        take_profit: float | None = None
        if target.side != "flat":
            risk_distance = context.atr * profile.atr_stop_multiplier
            direction = 1.0 if target.side == "long" else -1.0
            stop_loss = context.current_price - direction * risk_distance
            take_profit = context.current_price + direction * risk_distance * profile.reward_ratio
        return TradePlan(
            target=target,
            stop_loss=stop_loss,
            take_profit=take_profit,
            component_signals=signals,
            fused_signal=fused,
        )
