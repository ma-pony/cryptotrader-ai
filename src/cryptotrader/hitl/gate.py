"""TradingCycle 的可配置人工审批开关。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.profiles.models import SignalProfile
    from cryptotrader.signals.models import TradingMode


def requires_approval(profile: SignalProfile, mode: TradingMode) -> bool:
    """回测必须无人值守。其余模式服从当前周期冻结的 Profile。"""
    return mode != "backtest" and profile.hitl_required
