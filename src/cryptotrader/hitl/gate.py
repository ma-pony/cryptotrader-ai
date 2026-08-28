"""TradingCycle 的可配置人工审批开关。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.execution.models import ExecutionBook
    from cryptotrader.profiles.models import SignalProfile
    from cryptotrader.signals.models import TradingMode


def requires_approval(profile: SignalProfile, mode: TradingMode) -> bool:
    """回测必须无人值守。其余模式服从当前周期冻结的 Profile。"""
    return mode != "backtest" and profile.hitl_required


def requires_book_approval(book: ExecutionBook) -> bool:
    """新主链只依据当前周期冻结的资金池配置决定是否等待审批。"""
    from cryptotrader.execution.models import ExecutionBook

    if not isinstance(book, ExecutionBook):
        raise ValueError("book must be an ExecutionBook")
    return book.hitl_required
