"""HITL 配置只控制新周期。回测固定关闭。"""

from cryptotrader.hitl.gate import requires_approval
from tests.factories.signal_fusion import profile


def test_profile_can_require_approval_for_live_and_paper():
    assert requires_approval(profile(hitl=True), "live") is True
    assert requires_approval(profile(hitl=True), "paper") is True


def test_profile_can_disable_approval():
    assert requires_approval(profile(hitl=False), "live") is False


def test_backtest_never_waits_for_human_approval():
    assert requires_approval(profile(hitl=True), "backtest") is False
