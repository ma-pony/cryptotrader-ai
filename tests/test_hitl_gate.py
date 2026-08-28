"""旧周期与新资金池分别遵守各自的 HITL 开关。"""

from cryptotrader.execution.models import ExecutionBook
from cryptotrader.hitl.gate import requires_approval
from tests.factories.signal_fusion import profile


def test_profile_can_require_approval_for_live_and_paper():
    assert requires_approval(profile(hitl=True), "live") is True
    assert requires_approval(profile(hitl=True), "paper") is True


def test_profile_can_disable_approval():
    assert requires_approval(profile(hitl=False), "live") is False


def test_backtest_never_waits_for_human_approval():
    assert requires_approval(profile(hitl=True), "backtest") is False


def test_book_hitl_gate_uses_the_frozen_execution_book_setting():
    from cryptotrader.hitl.gate import requires_book_approval

    live = ExecutionBook("live", "Live", "real", True, True, ())
    simulation = ExecutionBook("simulation", "Simulation", "simulated", True, False, ())

    assert requires_book_approval(live) is True
    assert requires_book_approval(simulation) is False
