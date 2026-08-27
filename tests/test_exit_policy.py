"""统一 ATR ExitPolicy 契约。"""

from __future__ import annotations

import pytest

from tests.factories.signal_fusion import context, profile, signal


def test_atr_exit_policy_builds_short_prices():
    from cryptotrader.decision.exit_policy import AtrExitPolicy
    from cryptotrader.decision.models import TargetPosition
    from cryptotrader.signals.fusion import FusedSignal

    signals = (signal("kronos", "short", 0.6),)
    fused = FusedSignal(-0.6, (), "short evidence")

    plan = AtrExitPolicy().build_plan(
        context=context(price=100.0, atr=5.0),
        target=TargetPosition("short", 0.5),
        signals=signals,
        fused=fused,
        profile=profile(atr_stop_multiplier=2.0, reward_ratio=3.0),
    )

    assert plan.stop_loss == pytest.approx(110.0)
    assert plan.take_profit == pytest.approx(70.0)
    assert plan.component_signals is signals
    assert plan.fused_signal is fused


def test_atr_exit_policy_builds_long_prices():
    from cryptotrader.decision.exit_policy import AtrExitPolicy
    from cryptotrader.decision.models import TargetPosition
    from cryptotrader.signals.fusion import FusedSignal

    plan = AtrExitPolicy().build_plan(
        context=context(price=100.0, atr=4.0),
        target=TargetPosition("long", 0.25),
        signals=(),
        fused=FusedSignal(0.4, (), "long evidence"),
        profile=profile(atr_stop_multiplier=1.5, reward_ratio=2.0),
    )

    assert plan.stop_loss == pytest.approx(94.0)
    assert plan.take_profit == pytest.approx(112.0)


def test_flat_target_has_no_exit_prices():
    from cryptotrader.decision.exit_policy import AtrExitPolicy
    from cryptotrader.decision.models import TargetPosition
    from cryptotrader.signals.fusion import FusedSignal

    plan = AtrExitPolicy().build_plan(
        context=context(),
        target=TargetPosition("flat", 0.0),
        signals=(),
        fused=FusedSignal(0.0, (), "neutral"),
        profile=profile(),
    )

    assert plan.stop_loss is None
    assert plan.take_profit is None
