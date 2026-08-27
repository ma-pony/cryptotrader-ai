"""融合分数到目标仓位的线性映射契约。"""

from __future__ import annotations

import pytest

from tests.factories.signal_fusion import profile


@pytest.mark.parametrize("score", [-0.2, 0.0, 0.2])
def test_neutral_band_maps_to_flat(score: float):
    from cryptotrader.decision.engine import DecisionEngine
    from cryptotrader.decision.models import TargetPosition
    from cryptotrader.signals.fusion import FusedSignal

    target = DecisionEngine().target_for(FusedSignal(score=score, contributions=(), reasoning=""), profile())

    assert target == TargetPosition("flat", 0.0)


def test_linear_long_mapping_after_neutral_band():
    from cryptotrader.decision.engine import DecisionEngine
    from cryptotrader.signals.fusion import FusedSignal

    target = DecisionEngine().target_for(FusedSignal(score=0.6, contributions=(), reasoning=""), profile())

    assert target.side == "long"
    assert target.size_ratio == pytest.approx(0.5)


def test_linear_short_mapping_respects_max_target_ratio():
    from cryptotrader.decision.engine import DecisionEngine
    from cryptotrader.signals.fusion import FusedSignal

    target = DecisionEngine().target_for(
        FusedSignal(score=-1.0, contributions=(), reasoning=""),
        profile(max_target_ratio=0.75),
    )

    assert target.side == "short"
    assert target.size_ratio == pytest.approx(0.75)
