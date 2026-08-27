"""确定性信号加权融合契约。"""

from __future__ import annotations

import pytest

from cryptotrader.profiles.models import ComponentWeight
from cryptotrader.signals.models import ComponentSignal


def test_weighted_fusion_preserves_configured_contributions():
    from cryptotrader.signals.fusion import WeightedSignalFusion

    signals = (
        ComponentSignal("kronos", "long", 0.8, "k"),
        ComponentSignal("llm_committee", "short", 0.3, "l"),
    )
    weights = (
        ComponentWeight("kronos", True, 0.6),
        ComponentWeight("llm_committee", True, 0.4),
    )

    fused = WeightedSignalFusion().fuse(signals, weights)

    assert fused.score == pytest.approx(0.36)
    assert [item.signed_score for item in fused.contributions] == pytest.approx([0.8, -0.3])
    assert [item.weighted_score for item in fused.contributions] == pytest.approx([0.48, -0.12])
    assert "kronos: +0.4800" in fused.reasoning
    assert "llm_committee: -0.1200" in fused.reasoning


def test_fusion_rejects_missing_enabled_signal():
    from cryptotrader.signals.fusion import WeightedSignalFusion

    with pytest.raises(ValueError, match="llm_committee"):
        WeightedSignalFusion().fuse(
            (ComponentSignal("kronos", "long", 0.8, "k"),),
            (
                ComponentWeight("kronos", True, 0.6),
                ComponentWeight("llm_committee", True, 0.4),
            ),
        )


def test_fusion_ignores_disabled_component_weight():
    from cryptotrader.signals.fusion import WeightedSignalFusion

    fused = WeightedSignalFusion().fuse(
        (ComponentSignal("kronos", "neutral", 0.9, "neutral"),),
        (
            ComponentWeight("kronos", True, 1.0),
            ComponentWeight("llm_committee", False, 0.0),
        ),
    )

    assert fused.score == 0.0
    assert [item.component_id for item in fused.contributions] == ["kronos"]


def test_fusion_rejects_duplicate_component_signals():
    from cryptotrader.signals.fusion import WeightedSignalFusion

    with pytest.raises(ValueError, match="duplicate"):
        WeightedSignalFusion().fuse(
            (
                ComponentSignal("kronos", "long", 0.8, "first"),
                ComponentSignal("kronos", "short", 0.2, "second"),
            ),
            (ComponentWeight("kronos", True, 1.0),),
        )
