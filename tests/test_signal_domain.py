"""可插拔信号融合的基础领域契约。"""

from __future__ import annotations

from dataclasses import fields

import pytest


def test_component_signal_rejects_confidence_outside_unit_interval():
    from cryptotrader.signals.models import ComponentSignal

    with pytest.raises(ValueError, match="confidence"):
        ComponentSignal("kronos", "long", 1.01, "超出范围")


def test_component_signal_requires_non_empty_component_id():
    from cryptotrader.signals.models import ComponentSignal

    with pytest.raises(ValueError, match="component_id"):
        ComponentSignal("", "neutral", 0.0, "无组件")


def test_data_requirements_merge_uses_largest_limit_per_timeframe():
    from cryptotrader.signals.models import CandleRequirement, DataRequirements

    merged = DataRequirements.merge(
        DataRequirements(candles=(CandleRequirement("1h", 100),), news=True, kronos_aux=True),
        DataRequirements(
            candles=(CandleRequirement("1h", 200), CandleRequirement("4h", 512)),
            onchain=True,
        ),
    )

    assert merged.candles == (
        CandleRequirement("1h", 200),
        CandleRequirement("4h", 512),
    )
    assert merged.onchain is True
    assert merged.news is True
    assert merged.macro is False
    assert merged.kronos_aux is True


@pytest.mark.parametrize(
    ("timeframe", "limit"),
    [("", 100), ("1h", 0), ("1h", -1)],
)
def test_candle_requirement_rejects_invalid_input(timeframe: str, limit: int):
    from cryptotrader.signals.models import CandleRequirement

    with pytest.raises(ValueError, match="candle requirement"):
        CandleRequirement(timeframe, limit)


def test_target_position_flat_requires_zero_size():
    from cryptotrader.decision.models import TargetPosition

    with pytest.raises(ValueError, match="flat"):
        TargetPosition(side="flat", size_ratio=0.2)


@pytest.mark.parametrize(
    ("side", "size_ratio"),
    [("long", 0.0), ("short", 0.0), ("long", 1.01), ("short", -0.1)],
)
def test_target_position_rejects_invalid_non_flat_size(side: str, size_ratio: float):
    from cryptotrader.decision.models import TargetPosition

    with pytest.raises(ValueError, match=r"size_ratio|non-flat"):
        TargetPosition(side=side, size_ratio=size_ratio)


def test_target_position_exposes_signed_ratio():
    from cryptotrader.decision.models import TargetPosition

    assert TargetPosition("long", 0.4).signed_ratio == 0.4
    assert TargetPosition("short", 0.4).signed_ratio == -0.4
    assert TargetPosition("flat", 0.0).signed_ratio == 0.0


def test_position_snapshot_exposes_signed_amount_and_ratio():
    from cryptotrader.signals.models import PositionSnapshot

    short = PositionSnapshot("short", amount=2.5, size_ratio=0.3, avg_price=100.0)

    assert short.signed_amount == -2.5
    assert short.signed_ratio == -0.3


def test_cycle_request_exposes_only_canonical_admission_fields():
    from cryptotrader.decision.models import CycleRequest
    from cryptotrader.pair import Pair

    request = CycleRequest(Pair.parse("BTC/USDT:USDT"))

    assert request.pair.canonical() == "BTC/USDT:USDT"
    assert [field.name for field in fields(request)] == [
        "pair",
        "mode",
        "origin",
        "decision_id",
        "confirmed_book_ids",
    ]
    assert (request.mode, request.origin, request.decision_id, request.confirmed_book_ids) == (
        "trading",
        "manual",
        None,
        None,
    )
