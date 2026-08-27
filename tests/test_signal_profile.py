"""全局 SignalProfile 的验证契约。"""

from __future__ import annotations

import pytest


def _profile(*components, **overrides):
    from cryptotrader.profiles.models import SignalProfile

    values = {
        "revision": 1,
        "components": components,
        "neutral_threshold": 0.2,
        "max_target_ratio": 1.0,
        "atr_stop_multiplier": 2.0,
        "reward_ratio": 2.0,
        "hitl_required": False,
    }
    values.update(overrides)
    return SignalProfile(**values)


def test_profile_requires_enabled_weights_to_sum_to_one():
    from cryptotrader.profiles.models import ComponentWeight, validate_signal_profile

    profile = _profile(
        ComponentWeight("kronos", True, 0.6),
        ComponentWeight("llm_committee", True, 0.3),
    )

    with pytest.raises(ValueError, match=r"1\.0"):
        validate_signal_profile(profile, {"kronos", "llm_committee"})


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_component_weight_rejects_values_outside_unit_interval(weight: float):
    from cryptotrader.profiles.models import ComponentWeight

    with pytest.raises(ValueError, match="weight"):
        ComponentWeight("kronos", True, weight)


def test_component_weight_requires_non_empty_id():
    from cryptotrader.profiles.models import ComponentWeight

    with pytest.raises(ValueError, match="component_id"):
        ComponentWeight("", True, 1.0)


def test_profile_rejects_uninstalled_component():
    from cryptotrader.profiles.models import ComponentWeight, validate_signal_profile

    profile = _profile(ComponentWeight("missing", True, 1.0))

    with pytest.raises(ValueError, match="missing"):
        validate_signal_profile(profile, {"kronos"})


def test_profile_rejects_duplicate_component_id():
    from cryptotrader.profiles.models import ComponentWeight, validate_signal_profile

    profile = _profile(
        ComponentWeight("kronos", True, 0.5),
        ComponentWeight("kronos", True, 0.5),
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_signal_profile(profile, {"kronos"})


def test_profile_requires_at_least_one_enabled_component():
    from cryptotrader.profiles.models import ComponentWeight, validate_signal_profile

    profile = _profile(ComponentWeight("kronos", False, 0.0))

    with pytest.raises(ValueError, match="enabled"):
        validate_signal_profile(profile, {"kronos"})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"neutral_threshold": -0.1}, "neutral_threshold"),
        ({"neutral_threshold": 1.0}, "neutral_threshold"),
        ({"max_target_ratio": 0.0}, "max_target_ratio"),
        ({"max_target_ratio": 1.1}, "max_target_ratio"),
        ({"atr_stop_multiplier": 0.0}, "atr_stop_multiplier"),
        ({"reward_ratio": 0.0}, "reward_ratio"),
    ],
)
def test_profile_rejects_invalid_decision_parameters(overrides: dict, message: str):
    from cryptotrader.profiles.models import ComponentWeight, validate_signal_profile

    profile = _profile(ComponentWeight("kronos", True, 1.0), **overrides)

    with pytest.raises(ValueError, match=message):
        validate_signal_profile(profile, {"kronos"})


def test_profile_accepts_valid_installed_components():
    from cryptotrader.profiles.models import ComponentWeight, validate_signal_profile

    profile = _profile(
        ComponentWeight("kronos", True, 0.6),
        ComponentWeight("llm_committee", True, 0.4),
    )

    assert validate_signal_profile(profile, {"kronos", "llm_committee"}) is profile
