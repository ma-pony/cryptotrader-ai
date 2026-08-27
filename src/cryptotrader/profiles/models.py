"""运行时唯一的全局 SignalProfile。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection


@dataclass(frozen=True)
class ComponentWeight:
    component_id: str
    enabled: bool
    weight: float

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("component_id must not be empty")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")


@dataclass(frozen=True)
class SignalProfile:
    revision: int
    components: tuple[ComponentWeight, ...]
    neutral_threshold: float
    max_target_ratio: float
    atr_stop_multiplier: float
    reward_ratio: float
    hitl_required: bool


def validate_signal_profile(profile: SignalProfile, installed_component_ids: Collection[str]) -> SignalProfile:
    """验证完整 Profile; 运行时禁止归一化或静默修复。"""

    component_ids = [item.component_id for item in profile.components]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("duplicate component_id in signal profile")

    installed = set(installed_component_ids)
    missing = sorted(set(component_ids) - installed)
    if missing:
        raise ValueError(f"uninstalled components: {', '.join(missing)}")

    enabled = tuple(item for item in profile.components if item.enabled)
    if not enabled:
        raise ValueError("at least one component must be enabled")
    total_weight = math.fsum(item.weight for item in enabled)
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"enabled component weights must sum to 1.0, got {total_weight}")

    if not 0.0 <= profile.neutral_threshold < 1.0:
        raise ValueError("neutral_threshold must be in [0, 1)")
    if not 0.0 < profile.max_target_ratio <= 1.0:
        raise ValueError("max_target_ratio must be in (0, 1]")
    if profile.atr_stop_multiplier <= 0.0:
        raise ValueError("atr_stop_multiplier must be positive")
    if profile.reward_ratio <= 0.0:
        raise ValueError("reward_ratio must be positive")

    return profile
