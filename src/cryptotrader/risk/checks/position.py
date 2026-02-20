"""Position size risk checks."""

from __future__ import annotations

from cryptotrader.config import PositionConfig
from cryptotrader.models import CheckResult, TradeVerdict


class MaxPositionSize:
    name = "max_position_size"

    def __init__(self, config: PositionConfig) -> None:
        self._max_pct = config.max_single_pct

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        total = portfolio.get("total_value", 0)
        if total <= 0:
            return CheckResult(passed=False, reason="Invalid portfolio value")
        position_pct = verdict.position_scale
        if position_pct > self._max_pct:
            return CheckResult(passed=False, reason=f"Position {position_pct:.2%} exceeds max {self._max_pct:.2%}")
        return CheckResult(passed=True)


class MaxTotalExposure:
    name = "max_total_exposure"

    def __init__(self, config: PositionConfig) -> None:
        self._max_pct = config.max_total_exposure_pct

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        total = portfolio.get("total_value", 0)
        if total <= 0:
            return CheckResult(passed=False, reason="Invalid portfolio value")
        positions = portfolio.get("positions", {})
        exposure = sum(positions.values()) / total
        if exposure > self._max_pct:
            return CheckResult(passed=False, reason=f"Total exposure {exposure:.2%} exceeds max {self._max_pct:.2%}")
        return CheckResult(passed=True)
