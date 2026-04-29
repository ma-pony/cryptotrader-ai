"""Position size risk checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.models import CheckResult, TradeVerdict

if TYPE_CHECKING:
    from cryptotrader.config import PositionConfig


class MaxPositionSize:
    name = "max_position_size"

    def __init__(self, config: PositionConfig) -> None:
        self._max_pct = config.max_single_pct

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        # position_scale is clamped to [0, 1] by TradeVerdict.__post_init__,
        # so target = max_pct * scale can never exceed max_pct.
        # The execution layer computes delta (target - existing) for same-direction
        # orders, so we don't need to add existing + new here.
        # MaxTotalExposure handles the overall exposure limit.
        return CheckResult(passed=True)


class MaxTotalExposure:
    name = "max_total_exposure"

    def __init__(self, config: PositionConfig) -> None:
        self._max_pct = config.max_total_exposure_pct
        self._max_single_pct = config.max_single_pct

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        if verdict.action in ("hold", "close"):
            return CheckResult(passed=True)

        total = portfolio.get("total_value", 0)
        if total <= 0:
            return CheckResult(passed=True)
        positions = portfolio.get("positions", {})
        existing = 0.0
        for v in positions.values():
            if isinstance(v, dict):
                amount = v.get("amount", 0) or 0
                avg_price = v.get("avg_price", 0) or 0
                existing += abs(amount * avg_price)
            else:
                try:
                    existing += abs(float(v))
                except (TypeError, ValueError):
                    continue
        existing_pct = min(existing / total, self._max_pct)

        projected_new = self._max_single_pct * verdict.position_scale
        projected_total = existing_pct + projected_new

        if projected_total > self._max_pct:
            remaining = self._max_pct - existing_pct
            if remaining > 0.01 and self._max_single_pct > 0:
                # PROD-I3: Propose a scale via CheckResult.scale_adjustment instead
                # of mutating verdict.position_scale in place. risk_check aggregates
                # all proposals and emits the final scale via the node return delta.
                proposed = max(0.0, min(1.0, remaining / self._max_single_pct))
                return CheckResult(
                    passed=True,
                    scale_adjustment=proposed,
                    reason=(
                        f"Scale clamped {projected_new:.2%} -> {remaining:.2%} "
                        f"to fit exposure limit {self._max_pct:.2%}"
                    ),
                )
            return CheckResult(
                passed=False,
                reason=(
                    f"No remaining exposure budget: existing {existing_pct:.2%} already at max {self._max_pct:.2%}"
                ),
            )
        return CheckResult(passed=True)
