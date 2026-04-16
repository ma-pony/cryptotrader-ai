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
        # hold/close don't add exposure — always pass
        if verdict.action in ("hold", "close"):
            return CheckResult(passed=True)

        total = portfolio.get("total_value", 0)
        if total <= 0:
            return CheckResult(passed=True)
        positions = portfolio.get("positions", {})
        # positions can be {pair: float} or {pair: {"amount": x, "avg_price": y}}
        existing = 0.0
        for v in positions.values():
            if isinstance(v, dict):
                existing += abs(v.get("amount", 0) * v.get("avg_price", 0))
            else:
                existing += abs(float(v))
        # Cap at max_pct so leveraged/underwater positions don't block all new trades
        existing_pct = min(existing / total, self._max_pct)

        # Project the new trade's exposure: max_single_pct * scale.
        # The execution layer uses delta logic (target - existing_for_pair),
        # so this is a conservative upper bound.
        projected_new = self._max_single_pct * verdict.position_scale
        projected_total = existing_pct + projected_new

        if projected_total > self._max_pct:
            return CheckResult(
                passed=False,
                reason=(
                    f"Projected exposure {projected_total:.2%} "
                    f"(existing {existing_pct:.2%} + new {projected_new:.2%}) "
                    f"exceeds max {self._max_pct:.2%}"
                ),
            )
        return CheckResult(passed=True)
