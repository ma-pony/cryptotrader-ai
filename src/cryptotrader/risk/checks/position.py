"""Position size risk checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.risk.models import RiskCheckResult

if TYPE_CHECKING:
    from cryptotrader.config import PositionConfig
    from cryptotrader.risk.models import RiskRequest


def _existing_exposure(positions: dict, current_pair: str, leverage: int) -> tuple[float, float]:
    notional_total = 0.0
    margin_total = 0.0
    for pair_key, value in positions.items():
        if pair_key == current_pair:
            continue
        if isinstance(value, dict):
            amount = value.get("amount", 0) or 0
            avg_price = value.get("avg_price", 0) or 0
            notional = abs(amount * avg_price)
            market_type = (value.get("market_type") or "").lower()
            if not market_type:
                market_type = "swap" if isinstance(pair_key, str) and ":" in pair_key else "spot"
            position_leverage = 1 if market_type == "spot" else leverage
        else:
            try:
                notional = abs(float(value))
            except (TypeError, ValueError):
                continue
            position_leverage = 1
        notional_total += notional
        margin_total += notional / position_leverage
    return notional_total, margin_total


class MaxPositionSize:
    name = "max_position_size"

    def __init__(self, config: PositionConfig) -> None:
        self._max_pct = config.max_single_pct

    async def evaluate(self, request: RiskRequest, portfolio: dict) -> RiskCheckResult:
        return RiskCheckResult(passed=True)


class MaxTotalExposure:
    """Dual-cap exposure check (2026-05-07).

    Two independent caps so leverage actually buys capital efficiency without
    breaking risk semantics:

    - **Notional cap** (``max_total_exposure_pct``): sum |amount × avg_price|
      / equity. This is *price exposure* — the same regardless of leverage.
      Caps drawdown sensitivity to a market move.

    - **Margin cap** (``max_margin_used_pct``): for derivative positions,
      margin = notional / leverage. Sums across all positions. Spot positions
      contribute their full notional (margin == notional in spot). Caps
      capital lockup and preserves a free-margin buffer for stop-loss
      execution + funding payments + adverse moves.

    Either cap rejecting → REJECT. Within remaining notional budget the check
    will cap the target ratio; margin cap reject is hard-fail (no
    auto-clamp because clamping notional doesn't necessarily save margin).
    """

    name = "max_total_exposure"

    def __init__(self, config: PositionConfig, leverage: int = 1) -> None:
        self._max_notional_pct = config.max_total_exposure_pct
        self._max_margin_pct = getattr(config, "max_margin_used_pct", 0.40)
        self._max_single_pct = config.max_single_pct
        self._leverage = max(1, int(leverage))

    async def evaluate(self, request: RiskRequest, portfolio: dict) -> RiskCheckResult:
        if request.reduces_exposure:
            return RiskCheckResult(passed=True)

        total = portfolio.get("total_value", 0)
        if total <= 0:
            return RiskCheckResult(passed=True)

        current_pair = request.context.pair.canonical()
        existing_notional, existing_margin = _existing_exposure(
            portfolio.get("positions", {}),
            current_pair,
            self._leverage,
        )

        existing_notional_pct = existing_notional / total
        existing_margin_pct = existing_margin / total

        # 1) Margin cap is hard — adding ANY new position (≥ some scale * single)
        # consumes more margin. If we're already at max, no scale clamping helps.
        if existing_margin_pct >= self._max_margin_pct:
            return RiskCheckResult(
                passed=False,
                reason=(
                    f"No remaining margin budget: existing {existing_margin_pct:.2%} already at max"
                    f" {self._max_margin_pct:.2%}"
                ),
            )

        target_notional_pct = self._max_single_pct * request.target.size_ratio
        target_margin_pct = target_notional_pct / (1 if request.context.market_type == "spot" else self._leverage)
        if existing_margin_pct + target_margin_pct > self._max_margin_pct:
            return RiskCheckResult(
                passed=False,
                reason=(
                    f"Projected margin {existing_margin_pct + target_margin_pct:.2%} exceeds max"
                    f" {self._max_margin_pct:.2%}"
                ),
            )

        existing_notional_pct = min(existing_notional_pct, self._max_notional_pct)
        projected_total = existing_notional_pct + target_notional_pct

        if projected_total > self._max_notional_pct:
            remaining = self._max_notional_pct - existing_notional_pct
            if remaining > 0.01 and self._max_single_pct > 0:
                proposed = max(0.0, min(1.0, remaining / self._max_single_pct))
                return RiskCheckResult(
                    passed=True,
                    size_ratio_cap=proposed,
                    reason=(
                        f"Target ratio clamped {request.target.size_ratio:.2%} -> {proposed:.2%} "
                        f"to fit notional limit {self._max_notional_pct:.2%}"
                    ),
                )
            return RiskCheckResult(
                passed=False,
                reason=(
                    f"No remaining notional budget: existing {existing_notional_pct:.2%} already at max"
                    f" {self._max_notional_pct:.2%}"
                ),
            )
        return RiskCheckResult(passed=True)
