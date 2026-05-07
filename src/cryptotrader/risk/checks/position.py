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
    will scale-clamp the verdict; margin cap reject is hard-fail (no
    auto-clamp because clamping notional doesn't necessarily save margin).
    """

    name = "max_total_exposure"

    def __init__(self, config: PositionConfig, leverage: int = 1) -> None:
        self._max_notional_pct = config.max_total_exposure_pct
        self._max_margin_pct = getattr(config, "max_margin_used_pct", 0.40)
        self._max_single_pct = config.max_single_pct
        self._leverage = max(1, int(leverage))

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        if verdict.action in ("hold", "close"):
            return CheckResult(passed=True)

        total = portfolio.get("total_value", 0)
        if total <= 0:
            return CheckResult(passed=True)

        positions = portfolio.get("positions", {})
        existing_notional = 0.0
        existing_margin = 0.0
        for v in positions.values():
            if isinstance(v, dict):
                amount = v.get("amount", 0) or 0
                avg_price = v.get("avg_price", 0) or 0
                notional = abs(amount * avg_price)
                existing_notional += notional
                # Spot has leverage=1 effectively (margin == notional).
                # Treat all non-spot as the configured leverage.
                market = (v.get("market_type") or "spot").lower()
                lev = 1 if market == "spot" else self._leverage
                existing_margin += notional / lev
            else:
                try:
                    existing_notional += abs(float(v))
                    existing_margin += abs(float(v))
                except (TypeError, ValueError):
                    continue

        existing_notional_pct = existing_notional / total
        existing_margin_pct = existing_margin / total

        # 1) Margin cap is hard — adding ANY new position (≥ some scale * single)
        # consumes more margin. If we're already at max, no scale clamping helps.
        if existing_margin_pct >= self._max_margin_pct:
            return CheckResult(
                passed=False,
                reason=(
                    f"No remaining margin budget: existing {existing_margin_pct:.2%} already at max"
                    f" {self._max_margin_pct:.2%}"
                ),
            )

        # 2) Notional cap with auto-clamp. Same logic as before but uses
        # ``max_total_exposure_pct`` (renamed semantically to "notional cap").
        # Cap existing_notional_pct at the max so projected_total math stays sane.
        existing_notional_pct = min(existing_notional_pct, self._max_notional_pct)
        projected_new = self._max_single_pct * verdict.position_scale
        projected_total = existing_notional_pct + projected_new

        if projected_total > self._max_notional_pct:
            remaining = self._max_notional_pct - existing_notional_pct
            if remaining > 0.01 and self._max_single_pct > 0:
                # PROD-I3: Propose a scale via CheckResult.scale_adjustment instead
                # of mutating verdict.position_scale in place.
                proposed = max(0.0, min(1.0, remaining / self._max_single_pct))
                return CheckResult(
                    passed=True,
                    scale_adjustment=proposed,
                    reason=(
                        f"Scale clamped {projected_new:.2%} -> {remaining:.2%} "
                        f"to fit notional limit {self._max_notional_pct:.2%}"
                    ),
                )
            return CheckResult(
                passed=False,
                reason=(
                    f"No remaining notional budget: existing {existing_notional_pct:.2%} already at max"
                    f" {self._max_notional_pct:.2%}"
                ),
            )
        return CheckResult(passed=True)
