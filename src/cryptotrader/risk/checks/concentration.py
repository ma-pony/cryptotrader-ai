"""Same-direction concentration check.

Catches the failure mode where the bot's 4 agents are all driven by one macro
factor and end up opening N synchronous same-direction positions across N
different pairs. The pair-level ``CorrelationCheck`` only knows about hard-
coded asset families (BTC/WBTC, ETH/STETH, etc.) and cannot see this kind of
"5 different coins all short because of one macro signal" concentration.

The cap is on **distinct pairs in the same direction**, not on per-pair
exposure (that's ``MaxPositionSize``) and not on total notional (that's
``MaxTotalExposure``). So adding to an existing same-direction position
does NOT consume a slot — only opening a new pair (or flipping from flat
or from the opposite side) does.

``close`` and ``hold`` are unaffected: risk reduction must never be blocked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.models import CheckResult, TradeVerdict

if TYPE_CHECKING:
    from cryptotrader.config import PositionConfig


def _amount_of(pos_data) -> float:
    """Extract the signed position amount regardless of storage shape."""
    if isinstance(pos_data, int | float):
        return float(pos_data)
    if isinstance(pos_data, dict):
        return float(pos_data.get("amount", 0) or 0)
    return 0.0


class MacroConcentrationCheck:
    name = "macro_concentration"

    def __init__(self, config: PositionConfig) -> None:
        self._max_same_direction = config.max_same_direction_positions

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        # Only directional adds count — closes / holds always pass.
        if verdict.action not in ("long", "short"):
            return CheckResult(passed=True)

        positions = portfolio.get("positions", {}) or {}
        my_pair = portfolio.get("pair", "")

        # Count DISTINCT pairs that will be in the target direction *after*
        # this trade. Three cases that consume a slot:
        #   (a) my_pair is currently flat   → new entry → +1 distinct pair
        #   (b) my_pair is opposite direction → flip → +1 distinct pair
        #   (c) my_pair is already same direction → add-to-existing → +0
        #
        # Iterate other pairs first; my_pair is handled explicitly below so
        # we do NOT double-count when adding to an existing same-direction
        # position (the historical bug that误拒了 5 笔合法加仓 between
        # 04:55 and 23:55 on 2026-05-07).
        target_pairs: set[str] = set()
        for pos_pair, pos_data in positions.items():
            if pos_pair == my_pair:
                continue
            amount = _amount_of(pos_data)
            if (verdict.action == "long" and amount > 0) or (verdict.action == "short" and amount < 0):
                target_pairs.add(pos_pair)

        # The trade always lands my_pair in the target direction
        # (long → +amount, short → −amount) regardless of starting state.
        if my_pair:
            target_pairs.add(my_pair)

        if len(target_pairs) > self._max_same_direction:
            # Existing-direction count for the operator-facing reason
            # (so the rejection log still says "Already N short positions").
            existing_same_dir = len(target_pairs - {my_pair}) if my_pair else len(target_pairs)
            return CheckResult(
                passed=False,
                reason=(
                    f"Already {existing_same_dir} {verdict.action} positions; opening {my_pair or 'this pair'} "
                    f"would make {len(target_pairs)} > max_same_direction_positions="
                    f"{self._max_same_direction} (synchronous stop-loss cascade risk)"
                ),
            )
        return CheckResult(passed=True)
