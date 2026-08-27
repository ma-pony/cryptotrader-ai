"""Available-margin pre-flight risk check (spec 021 D1).

Production observation 2026-05-11: OKX rejected a SHORT DOGE order with
``sCode=51008 Insufficient USDT margin in account`` even though the in-process
``MaxTotalExposure`` check passed. Root cause: that check uses
``portfolio['total_value']`` (= cash + position value) which inflates apparent
buying power — it does NOT account for the fact that existing perp positions
have already *locked* USDT as initial margin, so the **free** cash available
to open new positions is much smaller.

This check fixes the gap by comparing projected new-position initial margin
against ``portfolio['free_cash']`` (the OKX-reported free balance), with a
small safety buffer for fees + funding accrual.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cryptotrader.risk.models import RiskCheckResult

if TYPE_CHECKING:
    from cryptotrader.config import PositionConfig
    from cryptotrader.risk.models import RiskRequest


logger = logging.getLogger(__name__)


class AvailableMargin:
    """Reject when projected new-position margin > free USDT * safety_buffer."""

    name = "available_margin"

    def __init__(
        self,
        config: PositionConfig,
        leverage: int = 1,
        safety_buffer: float = 0.95,
    ) -> None:
        self._max_single_pct = config.max_single_pct
        self._leverage = max(1, int(leverage))
        # Keep 5% headroom for taker fees, funding, and slippage between gate
        # eval and matching engine.
        self._safety_buffer = safety_buffer

    async def evaluate(self, request: RiskRequest, portfolio: dict) -> RiskCheckResult:
        target = request.target
        current = request.context.current_position
        if target.side == "flat":
            return RiskCheckResult(passed=True)
        same_direction = current.side == target.side
        current_ratio = current.size_ratio if same_direction else 0.0
        incremental_ratio = max(0.0, target.size_ratio - current_ratio)
        if incremental_ratio == 0.0:
            return RiskCheckResult(passed=True)

        # `free_cash` is preferred (OKX-reported free USDT). Fall back to `cash`
        # when the portfolio shape predates the field (paper / older code).
        free_cash = portfolio.get("free_cash")
        if free_cash is None:
            free_cash = portfolio.get("cash", 0.0)
        free_cash = float(free_cash or 0.0)
        if free_cash <= 0:
            return RiskCheckResult(
                passed=False,
                reason="No free USDT margin available on exchange.",
            )

        total = float(portfolio.get("total_value", 0.0) or 0.0)
        if total <= 0:
            # Without equity we cannot translate position_scale to notional —
            # be conservative and pass through (other checks will catch it).
            return RiskCheckResult(passed=True)

        incremental_notional = total * self._max_single_pct * incremental_ratio
        required_margin = incremental_notional / self._leverage
        usable = free_cash * self._safety_buffer

        if required_margin > usable:
            allowed_increment = usable * self._leverage / (total * self._max_single_pct)
            proposed = min(target.size_ratio, current_ratio + allowed_increment)
            if proposed > current_ratio + 0.01:
                return RiskCheckResult(
                    passed=True,
                    size_ratio_cap=proposed,
                    reason=(
                        f"Target ratio clamped to fit free margin: "
                        f"required {required_margin:.2f} USDT > "
                        f"usable {usable:.2f} (free={free_cash:.2f}, "
                        f"buffer={self._safety_buffer:.0%}). "
                        f"New target ratio={proposed:.2%}."
                    ),
                )
            return RiskCheckResult(
                passed=False,
                reason=(
                    f"Insufficient free USDT margin: required {required_margin:.2f} > "
                    f"usable {usable:.2f} (free={free_cash:.2f}, lev={self._leverage}x)."
                ),
            )

        return RiskCheckResult(passed=True)
