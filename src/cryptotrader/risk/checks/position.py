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
        if verdict.position_scale > 1.0:
            return CheckResult(passed=False, reason=f"Position scale {verdict.position_scale:.2%} exceeds 100%")
        # New trade size = max_pct * scale
        new_trade_pct = self._max_pct * verdict.position_scale
        # Check existing position in this pair
        pair = portfolio.get("pair", "")
        positions = portfolio.get("positions", {})
        # Handle both {pair: float} and {pair: {"amount": x, "avg_price": y}} formats
        raw_pos = positions.get(pair, 0)
        if isinstance(raw_pos, dict):
            existing = abs(raw_pos.get("amount", 0) * raw_pos.get("avg_price", 0))
        else:
            existing = abs(float(raw_pos))
        existing_pct = existing / total if total > 0 else 0
        combined_pct = existing_pct + new_trade_pct
        if combined_pct > self._max_pct:
            return CheckResult(
                passed=False,
                reason=f"Combined position {combined_pct:.2%} (existing {existing_pct:.2%} + new {new_trade_pct:.2%}) exceeds max {self._max_pct:.2%}",
            )
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
        # positions can be {pair: float} or {pair: {"amount": x, "avg_price": y}}
        raw = 0.0
        for v in positions.values():
            if isinstance(v, dict):
                raw += abs(v.get("amount", 0) * v.get("avg_price", 0))
            else:
                raw += abs(float(v))
        exposure = raw / total
        if exposure > self._max_pct:
            return CheckResult(passed=False, reason=f"Total exposure {exposure:.2%} exceeds max {self._max_pct:.2%}")
        return CheckResult(passed=True)
