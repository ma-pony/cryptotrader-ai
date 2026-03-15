"""Volatility risk checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.models import CheckResult, TradeVerdict

if TYPE_CHECKING:
    from cryptotrader.config import VolatilityConfig


class VolatilityGate:
    name = "volatility_gate"

    def __init__(self, config: VolatilityConfig) -> None:
        self._threshold = config.flash_crash_threshold
        self._lookback = config.flash_crash_lookback

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        prices = portfolio.get("recent_prices", [])
        if len(prices) < 2:
            return CheckResult(passed=True)
        # Use recent window for flash crash detection,
        # not global peak which triggers false positives during normal downtrends
        lookback = min(self._lookback, len(prices))
        recent = prices[-lookback:]
        peak = max(recent)
        trough = min(recent)
        current = prices[-1]
        drop = (peak - current) / peak if peak > 0 else 0
        spike = (current - trough) / trough if trough > 0 else 0

        # Directional: only block counter-trend entries (catching falling knives / shorting spikes)
        # Going WITH the trend is safe — the gate protects against reversal entries
        # "close" is always allowed — it reduces risk, never increases it
        if drop > self._threshold and verdict.action == "long":
            return CheckResult(
                passed=False,
                reason=f"Flash crash detected: {drop:.2%} drop in last {lookback} candles (blocking long)",
            )
        if spike > self._threshold and verdict.action == "short":
            return CheckResult(
                passed=False,
                reason=f"Rapid spike detected: {spike:.2%} rise in last {lookback} candles (blocking short)",
            )
        return CheckResult(passed=True)


class FundingRateGate:
    name = "funding_rate_gate"

    def __init__(self, config: VolatilityConfig) -> None:
        self._threshold = config.funding_rate_threshold

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        rate = abs(portfolio.get("funding_rate", 0))
        if rate > self._threshold:
            return CheckResult(passed=False, reason=f"Funding rate {rate} exceeds threshold {self._threshold}")
        return CheckResult(passed=True)
