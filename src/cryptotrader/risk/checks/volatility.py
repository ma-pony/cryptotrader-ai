"""Volatility risk checks."""

from __future__ import annotations

from cryptotrader.config import VolatilityConfig
from cryptotrader.models import CheckResult, TradeVerdict


class VolatilityGate:
    name = "volatility_gate"

    def __init__(self, config: VolatilityConfig) -> None:
        self._threshold = config.flash_crash_threshold

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        prices = portfolio.get("recent_prices", [])
        if len(prices) < 2:
            return CheckResult(passed=True)
        # Use recent window (last 10 candles) for flash crash detection,
        # not global peak which triggers false positives during normal downtrends
        lookback = min(10, len(prices))
        recent = prices[-lookback:]
        peak = max(recent)
        current = prices[-1]
        drop = (peak - current) / peak if peak > 0 else 0
        if drop > self._threshold:
            return CheckResult(passed=False, reason=f"Flash crash detected: {drop:.2%} drop in last {lookback} candles")
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
