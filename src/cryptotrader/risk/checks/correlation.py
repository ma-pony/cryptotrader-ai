"""Correlation risk check (placeholder)."""

from __future__ import annotations

from cryptotrader.models import CheckResult, TradeVerdict


class CorrelationCheck:
    name = "correlation"

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        return CheckResult(passed=True)
