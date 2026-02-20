"""Exchange health risk check."""

from __future__ import annotations

from cryptotrader.config import ExchangeCheckConfig
from cryptotrader.models import CheckResult, TradeVerdict


class ExchangeHealthCheck:
    name = "exchange_health"

    def __init__(self, config: ExchangeCheckConfig) -> None:
        self._max_latency = config.max_api_latency_ms

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        latency = portfolio.get("api_latency_ms", 0)
        if latency > self._max_latency:
            return CheckResult(passed=False, reason=f"API latency {latency}ms exceeds max {self._max_latency}ms")
        return CheckResult(passed=True)
