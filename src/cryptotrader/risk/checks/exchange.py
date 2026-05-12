"""Exchange health risk check."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.models import CheckResult, TradeVerdict

if TYPE_CHECKING:
    from cryptotrader.config import ExchangeCheckConfig


class ExchangeHealthCheck:
    name = "exchange_health"

    def __init__(self, config: ExchangeCheckConfig) -> None:
        self._max_latency = config.max_api_latency_ms

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        # spec 021 H3 (option 2): trade endpoint cooldown. When an earlier
        # pair in the same cycle string exhausted retries on OKX
        # sCode=50013, the exchange module stamps a 5-min unavailability
        # window. Reject actionable verdicts cheaply so we don't waste
        # ~24s per pair re-probing a known-down endpoint. Hold verdicts
        # don't issue orders so the cooldown is irrelevant for them.
        cooldown_remaining = portfolio.get("trade_unavailable_remaining_s", 0.0)
        action = getattr(verdict, "action", "hold")
        if cooldown_remaining > 0 and action not in ("hold", "close"):
            return CheckResult(
                passed=False,
                reason=f"Exchange trade endpoint unavailable, cooldown {cooldown_remaining:.0f}s remaining",
            )

        latency = portfolio.get("api_latency_ms", 0)
        if latency > self._max_latency:
            return CheckResult(passed=False, reason=f"API latency {latency}ms exceeds max {self._max_latency}ms")
        return CheckResult(passed=True)
