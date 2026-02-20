"""Loss limit risk checks."""

from __future__ import annotations

from cryptotrader.config import LossConfig
from cryptotrader.models import CheckResult, TradeVerdict


class DailyLossLimit:
    name = "daily_loss_limit"

    def __init__(self, config: LossConfig) -> None:
        self._max_pct = config.max_daily_loss_pct
        self.circuit_breaker = False

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        if self.circuit_breaker:
            return CheckResult(passed=False, reason="Circuit breaker active")
        total = portfolio.get("total_value", 0)
        if total <= 0:
            return CheckResult(passed=False, reason="Invalid portfolio value")
        daily_pnl = portfolio.get("daily_pnl", 0)
        loss_pct = abs(daily_pnl) / total if daily_pnl < 0 else 0
        if loss_pct > self._max_pct:
            self.circuit_breaker = True
            return CheckResult(passed=False, reason=f"Daily loss {loss_pct:.2%} exceeds max {self._max_pct:.2%}")
        return CheckResult(passed=True)


class DrawdownLimit:
    name = "drawdown_limit"

    def __init__(self, config: LossConfig) -> None:
        self._max_pct = config.max_drawdown_pct

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        drawdown = portfolio.get("drawdown", 0)
        if drawdown > self._max_pct:
            return CheckResult(passed=False, reason=f"Drawdown {drawdown:.2%} exceeds max {self._max_pct:.2%}")
        return CheckResult(passed=True)
