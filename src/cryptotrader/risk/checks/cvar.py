"""CVaR risk check."""

from __future__ import annotations

import numpy as np

from cryptotrader.config import LossConfig
from cryptotrader.models import CheckResult, TradeVerdict


class CVaRCheck:
    name = "cvar"

    def __init__(self, config: LossConfig) -> None:
        self._max_cvar = config.max_cvar_95

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        returns = portfolio.get("returns_60d", [])
        if len(returns) < 2:
            return CheckResult(passed=True)
        var_95 = np.percentile(returns, 5)
        cvar = float(np.mean([r for r in returns if r <= var_95]))
        if abs(cvar) > self._max_cvar:
            return CheckResult(passed=False, reason=f"CVaR {cvar:.4f} exceeds max {self._max_cvar}")
        return CheckResult(passed=True)
