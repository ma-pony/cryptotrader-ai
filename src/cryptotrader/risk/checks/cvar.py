"""CVaR risk check."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cryptotrader.models import CheckResult, TradeVerdict

if TYPE_CHECKING:
    from cryptotrader.config import LossConfig


class CVaRCheck:
    name = "cvar"

    def __init__(self, config: LossConfig) -> None:
        self._max_cvar = config.max_cvar_95
        self._min_returns = config.cvar_min_returns

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        returns = portfolio.get("returns_60d", [])
        if len(returns) < self._min_returns:
            return CheckResult(passed=True, reason=f"Insufficient data for CVaR (need {self._min_returns}+ returns)")
        var_95 = np.percentile(returns, 5)
        cvar = float(np.mean([r for r in returns if r <= var_95]))
        if abs(cvar) > self._max_cvar:
            return CheckResult(passed=False, reason=f"CVaR {cvar:.4f} exceeds max {self._max_cvar}")
        return CheckResult(passed=True)
