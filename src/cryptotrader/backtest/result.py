"""Backtest result dataclass and statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class BacktestResult:
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "total_return": f"{self.total_return:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "win_rate": f"{self.win_rate:.2%}",
            "num_trades": len(self.trades),
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"summary": self.summary(), "trades": self.trades,
                        "equity_curve": self.equity_curve}, f, indent=2, default=str)
