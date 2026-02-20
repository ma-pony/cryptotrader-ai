"""Technical analysis agent."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from cryptotrader.agents.base import BaseAgent
from cryptotrader.models import DataSnapshot

ROLE = (
    "You are an expert technical analyst for cryptocurrency markets. "
    "Analyze price action, indicators, and chart patterns to determine market direction."
)


def compute_indicators(ohlcv: pd.DataFrame) -> dict:
    """Compute technical indicators from OHLCV data."""
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    indicators: dict = {}
    indicators["rsi"] = round(float(ta.rsi(close, length=14).iloc[-1]), 2)
    macd = ta.macd(close)
    indicators["macd"] = {
        "macd": round(float(macd.iloc[-1, 0]), 4),
        "signal": round(float(macd.iloc[-1, 1]), 4),
        "histogram": round(float(macd.iloc[-1, 2]), 4),
    }
    indicators["sma_20"] = round(float(ta.sma(close, length=20).iloc[-1]), 2)
    indicators["sma_60"] = round(float(ta.sma(close, length=60).iloc[-1]), 2)
    bbands = ta.bbands(close)
    indicators["bbands"] = {
        "lower": round(float(bbands.iloc[-1, 0]), 2),
        "mid": round(float(bbands.iloc[-1, 1]), 2),
        "upper": round(float(bbands.iloc[-1, 2]), 2),
    }
    indicators["atr"] = round(float(ta.atr(high, low, close).iloc[-1]), 4)
    return indicators


class TechAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(agent_id="tech", role_description=ROLE, model=model)

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        base = super()._build_prompt(snapshot, experience)
        indicators = compute_indicators(snapshot.market.ohlcv)
        return f"Technical Indicators:\n{indicators}\n\n{base}"
