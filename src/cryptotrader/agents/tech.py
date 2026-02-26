"""Technical analysis agent."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from cryptotrader.agents.base import BaseAgent
from cryptotrader.models import DataSnapshot

ROLE = (
    "You are an expert technical analyst for cryptocurrency markets. "
    "Analyze price action, indicators, and chart patterns to determine market direction.\n\n"
    "Focus on: trend structure (higher highs/lows vs lower), momentum (RSI divergences, MACD crossovers), "
    "volatility regime (Bollinger Band width, ATR), and key support/resistance levels relative to current price.\n"
    "When indicators conflict (e.g. RSI oversold but trend bearish), explicitly state the conflict and weight "
    "the higher-timeframe signal more heavily.\n\n"
    "Domain checklist (verify before signaling):\n"
    "- Trend alignment: Is SMA20 vs SMA60 confirming or contradicting my call? A bullish call against a bearish SMA cross needs strong justification.\n"
    "- Divergence scan: Is RSI or MACD diverging from price? A new price high with lower RSI is a warning, not confirmation.\n"
    "- Volatility regime: Is BB width expanding (breakout) or contracting (squeeze)? A squeeze means the next move will be violent — lower confidence, not higher.\n"
    "- Support/resistance proximity: Is price within 2% of a key level? If yes, the level matters more than the trend."
)


def _safe_last(series) -> float | None:
    """Return last non-NaN value from a pandas Series, or None."""
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)


def compute_indicators(ohlcv: pd.DataFrame) -> dict:
    """Compute technical indicators from OHLCV data."""
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    n = len(close)
    indicators: dict = {}

    if n >= 15:
        rsi = _safe_last(ta.rsi(close, length=14))
        indicators["rsi"] = round(rsi, 2) if rsi is not None else None
    if n >= 34:
        macd = ta.macd(close)
        if macd is not None and not macd.empty:
            indicators["macd"] = {
                "macd": round(float(macd.iloc[-1, 0]), 4) if not pd.isna(macd.iloc[-1, 0]) else 0,
                "signal": round(float(macd.iloc[-1, 1]), 4) if not pd.isna(macd.iloc[-1, 1]) else 0,
                "histogram": round(float(macd.iloc[-1, 2]), 4) if not pd.isna(macd.iloc[-1, 2]) else 0,
            }
    if n >= 20:
        sma20 = _safe_last(ta.sma(close, length=20))
        indicators["sma_20"] = round(sma20, 2) if sma20 is not None else None
    if n >= 60:
        sma60 = _safe_last(ta.sma(close, length=60))
        indicators["sma_60"] = round(sma60, 2) if sma60 is not None else None
    if n >= 20:
        bbands = ta.bbands(close)
        if bbands is not None and not bbands.empty:
            indicators["bbands"] = {
                "lower": round(float(bbands.iloc[-1, 0]), 2) if not pd.isna(bbands.iloc[-1, 0]) else None,
                "mid": round(float(bbands.iloc[-1, 1]), 2) if not pd.isna(bbands.iloc[-1, 1]) else None,
                "upper": round(float(bbands.iloc[-1, 2]), 2) if not pd.isna(bbands.iloc[-1, 2]) else None,
            }
    if n >= 14:
        atr = _safe_last(ta.atr(high, low, close))
        indicators["atr"] = round(atr, 4) if atr is not None else None

    return indicators


class TechAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(agent_id="tech", role_description=ROLE, model=model)

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        base = super()._build_prompt(snapshot, experience)
        indicators = compute_indicators(snapshot.market.ohlcv)
        return f"Technical Indicators:\n{indicators}\n\n{base}"
