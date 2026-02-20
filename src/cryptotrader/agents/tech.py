"""Technical analysis agent — answers: trend or range? strength? reversal signals?"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from cryptotrader.agents.base import BaseAgent
from cryptotrader.models import DataSnapshot

ROLE = """You are a technical analyst specializing in crypto market regime detection.

Your job is NOT to predict direction. Your job is to answer THREE questions:
1. REGIME: Is the market trending or ranging? (trending_up / trending_down / ranging)
2. STRENGTH: How strong is the current regime? (0.0 = no conviction, 1.0 = extreme)
3. REVERSAL SIGNALS: Are there any signs the current regime is about to change?

Decision framework:
- ADX > 25 + price above SMA60 = trending_up
- ADX > 25 + price below SMA60 = trending_down
- ADX < 20 = ranging (low conviction, prefer hold)
- RSI < 30 in downtrend = potential oversold bounce (reversal signal)
- RSI > 70 in uptrend = potential overbought pullback (reversal signal)
- MACD histogram divergence from price = early reversal warning
- Price near Bollinger Band extremes in ranging market = mean reversion opportunity

You must respond with JSON:
{
  "regime": "trending_up|trending_down|ranging",
  "strength": 0.0-1.0,
  "reversal_signals": ["list of specific signals if any"],
  "support": nearest_support_price,
  "resistance": nearest_resistance_price,
  "direction": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "reasoning": "...",
  "key_factors": [...],
  "risk_flags": [...]
}"""


def compute_indicators(ohlcv: pd.DataFrame) -> dict:
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
    adx = ta.adx(high, low, close)
    indicators["adx"] = round(float(adx.iloc[-1, 0]), 2) if adx is not None else 0.0
    indicators["price"] = round(float(close.iloc[-1]), 2)
    # Recent price changes for context
    indicators["change_7d"] = round(float((close.iloc[-1] / close.iloc[-7] - 1) * 100), 2) if len(close) >= 7 else 0.0
    indicators["change_30d"] = round(float((close.iloc[-1] / close.iloc[-30] - 1) * 100), 2) if len(close) >= 30 else 0.0
    return indicators


class TechAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(agent_id="tech", role_description=ROLE, model=model)

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        indicators = compute_indicators(snapshot.market.ohlcv)
        parts = [
            f"Pair: {snapshot.pair}",
            f"Timestamp: {snapshot.timestamp}",
            f"\nTechnical Indicators:\n{indicators}",
        ]
        if experience:
            parts.append(f"\nPast experience:\n{experience}")
        parts.append(
            "\nRespond with the JSON format specified in your role. "
            "Focus on regime detection, not direction prediction."
        )
        return "\n".join(parts)
