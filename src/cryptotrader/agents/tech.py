"""Technical analysis agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pandas_ta as ta

from cryptotrader.agents.base import BaseAgent

if TYPE_CHECKING:
    from cryptotrader.models import DataSnapshot

ROLE = (
    "You are an expert technical analyst for cryptocurrency markets. "
    "Analyze price action, indicators, and chart patterns to determine market direction.\n\n"
    "Focus on: trend structure (higher highs/lows vs lower), momentum (RSI divergences, MACD crossovers), "
    "volatility regime (Bollinger Band width, ATR), and key support/resistance levels relative to current price.\n"
    "When indicators conflict (e.g. RSI oversold but trend bearish), explicitly state the conflict and weight "
    "the higher-timeframe signal more heavily.\n\n"
    "Domain checklist (verify before signaling):\n"
    "- Trend alignment: Is SMA20 vs SMA60 confirming or contradicting my call? A bullish call against a bearish "
    "SMA cross needs strong justification.\n"
    "- Divergence scan: Is RSI or MACD diverging from price? A new price high with lower RSI is a warning, not "
    "confirmation.\n"
    "- Volatility regime: Is BB width expanding (breakout) or contracting (squeeze)? A squeeze means the next move "
    "will be violent — lower confidence, not higher.\n"
    "- Support/resistance proximity: Is price within 2% of a key level? If yes, the level matters more than the "
    "trend."
)


def _safe_last(series) -> float | None:
    """Return last non-NaN value from a pandas Series, or None."""
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)


def _recent_values(series, n: int = 5) -> list[float]:
    """Return the last N non-NaN values from a pandas Series as rounded floats."""
    if series is None or series.empty:
        return []
    clean = series.dropna()
    tail = clean.iloc[-n:] if len(clean) >= n else clean
    return [round(float(v), 2) for v in tail]


def _compute_volume_indicators(close: pd.Series, volume: pd.Series, n: int) -> dict:
    """Compute volume-related indicators: volume ratio and OBV trend."""
    result: dict = {}
    if n < 2:
        return result

    # Volume ratio: current bar vs 20-bar average
    avg_window = min(20, n)
    avg_vol = volume.iloc[-avg_window:].mean()
    if avg_vol and avg_vol > 0:
        ratio = float(volume.iloc[-1]) / float(avg_vol)
        if ratio > 2.0:
            label = "SPIKE"
        elif ratio > 1.5:
            label = "HIGH"
        elif ratio < 0.5:
            label = "LOW"
        else:
            label = "NORMAL"
        result["volume_ratio"] = round(ratio, 2)
        result["volume_label"] = label

    # OBV trend: slope direction over last 5 bars
    obv = ta.obv(close, volume)
    if obv is not None and not obv.empty:
        obv_tail = _recent_values(obv, n=5)
        if len(obv_tail) >= 2:
            slope = obv_tail[-1] - obv_tail[0]
            result["obv_trend"] = "rising" if slope > 0 else "falling"

    return result


def _compute_trend_fields(
    close: pd.Series,
    rsi_series,
    macd_df,
    sma20: float | None,
    sma60: float | None,
) -> dict:
    """Compute trend/context fields: RSI trend, MACD hist trend, price vs SMAs."""
    result: dict = {}

    if rsi_series is not None and not rsi_series.empty:
        result["rsi_trend"] = _recent_values(rsi_series, n=5)

    if macd_df is not None and not macd_df.empty:
        hist_col = macd_df.iloc[:, 2]
        result["macd_hist_trend"] = _recent_values(hist_col, n=5)

    last_close = float(close.iloc[-1])
    if sma20 is not None and sma20 != 0:
        result["price_vs_sma20"] = round((last_close - sma20) / sma20 * 100, 2)
    if sma60 is not None and sma60 != 0:
        result["price_vs_sma60"] = round((last_close - sma60) / sma60 * 100, 2)

    return result


def compute_indicators(ohlcv: pd.DataFrame) -> dict:
    """Compute technical indicators from OHLCV data."""
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"] if "volume" in ohlcv.columns else None
    n = len(close)
    indicators: dict = {}

    rsi_series = None
    macd_df = None
    sma20: float | None = None
    sma60: float | None = None

    if n >= 15:
        rsi_series = ta.rsi(close, length=14)
        rsi = _safe_last(rsi_series)
        indicators["rsi"] = round(rsi, 2) if rsi is not None else None
    if n >= 34:
        macd_df = ta.macd(close)
        if macd_df is not None and not macd_df.empty:
            indicators["macd"] = {
                "macd": round(float(macd_df.iloc[-1, 0]), 4) if not pd.isna(macd_df.iloc[-1, 0]) else 0,
                "signal": round(float(macd_df.iloc[-1, 1]), 4) if not pd.isna(macd_df.iloc[-1, 1]) else 0,
                "histogram": round(float(macd_df.iloc[-1, 2]), 4) if not pd.isna(macd_df.iloc[-1, 2]) else 0,
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

    # Volume indicators
    if volume is not None:
        indicators.update(_compute_volume_indicators(close, volume, n))

    # Trend context fields
    indicators.update(_compute_trend_fields(close, rsi_series, macd_df if n >= 34 else None, sma20, sma60))

    return indicators


class TechAgent(BaseAgent):
    def __init__(self, model: str = "") -> None:
        super().__init__(agent_id="tech", role_description=ROLE, model=model)

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        base = super()._build_prompt(snapshot, experience)
        indicators = compute_indicators(snapshot.market.ohlcv)
        return f"Technical Indicators:\n{indicators}\n\n{base}"
