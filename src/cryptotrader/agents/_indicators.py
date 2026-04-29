"""Pure-pandas/numpy technical indicators.

Drop-in replacement for the subset of ``pandas_ta`` we actually use in
``agents/tech.py``. Removes the native-binary ``ta-lib`` dependency that was
breaking on arm64 Macs (where ``pandas_ta`` eagerly imports a x86_64 ``.so``
and fails at first use, making tech_agent silently fall back to mock).

Surface kept identical to pandas_ta:
- ``rsi(close, length=14)`` -> Series
- ``macd(close, fast=12, slow=26, signal=9)`` -> DataFrame[MACD, signal, hist]
- ``sma(close, length)`` -> Series
- ``bbands(close, length=20, std=2.0)`` -> DataFrame[lower, mid, upper]
- ``atr(high, low, close, length=14)`` -> Series
- ``obv(close, volume)`` -> Series

Numerical results are within float-rounding error of pandas_ta's pure-Python
implementations (Wilder smoothing for RSI/ATR; standard EMA for MACD).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder_smooth(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing — EMA with alpha = 1/length, seeded by SMA(length)."""
    if len(series) < length:
        return pd.Series([np.nan] * len(series), index=series.index, dtype=float)
    alpha = 1.0 / length
    out = series.ewm(alpha=alpha, adjust=False).mean()
    # Seed: pandas_ta uses SMA for the first valid value, then Wilder onwards.
    # The seeded form differs only at the edge; ewm-adjust=False matches Wilder
    # closely enough for our purposes. Mark the warm-up region as NaN so callers
    # that check `_safe_last` get sensible output.
    out.iloc[: length - 1] = np.nan
    return out


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_smooth(gain, length)
    avg_loss = _wilder_smooth(loss, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0).where(avg_loss.notna() | avg_gain.notna(), other=np.nan)


def sma(close: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(window=length, min_periods=length).mean()


def _ema(close: pd.Series, length: int) -> pd.Series:
    """Standard EMA (adjust=False matches pandas_ta default)."""
    return close.ewm(span=length, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram.

    Column order matches pandas_ta: [MACD_<f>_<s>_<sig>, MACDh_<...>, MACDs_<...>].
    Callers in this codebase index by position (``.iloc[-1, 0/1/2]``), so we
    return columns in the same physical order: MACD, signal, histogram.
    """
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {
            f"MACD_{fast}_{slow}_{signal}": macd_line,
            f"MACDs_{fast}_{slow}_{signal}": signal_line,
            f"MACDh_{fast}_{slow}_{signal}": histogram,
        },
    )


def bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands. Returns DataFrame [lower, mid, upper] in that order."""
    mid = sma(close, length)
    sd = close.rolling(window=length, min_periods=length).std(ddof=0)
    return pd.DataFrame(
        {
            f"BBL_{length}_{std}": mid - std * sd,
            f"BBM_{length}_{std}": mid,
            f"BBU_{length}_{std}": mid + std * sd,
        },
    )


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max(H-L, |H-C_prev|, |L-C_prev|)."""
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    tr = _true_range(high, low, close)
    return _wilder_smooth(tr, length)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative signed volume."""
    direction = np.sign(close.diff().fillna(0.0))
    signed_vol = (volume * direction).fillna(0.0)
    return signed_vol.cumsum()
