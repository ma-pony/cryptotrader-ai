"""Pure trigger condition functions — no I/O, no side effects."""

from __future__ import annotations

from typing import Any


def check_price_threshold(current_price: float, parameters: dict[str, Any]) -> bool:
    """Check if current price crossed a threshold.

    Parameters:
        direction: "above" | "below"
        price: float threshold
    """
    direction = parameters.get("direction", "below")
    threshold = float(parameters.get("price", 0))
    if threshold <= 0:
        return False
    if direction == "below":
        return current_price <= threshold
    if direction == "above":
        return current_price >= threshold
    return False


def check_pct_change(current_price: float, reference_price: float, parameters: dict[str, Any]) -> bool:
    """Check if percentage change exceeds threshold.

    Parameters:
        threshold_pct: float (e.g. 3.0 for 3%)
    """
    threshold_pct = float(parameters.get("threshold_pct", 0))
    if threshold_pct <= 0 or reference_price <= 0:
        return False
    pct_change = abs((current_price - reference_price) / reference_price) * 100
    return pct_change >= threshold_pct


def check_candle_pattern(candles: list[dict[str, float]], parameters: dict[str, Any]) -> bool:
    """Check for consecutive candle pattern.

    Parameters:
        candle_count: int (minimum consecutive candles)
        direction: "bearish" | "bullish"

    Each candle dict must have "open" and "close" keys.
    """
    count = int(parameters.get("candle_count", 3))
    direction = parameters.get("direction", "bearish")
    if len(candles) < count:
        return False

    recent = candles[-count:]
    if direction == "bearish":
        return all(c["close"] < c["open"] for c in recent)
    if direction == "bullish":
        return all(c["close"] > c["open"] for c in recent)
    return False


def check_funding_rate(funding_rate: float, parameters: dict[str, Any]) -> bool:
    """Check if funding rate exceeds threshold (absolute value).

    Parameters:
        threshold_pct: float (e.g. 0.1 for 0.1%)
    """
    threshold_pct = float(parameters.get("threshold_pct", 0))
    if threshold_pct <= 0:
        return False
    return abs(funding_rate) * 100 >= threshold_pct
