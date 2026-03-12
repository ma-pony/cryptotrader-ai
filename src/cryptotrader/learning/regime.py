"""Regime tagging — classify market conditions into discrete regime labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.config import RegimeThresholdsConfig


def tag_regime(snapshot_summary: dict, thresholds: RegimeThresholdsConfig) -> list[str]:
    """Return a list of regime tags based on snapshot summary data.

    Tags are discrete labels like "high_funding", "high_vol", "trending_up",
    etc. Used for condition-based experience matching (not RAG).
    """
    tags: list[str] = []
    _tag_funding(tags, snapshot_summary, thresholds)
    _tag_volatility(tags, snapshot_summary, thresholds)
    _tag_trend(tags, snapshot_summary, thresholds)
    _tag_sentiment(tags, snapshot_summary, thresholds)
    return tags


def _tag_funding(tags: list[str], summary: dict, t: RegimeThresholdsConfig) -> None:
    fr = summary.get("funding_rate", 0.0)
    if fr > t.high_funding:
        tags.append("high_funding")
    elif fr < t.negative_funding:
        tags.append("negative_funding")


def _tag_volatility(tags: list[str], summary: dict, t: RegimeThresholdsConfig) -> None:
    vol = summary.get("volatility", 0.0)
    if vol > t.high_vol:
        tags.append("high_vol")
    elif vol < t.low_vol:
        tags.append("low_vol")


def _tag_trend(tags: list[str], summary: dict, t: RegimeThresholdsConfig) -> None:
    price_change_7d = summary.get("price_change_7d")
    if price_change_7d is None:
        return
    if price_change_7d > t.trending_up:
        tags.append("trending_up")
    elif price_change_7d < t.trending_down:
        tags.append("trending_down")


def _tag_sentiment(tags: list[str], summary: dict, t: RegimeThresholdsConfig) -> None:
    fng = summary.get("fear_greed_index")
    if fng is None:
        return
    if fng <= t.extreme_fear_fng:
        tags.append("extreme_fear")
    elif fng >= t.extreme_greed_fng:
        tags.append("extreme_greed")


def regime_overlap(tags_a: list[str], tags_b: list[str]) -> float:
    """Jaccard similarity between two sets of regime tags.

    Returns 0.0 if both are empty, 1.0 if identical.
    """
    set_a = set(tags_a)
    set_b = set(tags_b)
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)
