"""Partial verdict — quick decision from incomplete agent analyses."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DIRECTION_SCORE = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}


def make_partial_verdict(analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Generate a quick verdict from partial agent analyses.

    Uses weighted average for confidence and majority vote for direction.
    Returns a verdict dict with ``is_partial: True``.
    """
    if not analyses:
        return {
            "action": "hold",
            "confidence": 0.0,
            "position_scale": 0.0,
            "reasoning": "No agent analyses available",
            "is_partial": True,
            "completed_agents": [],
            "missing_agents": [],
        }

    total_weight = 0.0
    weighted_score = 0.0
    weighted_conf = 0.0
    directions: list[str] = []

    for analysis in analyses.values():
        conf = float(analysis.get("confidence", 0))
        direction = analysis.get("direction", "neutral")
        score = _DIRECTION_SCORE.get(direction, 0.0)
        weighted_score += score * conf
        weighted_conf += conf
        total_weight += 1.0
        directions.append(direction)

    avg_conf = weighted_conf / total_weight if total_weight else 0.0
    avg_score = weighted_score / total_weight if total_weight else 0.0

    if avg_score > 0.2:
        action = "long"
    elif avg_score < -0.2:
        action = "short"
    else:
        action = "hold"

    all_agents = {"tech_agent", "chain_agent", "news_agent", "macro_agent"}
    completed = list(analyses.keys())
    missing = sorted(all_agents - set(completed))

    return {
        "action": action,
        "confidence": round(avg_conf, 4),
        "position_scale": round(avg_conf * abs(avg_score), 4),
        "reasoning": f"Partial verdict from {len(completed)}/{len(all_agents)} agents",
        "is_partial": True,
        "completed_agents": completed,
        "missing_agents": missing,
    }
