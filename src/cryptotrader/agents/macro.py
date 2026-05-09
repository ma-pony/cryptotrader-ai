"""Macro environment analysis agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.agents.base import BaseAgent

if TYPE_CHECKING:
    from cryptotrader.agents.prompt_builder import PromptBuilder


def _format_fear_greed_trend(history: list[int]) -> str:
    """Format 7-day Fear & Greed history as a trend string with direction label."""
    if not history or len(history) < 2:
        return ""
    # API returns newest-first; reverse to get chronological order
    chron = list(reversed(history))
    trend_str = "→".join(str(v) for v in chron)
    delta = chron[-1] - chron[0]
    if delta > 10:
        direction = "recovering"
    elif delta < -10:
        direction = "deteriorating"
    else:
        direction = "stable"
    return f"7d trend: {trend_str} ({direction})"


def _format_etf_top_flows(top_flows: list[dict]) -> str:
    """Format per-ticker ETF flows as a compact string."""
    if not top_flows:
        return ""
    parts = []
    for item in top_flows:
        ticker = item.get("ticker", "?")
        flow = item.get("dailyNetInflow", 0)
        sign = "+" if flow >= 0 else ""
        parts.append(f"{ticker} {sign}${flow / 1e6:,.0f}M")
    return ", ".join(parts)


class MacroAgent(BaseAgent):
    def __init__(self, *, prompt_builder: PromptBuilder, model: str = "") -> None:
        super().__init__(agent_id="macro", prompt_builder=prompt_builder, model=model)
