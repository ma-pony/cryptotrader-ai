"""Verbal experience from historical journal entries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.journal.search import search_by_regime, search_similar

if TYPE_CHECKING:
    from cryptotrader.config import RegimeThresholdsConfig
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.models import DecisionCommit


async def get_experience(
    store: JournalStore,
    snapshot_summary: dict,
    regime_tags: list[str] | None = None,
    thresholds: RegimeThresholdsConfig | None = None,
) -> list[DecisionCommit]:
    """Search for similar historical conditions and return matched commits.

    Prefers regime-based search when regime_tags are provided.
    Falls back to numeric similarity matching.
    Returns list[DecisionCommit] (formatting is done downstream by context.py).
    """
    # Regime-based search (preferred)
    if regime_tags and thresholds:
        cases = await search_by_regime(store, regime_tags, thresholds, limit=5)
        if cases:
            return cases

    # Fallback: numeric similarity
    fr = snapshot_summary.get("funding_rate", 0.0)
    vol = snapshot_summary.get("volatility", 0.0)
    price_change_7d = snapshot_summary.get("price_change_7d")
    return await search_similar(store, fr, vol, limit=3, price_change_7d=price_change_7d)


async def format_experience_text(cases: list[DecisionCommit]) -> str:
    """Format cases as plain text (legacy compatibility for non-GSSC paths)."""
    if not cases:
        return ""
    lines: list[str] = []
    for dc in cases:
        outcome = f"pnl={dc.pnl}" if dc.pnl is not None else "no outcome yet"
        verdict_action = dc.verdict.action if dc.verdict else "hold"
        line = f"- {dc.pair} @ {dc.timestamp:%Y-%m-%d}: verdict={verdict_action}, {outcome}"
        if dc.retrospective:
            line += f"\n  Lesson: {dc.retrospective}"
        lines.append(line)
    return "Historical similar conditions:\n" + "\n".join(lines)
