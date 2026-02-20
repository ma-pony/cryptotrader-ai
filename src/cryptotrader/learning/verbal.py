"""Verbal experience from historical journal entries."""

from __future__ import annotations

from cryptotrader.journal.search import search_similar
from cryptotrader.journal.store import JournalStore


async def get_experience(store: JournalStore, snapshot_summary: dict) -> str:
    """Search for similar historical conditions and format as verbal experience."""
    fr = snapshot_summary.get("funding_rate", 0.0)
    vol = snapshot_summary.get("volatility", 0.0)
    similar = await search_similar(store, fr, vol, limit=3)
    if not similar:
        return ""
    lines: list[str] = []
    for dc in similar:
        outcome = f"pnl={dc.pnl}" if dc.pnl is not None else "no outcome yet"
        verdict_action = dc.verdict.action if dc.verdict else "hold"
        lines.append(
            f"- {dc.pair} @ {dc.timestamp:%Y-%m-%d}: verdict={verdict_action}, {outcome}"
        )
    return "Historical similar conditions:\n" + "\n".join(lines)
