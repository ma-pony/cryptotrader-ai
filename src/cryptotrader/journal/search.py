"""Search journal for similar market conditions."""

from __future__ import annotations

from cryptotrader.journal.store import JournalStore
from cryptotrader.models import DecisionCommit


async def search_similar(
    store: JournalStore,
    funding_rate: float,
    volatility: float,
    limit: int = 3,
) -> list[DecisionCommit]:
    """Fetch recent commits and filter by similar funding_rate/volatility (within 50%)."""
    recent = await store.log(limit=100)
    matches: list[DecisionCommit] = []
    for dc in recent:
        s = dc.snapshot_summary
        fr = s.get("funding_rate", 0.0)
        vol = s.get("volatility", 0.0)
        if _within_range(fr, funding_rate) and _within_range(vol, volatility):
            matches.append(dc)
            if len(matches) >= limit:
                break
    return matches


def _within_range(a: float, b: float) -> bool:
    # Both near zero — only match if both are truly near zero
    if abs(b) < 0.001 and abs(a) < 0.001:
        return True
    if abs(b) < 0.001 or abs(a) < 0.001:
        return False  # one is near zero, the other isn't
    return abs(a - b) / max(abs(a), abs(b)) <= 0.5
