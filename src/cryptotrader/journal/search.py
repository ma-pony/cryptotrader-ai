"""Search journal for similar market conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.config import RegimeThresholdsConfig
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.models import DecisionCommit


async def search_by_regime(
    store: JournalStore,
    regime_tags: list[str],
    thresholds: RegimeThresholdsConfig,
    limit: int = 5,
) -> list[DecisionCommit]:
    """Search journal for commits matching current regime tags (Jaccard overlap).

    Fetches recent commits, tags each one's snapshot_summary with regime labels,
    and returns top matches sorted by Jaccard overlap.
    """
    from cryptotrader.learning.regime import regime_overlap, tag_regime

    if not regime_tags:
        return []
    recent = await store.log(limit=200)
    scored: list[tuple[float, DecisionCommit]] = []
    for dc in recent:
        dc_tags = tag_regime(dc.snapshot_summary or {}, thresholds)
        overlap = regime_overlap(regime_tags, dc_tags)
        if overlap > 0:
            scored.append((overlap, dc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [dc for _, dc in scored[:limit]]


async def search_similar(
    store: JournalStore,
    funding_rate: float,
    volatility: float,
    limit: int = 3,
    price_change_7d: float | None = None,
) -> list[DecisionCommit]:
    """Fetch recent commits and filter by similar market conditions.

    Matches on funding_rate and volatility (within 50% relative tolerance),
    plus price trend direction when available (both must have same sign).
    """
    recent = await store.log(limit=100)
    matches: list[DecisionCommit] = []
    for dc in recent:
        s = dc.snapshot_summary
        fr = s.get("funding_rate", 0.0)
        vol = s.get("volatility", 0.0)
        if not (_within_range(fr, funding_rate) and _within_range(vol, volatility)):
            continue
        # Price trend direction: if caller provides 7d change, require same sign
        if price_change_7d is not None:
            hist_change = s.get("price_change_7d")
            if hist_change is not None and not _same_trend(price_change_7d, hist_change):
                continue
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


def _same_trend(a: float, b: float) -> bool:
    """Check if two price changes have the same directional trend.

    Both positive → True, both negative → True, both near-zero → True.
    One positive and one negative → False.
    """
    threshold = 0.005  # <0.5% change treated as flat/no-trend
    a_flat = abs(a) < threshold
    b_flat = abs(b) < threshold
    if a_flat or b_flat:
        return True  # flat is compatible with any direction
    return (a > 0) == (b > 0)
