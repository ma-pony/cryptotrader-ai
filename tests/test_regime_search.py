"""Tests for regime-based journal search."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

from cryptotrader.config import RegimeThresholdsConfig
from cryptotrader.journal.search import search_by_regime
from cryptotrader.models import AgentAnalysis, DecisionCommit, TradeVerdict


def _make_dc(funding_rate: float = 0.0001, volatility: float = 0.015, **kwargs) -> DecisionCommit:
    return DecisionCommit(
        hash="h1",
        parent_hash=None,
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        snapshot_summary={
            "funding_rate": funding_rate,
            "volatility": volatility,
            **kwargs,
        },
        analyses={
            "tech_agent": AgentAnalysis(
                agent_id="tech_agent",
                pair="BTC/USDT",
                direction="bullish",
                confidence=0.7,
                reasoning="test",
            ),
        },
        debate_rounds=0,
        verdict=TradeVerdict(action="hold", confidence=0.5),
    )


async def test_search_by_regime_returns_matching():
    """Commits with matching regime tags are returned."""
    store = AsyncMock()
    store.log.return_value = [
        _make_dc(funding_rate=0.0005, volatility=0.04),  # high_funding + high_vol
        _make_dc(funding_rate=0.0001, volatility=0.015),  # neutral
        _make_dc(funding_rate=0.0004, volatility=0.03),  # high_funding + high_vol
    ]

    thresholds = RegimeThresholdsConfig()
    results = await search_by_regime(store, ["high_funding", "high_vol"], thresholds, limit=5)

    assert len(results) == 2  # Only 2 match high_funding+high_vol


async def test_search_by_regime_sorted_by_overlap():
    """Results are sorted by Jaccard overlap (best match first)."""
    store = AsyncMock()
    store.log.return_value = [
        _make_dc(funding_rate=0.0005, volatility=0.015),  # high_funding only
        _make_dc(funding_rate=0.0005, volatility=0.04),  # high_funding + high_vol (better match)
    ]

    thresholds = RegimeThresholdsConfig()
    results = await search_by_regime(store, ["high_funding", "high_vol"], thresholds, limit=5)

    assert len(results) == 2
    # First result should have higher overlap (both tags match)
    first_summary = results[0].snapshot_summary
    assert first_summary["volatility"] == 0.04  # The one with both tags


async def test_search_by_regime_empty_tags():
    """Empty regime tags returns empty list."""
    store = AsyncMock()
    thresholds = RegimeThresholdsConfig()
    results = await search_by_regime(store, [], thresholds, limit=5)
    assert results == []


async def test_search_by_regime_respects_limit():
    """Results are limited to requested count."""
    store = AsyncMock()
    store.log.return_value = [_make_dc(funding_rate=0.0005, volatility=0.04) for _ in range(10)]

    thresholds = RegimeThresholdsConfig()
    results = await search_by_regime(store, ["high_funding"], thresholds, limit=3)
    assert len(results) == 3
