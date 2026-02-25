"""Tests for accuracy_report and calibrate_weights."""

import pytest
from datetime import UTC, datetime

from cryptotrader.journal.calibrate import accuracy_report, calibrate_weights
from cryptotrader.journal.store import JournalStore
from cryptotrader.models import (
    AgentAnalysis, DecisionCommit, TradeVerdict, GateResult,
)


def _make_commit(pair, pnl, directions, ts=None):
    """Helper to build a DecisionCommit with given agent directions and pnl."""
    if ts is None:
        ts = datetime.now(UTC)
    analyses = {}
    for agent_id, direction in directions.items():
        analyses[agent_id] = AgentAnalysis(
            agent_id=agent_id, pair=pair,
            direction=direction, confidence=0.8,
            reasoning="test",
        )
    return DecisionCommit(
        hash=f"h{id(directions)}", parent_hash=None,
        timestamp=ts, pair=pair,
        snapshot_summary={}, analyses=analyses,
        debate_rounds=1, pnl=pnl,
        verdict=TradeVerdict(action="long", confidence=0.8),
        risk_gate=GateResult(passed=True),
    )


@pytest.mark.asyncio
async def test_accuracy_bullish_correct():
    store = JournalStore()
    # Bullish agent + positive pnl = correct
    dc = _make_commit("BTC/USDT", pnl=100.0, directions={
        "tech": "bullish", "chain": "bearish",
    })
    await store.commit(dc)
    acc = await accuracy_report(store, days=30)
    assert acc["tech"] == 1.0  # 1/1 correct
    assert acc["chain"] == 0.0  # 0/1 correct


@pytest.mark.asyncio
async def test_accuracy_neutral_not_rewarded():
    store = JournalStore()
    # Neutral agent should NOT count as correct
    dc = _make_commit("BTC/USDT", pnl=100.0, directions={
        "tech": "neutral", "chain": "bullish",
    })
    await store.commit(dc)
    acc = await accuracy_report(store, days=30)
    assert acc["tech"] == 0.0  # neutral not rewarded
    assert acc["chain"] == 1.0


@pytest.mark.asyncio
async def test_accuracy_skips_no_pnl():
    store = JournalStore()
    dc = _make_commit("BTC/USDT", pnl=None, directions={"tech": "bullish"})
    await store.commit(dc)
    acc = await accuracy_report(store, days=30)
    assert acc == {}  # no commits with pnl


@pytest.mark.asyncio
async def test_calibrate_weights_normalized():
    store = JournalStore()
    # tech: 2/2 correct, chain: 1/2 correct
    dc1 = _make_commit("BTC/USDT", pnl=50.0, directions={
        "tech": "bullish", "chain": "bullish",
    })
    dc2 = _make_commit("BTC/USDT", pnl=-30.0, directions={
        "tech": "bearish", "chain": "bullish",
    })
    await store.commit(dc1)
    await store.commit(dc2)
    weights = await calibrate_weights(store, days=30)
    assert len(weights) == 2
    # Weights should sum to 1.0
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    # tech should have higher weight
    assert weights["tech"] > weights["chain"]
