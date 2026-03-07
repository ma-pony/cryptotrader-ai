"""Tests for accuracy_report, calibrate_weights, and bias detection."""

from datetime import UTC, datetime

import pytest

from cryptotrader.journal.calibrate import (
    accuracy_report,
    calibrate_weights,
    detect_biases,
    generate_bias_correction,
    generate_verdict_calibration,
)
from cryptotrader.journal.store import JournalStore
from cryptotrader.models import (
    AgentAnalysis,
    DecisionCommit,
    GateResult,
    TradeVerdict,
)


def _make_commit(pair, pnl, directions, ts=None):
    """Helper to build a DecisionCommit with given agent directions and pnl."""
    if ts is None:
        ts = datetime.now(UTC)
    analyses = {}
    for agent_id, direction in directions.items():
        analyses[agent_id] = AgentAnalysis(
            agent_id=agent_id,
            pair=pair,
            direction=direction,
            confidence=0.8,
            reasoning="test",
        )
    return DecisionCommit(
        hash=f"h{id(directions)}",
        parent_hash=None,
        timestamp=ts,
        pair=pair,
        snapshot_summary={},
        analyses=analyses,
        debate_rounds=1,
        pnl=pnl,
        verdict=TradeVerdict(action="long", confidence=0.8),
        risk_gate=GateResult(passed=True),
    )


@pytest.mark.asyncio
async def test_accuracy_bullish_correct():
    store = JournalStore()
    # Bullish agent + positive pnl = correct
    dc = _make_commit(
        "BTC/USDT",
        pnl=100.0,
        directions={
            "tech": "bullish",
            "chain": "bearish",
        },
    )
    await store.commit(dc)
    acc = await accuracy_report(store, days=30)
    assert acc["tech"] == 1.0  # 1/1 correct
    assert acc["chain"] == 0.0  # 0/1 correct


@pytest.mark.asyncio
async def test_accuracy_neutral_not_rewarded():
    store = JournalStore()
    # Neutral agent should NOT count as correct
    dc = _make_commit(
        "BTC/USDT",
        pnl=100.0,
        directions={
            "tech": "neutral",
            "chain": "bullish",
        },
    )
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
    dc1 = _make_commit(
        "BTC/USDT",
        pnl=50.0,
        directions={
            "tech": "bullish",
            "chain": "bullish",
        },
    )
    dc2 = _make_commit(
        "BTC/USDT",
        pnl=-30.0,
        directions={
            "tech": "bearish",
            "chain": "bullish",
        },
    )
    await store.commit(dc1)
    await store.commit(dc2)
    weights = await calibrate_weights(store, days=30)
    assert len(weights) == 2
    # Weights should sum to 1.0
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    # tech should have higher weight
    assert weights["tech"] > weights["chain"]


def _make_commit_with_conf(pair, pnl, agent_data, ts=None):
    """Helper with per-agent direction + confidence."""
    if ts is None:
        ts = datetime.now(UTC)
    analyses = {}
    for agent_id, (direction, confidence) in agent_data.items():
        analyses[agent_id] = AgentAnalysis(
            agent_id=agent_id,
            pair=pair,
            direction=direction,
            confidence=confidence,
            reasoning="test",
        )
    return DecisionCommit(
        hash=f"h{id(agent_data)}{pnl}",
        parent_hash=None,
        timestamp=ts,
        pair=pair,
        snapshot_summary={},
        analyses=analyses,
        debate_rounds=1,
        pnl=pnl,
        verdict=TradeVerdict(action="long", confidence=0.8),
        risk_gate=GateResult(passed=True),
    )


@pytest.mark.asyncio
async def test_detect_biases_bullish_bias():
    store = JournalStore()
    # tech always bullish (4/5), chain mixed
    for i in range(5):
        pnl = 10.0 if i % 2 == 0 else -10.0
        dc = _make_commit_with_conf(
            "BTC/USDT",
            pnl=pnl,
            agent_data={
                "tech": ("bullish", 0.8),
                "chain": ("bearish" if i % 2 == 0 else "bullish", 0.6),
            },
        )
        await store.commit(dc)
    biases = await detect_biases(store, days=30)
    assert "tech" in biases
    assert biases["tech"]["bullish_rate"] == 1.0  # always bullish
    assert biases["tech"]["sample_size"] == 5


@pytest.mark.asyncio
async def test_detect_biases_overconfidence():
    store = JournalStore()
    # Agent is always confident (0.9) but wrong half the time
    for _i in range(6):
        pnl = -50.0  # always wrong
        dc = _make_commit_with_conf(
            "BTC/USDT",
            pnl=pnl,
            agent_data={"tech": ("bullish", 0.9)},
        )
        await store.commit(dc)
    biases = await detect_biases(store, days=30)
    assert biases["tech"]["avg_conf_when_wrong"] == 0.9


@pytest.mark.asyncio
async def test_detect_biases_insufficient_data():
    store = JournalStore()
    # Only 2 commits — below threshold of 3
    for _i in range(2):
        dc = _make_commit_with_conf("BTC/USDT", pnl=10.0, agent_data={"tech": ("bullish", 0.7)})
        await store.commit(dc)
    biases = await detect_biases(store, days=30)
    assert biases == {}  # not enough data


def test_generate_bias_correction_overconfident():
    biases = {
        "tech_agent": {
            "accuracy": 0.35,
            "neutral_rate": 0.1,
            "bullish_rate": 0.8,
            "bearish_rate": 0.1,
            "avg_conf_when_right": 0.7,
            "avg_conf_when_wrong": 0.75,
            "sample_size": 10,
        }
    }
    result = generate_bias_correction(biases)
    assert "OVERCONFIDENT" in result
    assert "BULLISH BIAS" in result
    assert "LOW ACCURACY" in result


def test_generate_bias_correction_empty():
    assert generate_bias_correction({}) == ""


def test_generate_bias_correction_no_warnings():
    biases = {
        "tech_agent": {
            "accuracy": 0.65,
            "neutral_rate": 0.1,
            "bullish_rate": 0.5,
            "bearish_rate": 0.4,
            "avg_conf_when_right": 0.7,
            "avg_conf_when_wrong": 0.5,
            "sample_size": 10,
        }
    }
    assert generate_bias_correction(biases) == ""


def test_generate_verdict_calibration():
    biases = {
        "tech_agent": {
            "accuracy": 0.7,
            "avg_conf_when_wrong": 0.5,
            "sample_size": 10,
        },
        "chain_agent": {
            "accuracy": 0.3,
            "avg_conf_when_wrong": 0.8,
            "sample_size": 8,
        },
    }
    result = generate_verdict_calibration(biases)
    assert "TECH" in result
    assert "reliable" in result
    assert "CHAIN" in result
    assert "unreliable" in result
    assert "overconfident" in result


def test_generate_verdict_calibration_small_sample():
    biases = {"tech_agent": {"accuracy": 0.3, "avg_conf_when_wrong": 0.9, "sample_size": 3}}
    assert generate_verdict_calibration(biases) == ""  # below 5 sample threshold
