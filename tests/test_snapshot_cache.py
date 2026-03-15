"""Task 4.1 -- snapshot hash reuse tests.

Coverage:
1. collect_snapshot() computes and writes snapshot_hash
2. Same snapshot reuses prev_analyses and skips LLM calls
3. Different snapshot runs LLM and updates prev_snapshot_hash
4. First call without prev_snapshot_hash does not trigger reuse
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

from cryptotrader.models import (
    AgentAnalysis,
    DataSnapshot,
    MacroData,
    MarketData,
    NewsSentiment,
    OnchainData,
)


def _make_snapshot(pair="BTC/USDT", price=50000.0, funding_rate=0.01, volatility=0.02, imbalance=0.5):
    return DataSnapshot(
        timestamp=None,
        pair=pair,
        market=MarketData(
            pair=pair,
            ohlcv=None,
            ticker={"last": price, "baseVolume": 1000},
            funding_rate=funding_rate,
            orderbook_imbalance=imbalance,
            volatility=volatility,
        ),
        onchain=OnchainData(),
        news=NewsSentiment(),
        macro=MacroData(),
    )


def _base_state(pair="BTC/USDT", price=50000.0, **extra_data):
    snapshot = _make_snapshot(pair, price)
    data = {
        "snapshot": snapshot,
        "snapshot_summary": {
            "pair": pair,
            "price": price,
            "funding_rate": 0.01,
            "volatility": 0.02,
            "orderbook_imbalance": 0.5,
        },
        "experience": "",
        **extra_data,
    }
    return {
        "messages": [],
        "data": data,
        "metadata": {
            "pair": pair,
            "engine": "paper",
            "models": {},
            "analysis_model": "gpt-4o-mini",
            "debate_model": "gpt-4o-mini",
            "verdict_model": "gpt-4o-mini",
        },
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }


def _expected_hash(price=50000.0, funding_rate=0.01, volatility=0.02, imbalance=0.5):
    """Compute expected SHA256 hash matching the implementation."""
    key_fields = {
        "price": price,
        "funding_rate": funding_rate,
        "volatility": volatility,
        "orderbook_imbalance": imbalance,
    }
    payload = json.dumps(key_fields, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# --- collect_snapshot() writes snapshot_hash ---


async def test_collect_snapshot_writes_hash():
    """collect_snapshot() should write snapshot_hash into state['data']."""
    from cryptotrader.nodes.data import collect_snapshot

    state = _base_state()
    result = await collect_snapshot(state)

    assert "snapshot_hash" in result["data"], "snapshot_hash should be written to data"


async def test_collect_snapshot_hash_is_sha256():
    """snapshot_hash should be a 64-character hex string (SHA256)."""
    from cryptotrader.nodes.data import collect_snapshot

    state = _base_state()
    result = await collect_snapshot(state)

    h = result["data"]["snapshot_hash"]
    assert isinstance(h, str), "hash should be a string"
    assert len(h) == 64, f"SHA256 should be 64 chars, got {len(h)}"
    int(h, 16)  # must be valid hex


async def test_collect_snapshot_hash_deterministic():
    """Same snapshot data should produce the same hash."""
    from cryptotrader.nodes.data import collect_snapshot

    state1 = _base_state(price=50000.0)
    state2 = _base_state(price=50000.0)
    r1 = await collect_snapshot(state1)
    r2 = await collect_snapshot(state2)

    assert r1["data"]["snapshot_hash"] == r2["data"]["snapshot_hash"]


async def test_collect_snapshot_hash_changes_on_price():
    """Different price should produce a different hash."""
    from cryptotrader.nodes.data import collect_snapshot

    state1 = _base_state(price=50000.0)
    state2 = _base_state(price=51000.0)
    state2["data"]["snapshot"].market.ticker["last"] = 51000.0

    r1 = await collect_snapshot(state1)
    r2 = await collect_snapshot(state2)

    assert r1["data"]["snapshot_hash"] != r2["data"]["snapshot_hash"]


async def test_collect_snapshot_hash_correct_value():
    """snapshot_hash should match a manually computed SHA256."""
    from cryptotrader.nodes.data import collect_snapshot

    state = _base_state()
    result = await collect_snapshot(state)

    expected = _expected_hash(
        price=50000.0,
        funding_rate=0.01,
        volatility=0.02,
        imbalance=0.5,
    )
    assert result["data"]["snapshot_hash"] == expected


# --- same snapshot reuses prev_analyses, skips LLM ---


async def test_agent_reuses_prev_analyses_when_hash_same():
    """When snapshot_hash == prev_snapshot_hash, tech_analyze reuses prev_analyses without LLM."""
    from cryptotrader.nodes.agents import tech_analyze

    cached_hash = _expected_hash()
    prev_analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.9, "reasoning": "cached", "is_mock": False},
        "chain_agent": {"direction": "neutral", "confidence": 0.5, "reasoning": "cached", "is_mock": False},
    }

    state = _base_state(
        snapshot_hash=cached_hash,
        prev_snapshot_hash=cached_hash,
        prev_analyses=prev_analyses,
    )

    mock_agent_cls = MagicMock()
    mock_agent_inst = mock_agent_cls.return_value
    mock_agent_inst.analyze = AsyncMock()

    with patch("cryptotrader.agents.tech.TechAgent", mock_agent_cls):
        result = await tech_analyze(state)

    # LLM should NOT be called
    mock_agent_inst.analyze.assert_not_called()

    # Should return the cached analysis
    analyses = result["data"]["analyses"]
    assert "tech_agent" in analyses
    assert analyses["tech_agent"]["direction"] == "bullish"
    assert analyses["tech_agent"]["reasoning"] == "cached"


async def test_all_agents_reuse_when_hash_same():
    """All four agents should reuse prev_analyses when hash matches."""
    from cryptotrader.nodes.agents import chain_analyze, macro_analyze, news_analyze, tech_analyze

    cached_hash = _expected_hash()
    prev_analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.8, "reasoning": "cached_tech", "is_mock": False},
        "chain_agent": {"direction": "bearish", "confidence": 0.6, "reasoning": "cached_chain", "is_mock": False},
        "news_agent": {"direction": "neutral", "confidence": 0.5, "reasoning": "cached_news", "is_mock": False},
        "macro_agent": {"direction": "bullish", "confidence": 0.7, "reasoning": "cached_macro", "is_mock": False},
    }

    state = _base_state(
        snapshot_hash=cached_hash,
        prev_snapshot_hash=cached_hash,
        prev_analyses=prev_analyses,
    )

    agent_modules = {
        "tech_agent": ("cryptotrader.agents.tech", "TechAgent", tech_analyze),
        "chain_agent": ("cryptotrader.agents.chain", "ChainAgent", chain_analyze),
        "news_agent": ("cryptotrader.agents.news", "NewsAgent", news_analyze),
        "macro_agent": ("cryptotrader.agents.macro", "MacroAgent", macro_analyze),
    }

    for agent_key, (module, cls_name, fn) in agent_modules.items():
        with patch(f"{module}.{cls_name}") as mock_cls:
            mock_cls.return_value.analyze = AsyncMock()
            result = await fn(state)

        mock_cls.return_value.analyze.assert_not_called()
        assert result["data"]["analyses"][agent_key]["reasoning"] == prev_analyses[agent_key]["reasoning"]


# --- different snapshot runs LLM and updates prev_snapshot_hash ---


async def test_agent_runs_llm_when_hash_differs():
    """When snapshot_hash != prev_snapshot_hash, tech_analyze calls LLM normally."""
    from cryptotrader.nodes.agents import tech_analyze

    old_hash = _expected_hash(price=49000.0)
    new_hash = _expected_hash(price=50000.0)

    state = _base_state(
        snapshot_hash=new_hash,
        prev_snapshot_hash=old_hash,
        prev_analyses={},
    )

    mock_analysis = AgentAnalysis(
        agent_id="tech_agent",
        pair="BTC/USDT",
        direction="bullish",
        confidence=0.8,
        reasoning="new analysis",
    )

    with patch("cryptotrader.agents.tech.TechAgent") as mock_cls:
        mock_cls.return_value.analyze = AsyncMock(return_value=mock_analysis)
        result = await tech_analyze(state)

    mock_cls.return_value.analyze.assert_called_once()
    assert result["data"]["analyses"]["tech_agent"]["direction"] == "bullish"


async def test_agent_updates_prev_snapshot_hash_after_llm():
    """After LLM call with differing hash, _run_agent updates prev_snapshot_hash and prev_analyses."""
    from cryptotrader.nodes.agents import tech_analyze

    old_hash = _expected_hash(price=49000.0)
    new_hash = _expected_hash(price=50000.0)

    state = _base_state(
        snapshot_hash=new_hash,
        prev_snapshot_hash=old_hash,
        prev_analyses={},
    )

    mock_analysis = AgentAnalysis(
        agent_id="tech_agent",
        pair="BTC/USDT",
        direction="bearish",
        confidence=0.6,
        reasoning="updated",
    )

    with patch("cryptotrader.agents.tech.TechAgent") as mock_cls:
        mock_cls.return_value.analyze = AsyncMock(return_value=mock_analysis)
        result = await tech_analyze(state)

    assert result["data"].get("prev_snapshot_hash") == new_hash

    prev = result["data"].get("prev_analyses", {})
    assert "tech_agent" in prev
    assert prev["tech_agent"]["direction"] == "bearish"


# --- first call (no prev_snapshot_hash) does not trigger reuse ---


async def test_agent_no_reuse_without_prev_hash():
    """Without prev_snapshot_hash (first call), reuse is not triggered and LLM runs normally."""
    from cryptotrader.nodes.agents import tech_analyze

    new_hash = _expected_hash()

    state = _base_state(snapshot_hash=new_hash)

    mock_analysis = AgentAnalysis(
        agent_id="tech_agent",
        pair="BTC/USDT",
        direction="neutral",
        confidence=0.5,
        reasoning="first run",
    )

    with patch("cryptotrader.agents.tech.TechAgent") as mock_cls:
        mock_cls.return_value.analyze = AsyncMock(return_value=mock_analysis)
        result = await tech_analyze(state)

    mock_cls.return_value.analyze.assert_called_once()
    assert result["data"]["analyses"]["tech_agent"]["direction"] == "neutral"


async def test_agent_no_reuse_without_prev_analyses():
    """When prev_analyses is empty for the agent, reuse is not triggered."""
    from cryptotrader.nodes.agents import tech_analyze

    cached_hash = _expected_hash()

    state = _base_state(
        snapshot_hash=cached_hash,
        prev_snapshot_hash=cached_hash,
        prev_analyses={},  # empty -- no cached entry for tech_agent
    )

    mock_analysis = AgentAnalysis(
        agent_id="tech_agent",
        pair="BTC/USDT",
        direction="neutral",
        confidence=0.5,
        reasoning="no prev analyses",
    )

    with patch("cryptotrader.agents.tech.TechAgent") as mock_cls:
        mock_cls.return_value.analyze = AsyncMock(return_value=mock_analysis)
        result = await tech_analyze(state)

    # LLM should be called since no cached entry exists for tech_agent
    mock_cls.return_value.analyze.assert_called_once()
    assert result["data"]["analyses"]["tech_agent"]["direction"] == "neutral"
