"""Tests for agent self-reflection system."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from cryptotrader.config import ReflectionConfig
from cryptotrader.learning.reflect import (
    _build_reflection_prompt,
    _format_commit_for_agent,
    load_reflections,
    maybe_reflect,
    save_reflection,
)
from cryptotrader.models import AgentAnalysis, DecisionCommit, TradeVerdict


def _make_commit(
    pnl: float | None = 1.5,
    agents: dict | None = None,
    verdict_action: str = "long",
) -> DecisionCommit:
    """Create a test DecisionCommit with agent analyses."""
    if agents is None:
        agents = {
            "tech_agent": AgentAnalysis(
                agent_id="tech_agent",
                pair="BTC/USDT",
                direction="bullish",
                confidence=0.75,
                reasoning="RSI oversold at 28, MACD histogram turning positive",
                key_factors=["RSI超卖", "MACD金叉"],
            ),
            "chain_agent": AgentAnalysis(
                agent_id="chain_agent",
                pair="BTC/USDT",
                direction="bearish",
                confidence=0.60,
                reasoning="Exchange inflows increasing, whale selling detected",
                key_factors=["交易所流入", "鲸鱼卖出"],
            ),
            "news_agent": AgentAnalysis(
                agent_id="news_agent",
                pair="BTC/USDT",
                direction="neutral",
                confidence=0.50,
                reasoning="Mixed news, ETF approval rumours but regulatory concerns",
                key_factors=["ETF传闻", "监管风险"],
            ),
            "macro_agent": AgentAnalysis(
                agent_id="macro_agent",
                pair="BTC/USDT",
                direction="bullish",
                confidence=0.65,
                reasoning="DXY weakening, fear & greed at 25 (extreme fear)",
                key_factors=["DXY走弱", "极度恐惧"],
            ),
        }
    return DecisionCommit(
        hash="abc123",
        parent_hash=None,
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        snapshot_summary={"price": 95000, "volatility": 0.032, "funding_rate": 0.0015},
        analyses=agents,
        debate_rounds=2,
        verdict=TradeVerdict(action=verdict_action, confidence=0.7),
        pnl=pnl,
    )


async def test_load_save_reflections():
    """Write → read → verify content matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "reflections.db"
        await save_reflection(db_path, "tech_agent", "Memo: RSI oversold works best in high-vol")
        await save_reflection(db_path, "chain_agent", "Memo: Whale alerts are noisy")

        result = await load_reflections(db_path)
        assert result["tech_agent"] == "Memo: RSI oversold works best in high-vol"
        assert result["chain_agent"] == "Memo: Whale alerts are noisy"
        assert len(result) == 2


async def test_load_reflections_empty():
    """Loading from nonexistent DB returns empty dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "nonexistent.db"
        result = await load_reflections(db_path)
        assert result == {}


async def test_save_reflection_upsert():
    """Second save overwrites the first (upsert behavior)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "reflections.db"
        await save_reflection(db_path, "tech_agent", "Old memo")
        await save_reflection(db_path, "tech_agent", "New memo")

        result = await load_reflections(db_path)
        assert result["tech_agent"] == "New memo"
        assert len(result) == 1


async def test_format_commit_for_agent():
    """Verify correct agent analysis is extracted from commit."""
    dc = _make_commit(pnl=2.5)
    rec = _format_commit_for_agent("tech_agent", dc)
    assert rec is not None
    assert rec["direction"] == "bullish"
    assert rec["confidence"] == 0.75
    assert rec["pnl"] == 2.5
    assert rec["verdict_action"] == "long"
    assert "RSI超卖" in rec["key_factors"]
    assert rec["price"] == 95000


async def test_format_commit_missing_agent():
    """Agent not in commit returns None."""
    dc = _make_commit()
    rec = _format_commit_for_agent("nonexistent_agent", dc)
    assert rec is None


async def test_build_reflection_prompt():
    """Verify prompt structure contains domain context and records."""
    records = [
        {
            "date": "2024-01-15 10:00",
            "direction": "bullish",
            "confidence": 0.75,
            "reasoning": "RSI oversold",
            "key_factors": ["RSI超卖"],
            "pnl": 2.5,
            "verdict_action": "long",
            "price": 95000,
            "volatility": 0.032,
            "funding_rate": 0.0015,
        }
    ]
    system, user = _build_reflection_prompt("tech_agent", records)
    assert "Technical Analysis" in system
    assert "RSI" in user
    assert "MACD" in user
    assert "direction=bullish" in user
    assert "pnl=+2.50" in user
    assert "策略备忘录" in user


async def test_maybe_reflect_skips_when_not_due():
    """Reflection is skipped when cycle_count is not a multiple of every_n_cycles."""
    store = AsyncMock()
    config = ReflectionConfig(every_n_cycles=20)
    result = await maybe_reflect(store, cycle_count=5, config=config)
    assert result == {}
    store.log.assert_not_called()


async def test_maybe_reflect_skips_when_disabled():
    """Reflection is skipped when disabled."""
    store = AsyncMock()
    config = ReflectionConfig(enabled=False)
    result = await maybe_reflect(store, cycle_count=20, config=config)
    assert result == {}


async def test_maybe_reflect_skips_insufficient_data():
    """Reflection is skipped when not enough commits with PnL."""
    store = AsyncMock()
    # Return commits without PnL
    store.log.return_value = [_make_commit(pnl=None) for _ in range(5)]
    config = ReflectionConfig(every_n_cycles=20, min_commits_required=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "reflections.db"
        result = await maybe_reflect(store, cycle_count=20, config=config, db_path=db_path)
        assert result == {}


@patch("cryptotrader.learning.reflect.run_agent_reflection")
async def test_maybe_reflect_runs_and_saves(mock_reflect):
    """Full reflection cycle: LLM is called for each agent, results saved to SQLite."""
    mock_reflect.return_value = "Test memo: signal X works well"

    store = AsyncMock()
    commits = [_make_commit(pnl=float(i)) for i in range(15)]
    store.log.return_value = commits

    config = ReflectionConfig(every_n_cycles=20, min_commits_required=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "reflections.db"
        result = await maybe_reflect(store, cycle_count=20, config=config, db_path=db_path)

        # All 4 agents should have reflections
        assert len(result) == 4
        for agent_id in ("tech_agent", "chain_agent", "news_agent", "macro_agent"):
            assert agent_id in result
            assert result[agent_id] == "Test memo: signal X works well"

        # Verify persisted to SQLite
        loaded = await load_reflections(db_path)
        assert len(loaded) == 4
        assert loaded["tech_agent"] == "Test memo: signal X works well"


@patch("cryptotrader.learning.reflect.run_agent_reflection")
async def test_maybe_reflect_partial_failure(mock_reflect):
    """If one agent's reflection fails, others still succeed."""
    call_count = 0

    async def side_effect(agent_id, records, model):
        nonlocal call_count
        call_count += 1
        if agent_id == "chain_agent":
            raise RuntimeError("LLM timeout")
        return f"Memo for {agent_id}"

    mock_reflect.side_effect = side_effect

    store = AsyncMock()
    store.log.return_value = [_make_commit(pnl=float(i)) for i in range(15)]

    config = ReflectionConfig(every_n_cycles=20, min_commits_required=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "reflections.db"
        result = await maybe_reflect(store, cycle_count=20, config=config, db_path=db_path)

        # 3 agents succeed, 1 fails
        assert len(result) == 3
        assert "chain_agent" not in result
        assert "tech_agent" in result


async def test_reflection_injected_in_experience():
    """Verify _run_agent correctly injects reflection memo into experience."""
    from unittest.mock import MagicMock

    from cryptotrader.models import AgentAnalysis

    # Build a minimal mock analysis result
    mock_analysis = AgentAnalysis(
        agent_id="tech_agent",
        pair="BTC/USDT",
        direction="bullish",
        confidence=0.7,
        reasoning="test",
    )

    mock_agent_cls = MagicMock()
    mock_agent_cls.return_value.analyze = AsyncMock(return_value=mock_analysis)

    state = {
        "data": {
            "snapshot": MagicMock(pair="BTC/USDT"),
            "experience": "Historical context here",
            "agent_corrections": {},
            "agent_reflections": {"tech_agent": "My strategy memo: watch RSI divergence"},
        },
        "metadata": {
            "models": {},
            "analysis_model": "",
        },
    }

    with patch("cryptotrader.agents.tech.TechAgent", mock_agent_cls):
        from cryptotrader.nodes.agents import tech_analyze

        await tech_analyze(state)

    # Check that analyze was called with experience containing the reflection
    call_args = mock_agent_cls.return_value.analyze.call_args
    experience_arg = call_args[0][1]  # second positional arg
    assert "Strategy memo (your own prior self-reflection):" in experience_arg
    assert "watch RSI divergence" in experience_arg
    assert "Historical context here" in experience_arg
