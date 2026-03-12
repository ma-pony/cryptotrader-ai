"""Tests for debate round parallelization."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from cryptotrader.nodes.debate import _debate_one_agent, debate_round


def _make_analysis(direction="bullish", confidence=0.8):
    return {
        "direction": direction,
        "confidence": confidence,
        "reasoning": "test reasoning",
        "key_factors": ["factor1"],
        "risk_flags": ["risk1"],
    }


def _make_state():
    return {
        "messages": [],
        "data": {
            "analyses": {
                "tech_agent": _make_analysis("bullish", 0.8),
                "chain_agent": _make_analysis("bearish", 0.7),
                "news_agent": _make_analysis("neutral", 0.5),
                "macro_agent": _make_analysis("bullish", 0.6),
            },
        },
        "metadata": {
            "pair": "BTC/USDT",
            "engine": "paper",
            "debate_model": "test-model",
        },
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }


@pytest.mark.asyncio
async def test_debate_one_agent_success():
    """Single agent debate call returns updated analysis."""
    analysis = _make_analysis()
    others = {"chain_agent": _make_analysis("bearish")}
    response_json = (
        '{"direction": "bullish", "confidence": 0.85, "reasoning": "updated",'
        ' "key_factors": ["f1"], "risk_flags": [], "new_findings": "cross-domain"}'
    )
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=response_json))
    with patch("cryptotrader.nodes.debate.create_llm", return_value=mock_llm):
        aid, result = await _debate_one_agent("tech_agent", analysis, others, "BTC/USDT", "test")
    assert aid == "tech_agent"
    assert result["confidence"] == 0.85
    assert result["new_findings"] == "cross-domain"


@pytest.mark.asyncio
async def test_debate_one_agent_failure_returns_original():
    """If LLM call fails, original analysis is preserved."""
    analysis = _make_analysis()
    others = {"chain_agent": _make_analysis("bearish")}
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM timeout"))
    with patch("cryptotrader.nodes.debate.create_llm", return_value=mock_llm):
        aid, result = await _debate_one_agent("tech_agent", analysis, others, "BTC/USDT", "test")
    assert aid == "tech_agent"
    assert result == analysis


@pytest.mark.asyncio
async def test_debate_round_runs_parallel():
    """debate_round uses asyncio.gather for parallel execution."""
    state = _make_state()
    call_times = []

    async def mock_ainvoke(msgs):
        call_times.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.01)  # simulate async work
        return AIMessage(
            content='{"direction": "bullish", "confidence": 0.8, "reasoning": "r",'
            ' "key_factors": [], "risk_flags": [], "new_findings": ""}'
        )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = mock_ainvoke
    with (
        patch("cryptotrader.nodes.debate.create_llm", return_value=mock_llm),
        patch("cryptotrader.config.load_config") as mock_cfg,
    ):
        mock_cfg.return_value.models.debate = "test"
        mock_cfg.return_value.models.fallback = "fallback"
        result = await debate_round(state)

    assert len(result["data"]["analyses"]) == 4
    assert result["debate_round"] == 1
    # All 4 calls should start at roughly the same time (parallel)
    if len(call_times) == 4:
        time_spread = max(call_times) - min(call_times)
        assert time_spread < 0.05  # all started within 50ms


@pytest.mark.asyncio
async def test_debate_round_partial_failure():
    """If one agent fails in gather, others still succeed."""
    state = _make_state()
    call_count = 0

    async def mock_ainvoke(msgs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise Exception("LLM error")
        return AIMessage(
            content='{"direction": "bullish", "confidence": 0.75, "reasoning": "ok",'
            ' "key_factors": [], "risk_flags": [], "new_findings": ""}'
        )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = mock_ainvoke
    with (
        patch("cryptotrader.nodes.debate.create_llm", return_value=mock_llm),
        patch("cryptotrader.config.load_config") as mock_cfg,
    ):
        mock_cfg.return_value.models.debate = "test"
        mock_cfg.return_value.models.fallback = "fallback"
        result = await debate_round(state)

    # All 4 agents should have results (3 updated + 1 original)
    assert len(result["data"]["analyses"]) == 4
