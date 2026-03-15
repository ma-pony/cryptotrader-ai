"""Tests for debate gate and router."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cryptotrader.nodes.debate import debate_gate, debate_gate_router


def _make_state(analyses):
    """Build minimal state for debate gate tests."""
    return {
        "messages": [],
        "data": {"analyses": analyses},
        "metadata": {"pair": "BTC/USDT", "engine": "paper"},
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }


def _mock_config(skip_debate=True, consensus_threshold=0.5, confusion_threshold=0.05):
    from cryptotrader.config import AppConfig, DebateConfig

    cfg = AppConfig()
    cfg.debate = DebateConfig(
        skip_debate=skip_debate,
        consensus_skip_threshold=consensus_threshold,
        confusion_skip_threshold=confusion_threshold,
    )
    return cfg


@pytest.mark.asyncio
async def test_debate_gate_strong_consensus_skips():
    """Strong bullish consensus → debate skipped."""
    analyses = {
        "tech": {"direction": "bullish", "confidence": 0.9},
        "chain": {"direction": "bullish", "confidence": 0.85},
        "news": {"direction": "bullish", "confidence": 0.8},
        "macro": {"direction": "bullish", "confidence": 0.9},
    }
    state = _make_state(analyses)
    with patch("cryptotrader.config.load_config", return_value=_mock_config()):
        result = await debate_gate(state)
    assert result["data"]["debate_skipped"] is True
    # debate_skip_reason is the structured tag written into state for Dashboard observability
    assert result["data"]["debate_skip_reason"] == "consensus"


@pytest.mark.asyncio
async def test_debate_gate_shared_confusion_skips():
    """All neutral / low confidence → debate skipped (confusion)."""
    analyses = {
        "tech": {"direction": "neutral", "confidence": 0.2},
        "chain": {"direction": "neutral", "confidence": 0.1},
        "news": {"direction": "neutral", "confidence": 0.15},
        "macro": {"direction": "neutral", "confidence": 0.1},
    }
    state = _make_state(analyses)
    with patch("cryptotrader.config.load_config", return_value=_mock_config()):
        result = await debate_gate(state)
    assert result["data"]["debate_skipped"] is True
    # debate_skip_reason is the structured tag written into state for Dashboard observability
    assert result["data"]["debate_skip_reason"] == "confusion"


@pytest.mark.asyncio
async def test_debate_gate_disagreement_continues():
    """High disagreement → debate NOT skipped."""
    analyses = {
        "tech": {"direction": "bullish", "confidence": 0.9},
        "chain": {"direction": "bearish", "confidence": 0.9},
        "news": {"direction": "bullish", "confidence": 0.8},
        "macro": {"direction": "bearish", "confidence": 0.8},
    }
    state = _make_state(analyses)
    with patch("cryptotrader.config.load_config", return_value=_mock_config()):
        result = await debate_gate(state)
    assert result["data"]["debate_skipped"] is False


@pytest.mark.asyncio
async def test_debate_gate_skip_disabled():
    """skip_debate=False → never skips."""
    analyses = {
        "tech": {"direction": "bullish", "confidence": 0.9},
        "chain": {"direction": "bullish", "confidence": 0.9},
        "news": {"direction": "bullish", "confidence": 0.9},
        "macro": {"direction": "bullish", "confidence": 0.9},
    }
    state = _make_state(analyses)
    with patch("cryptotrader.config.load_config", return_value=_mock_config(skip_debate=False)):
        result = await debate_gate(state)
    assert result["data"]["debate_skipped"] is False


def test_debate_gate_router_skip():
    state = {"data": {"debate_skipped": True}}
    assert debate_gate_router(state) == "skip"


def test_debate_gate_router_debate():
    state = {"data": {"debate_skipped": False}}
    assert debate_gate_router(state) == "debate"


def test_debate_gate_router_missing_key():
    state = {"data": {}}
    assert debate_gate_router(state) == "debate"
