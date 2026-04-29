"""Tests for partial verdict — quick decision from incomplete analyses."""

from __future__ import annotations

import pytest

from cryptotrader.chat.partial_verdict import make_partial_verdict


def test_empty_analyses():
    result = make_partial_verdict({})
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0
    assert result["is_partial"] is True
    assert result["completed_agents"] == []


def test_two_opposite_agents_hold():
    analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.8},
        "news_agent": {"direction": "bearish", "confidence": 0.8},
    }
    result = make_partial_verdict(analyses)
    assert result["action"] == "hold"
    assert result["is_partial"] is True
    assert len(result["completed_agents"]) == 2
    assert len(result["missing_agents"]) == 2


def test_three_same_direction():
    analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.9},
        "chain_agent": {"direction": "bullish", "confidence": 0.7},
        "news_agent": {"direction": "bullish", "confidence": 0.8},
    }
    result = make_partial_verdict(analyses)
    assert result["action"] == "long"
    assert result["is_partial"] is True
    assert result["confidence"] == pytest.approx(0.8, abs=0.01)
    assert "macro_agent" in result["missing_agents"]


def test_majority_bearish():
    analyses = {
        "tech_agent": {"direction": "bearish", "confidence": 0.85},
        "chain_agent": {"direction": "bearish", "confidence": 0.75},
        "macro_agent": {"direction": "bullish", "confidence": 0.5},
    }
    result = make_partial_verdict(analyses)
    assert result["action"] == "short"
    assert result["is_partial"] is True


def test_single_agent():
    analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.9},
    }
    result = make_partial_verdict(analyses)
    assert result["action"] == "long"
    assert result["completed_agents"] == ["tech_agent"]
    assert len(result["missing_agents"]) == 3


def test_position_scale_proportional():
    analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 1.0},
        "chain_agent": {"direction": "bullish", "confidence": 1.0},
    }
    result = make_partial_verdict(analyses)
    assert result["position_scale"] > 0
    assert result["position_scale"] <= 1.0
