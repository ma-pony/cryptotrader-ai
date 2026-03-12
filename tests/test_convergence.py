"""Tests for compute_consensus_strength in convergence module."""

from __future__ import annotations

from cryptotrader.debate.convergence import compute_consensus_strength


def test_strong_bullish_consensus():
    """All agents bullish with high confidence → high strength."""
    analyses = {
        "tech": {"direction": "bullish", "confidence": 0.9},
        "chain": {"direction": "bullish", "confidence": 0.85},
        "news": {"direction": "bullish", "confidence": 0.8},
        "macro": {"direction": "bullish", "confidence": 0.9},
    }
    strength, mean_score = compute_consensus_strength(analyses)
    assert strength > 0.5
    assert mean_score > 0.8


def test_strong_bearish_consensus():
    """All agents bearish → high strength, negative mean."""
    analyses = {
        "tech": {"direction": "bearish", "confidence": 0.9},
        "chain": {"direction": "bearish", "confidence": 0.85},
        "news": {"direction": "bearish", "confidence": 0.8},
        "macro": {"direction": "bearish", "confidence": 0.9},
    }
    strength, mean_score = compute_consensus_strength(analyses)
    assert strength > 0.5
    assert mean_score < -0.8


def test_shared_confusion():
    """All neutral or very low confidence → low |mean_score|."""
    analyses = {
        "tech": {"direction": "neutral", "confidence": 0.3},
        "chain": {"direction": "neutral", "confidence": 0.2},
        "news": {"direction": "bullish", "confidence": 0.1},
        "macro": {"direction": "bearish", "confidence": 0.1},
    }
    _strength, mean_score = compute_consensus_strength(analyses)
    assert abs(mean_score) < 0.05


def test_high_disagreement():
    """Strong opposing views → low strength."""
    analyses = {
        "tech": {"direction": "bullish", "confidence": 0.9},
        "chain": {"direction": "bearish", "confidence": 0.9},
        "news": {"direction": "bullish", "confidence": 0.8},
        "macro": {"direction": "bearish", "confidence": 0.8},
    }
    strength, _mean_score = compute_consensus_strength(analyses)
    # Mean near zero, high dispersion → low strength
    assert strength < 0.1


def test_single_agent():
    """Single agent → returns (0, 0)."""
    analyses = {"tech": {"direction": "bullish", "confidence": 0.9}}
    strength, mean_score = compute_consensus_strength(analyses)
    assert strength == 0.0
    assert mean_score == 0.0


def test_empty_analyses():
    """Empty → returns (0, 0)."""
    strength, mean_score = compute_consensus_strength({})
    assert strength == 0.0
    assert mean_score == 0.0
