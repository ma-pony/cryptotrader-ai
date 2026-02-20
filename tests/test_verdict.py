"""Tests for verdict generation and divergence calculation."""

from cryptotrader.debate.convergence import compute_divergence, check_convergence
from cryptotrader.debate.verdict import make_verdict_weighted as make_verdict


def test_divergence_unanimous_bullish():
    analyses = {
        "a": {"direction": "bullish", "confidence": 0.8},
        "b": {"direction": "bullish", "confidence": 0.8},
    }
    assert compute_divergence(analyses) == 0.0


def test_divergence_split():
    analyses = {
        "a": {"direction": "bullish", "confidence": 0.8},
        "b": {"direction": "bearish", "confidence": 0.8},
    }
    d = compute_divergence(analyses)
    assert d > 0.5


def test_divergence_single_agent():
    assert compute_divergence({"a": {"direction": "bullish", "confidence": 1.0}}) == 0.0


def test_convergence_stable():
    assert check_convergence([0.5, 0.5], 0.5)


def test_convergence_not_stable():
    assert not check_convergence([0.5], 0.8)


def test_verdict_bullish():
    analyses = {
        "a": {"direction": "bullish", "confidence": 0.8},
        "b": {"direction": "bullish", "confidence": 0.6},
    }
    v = make_verdict(analyses, divergence=0.1)
    assert v.action == "long"
    assert v.confidence > 0


def test_verdict_bearish():
    analyses = {
        "a": {"direction": "bearish", "confidence": 0.9},
        "b": {"direction": "bearish", "confidence": 0.7},
    }
    v = make_verdict(analyses, divergence=0.1)
    assert v.action == "short"


def test_verdict_high_divergence_hold():
    analyses = {
        "a": {"direction": "bullish", "confidence": 0.9},
        "b": {"direction": "bearish", "confidence": 0.9},
    }
    v = make_verdict(analyses, divergence=0.8, divergence_threshold=0.7)
    assert v.action == "hold"


def test_verdict_position_scale():
    analyses = {"a": {"direction": "bullish", "confidence": 0.8}}
    v = make_verdict(analyses, divergence=0.3)
    assert v.position_scale == 0.7
