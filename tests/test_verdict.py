"""Tests for verdict generation and divergence calculation."""

from cryptotrader.debate.convergence import check_convergence, compute_divergence
from cryptotrader.debate.verdict import _format_constraints, _normalize_action, make_verdict_rules
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


# ── _normalize_action tests ──


def test_normalize_action_passthrough():
    assert _normalize_action("long") == "long"
    assert _normalize_action("short") == "short"
    assert _normalize_action("hold") == "hold"


def test_normalize_action_aliases():
    assert _normalize_action("buy") == "long"
    assert _normalize_action("bullish") == "long"
    assert _normalize_action("sell") == "short"
    assert _normalize_action("bearish") == "short"


def test_normalize_action_unknown():
    assert _normalize_action("maybe") == "hold"
    assert _normalize_action("") == "hold"
    assert _normalize_action("  LONG  ") == "long"


# ── _format_constraints tests ──


def test_format_constraints_empty():
    assert _format_constraints({}) == "No risk constraints available."


def test_format_constraints_circuit_breaker():
    result = _format_constraints({"circuit_breaker_active": True})
    assert "CIRCUIT BREAKER" in result


def test_format_constraints_no_circuit_breaker():
    result = _format_constraints({"circuit_breaker_active": False})
    assert "CIRCUIT BREAKER" not in result


def test_format_constraints_exhausted_loss():
    result = _format_constraints({"daily_loss_remaining_pct": 0.0})
    assert "EXHAUSTED" in result


def test_format_constraints_funding_rate_elevated():
    result = _format_constraints({"funding_rate": 0.001})
    assert "ELEVATED" in result


def test_format_constraints_funding_rate_negative():
    result = _format_constraints({"funding_rate": -0.0005})
    assert "NEGATIVE" in result


def test_format_constraints_full():
    result = _format_constraints(
        {
            "max_position_pct": 0.1,
            "remaining_exposure_pct": 0.3,
            "drawdown_current": 0.05,
            "max_drawdown_pct": 0.10,
            "volatility": 0.025,
        }
    )
    assert "10%" in result
    assert "30%" in result
    assert "drawdown" in result.lower()
    assert "volatility" in result.lower()


# ── make_verdict_rules tests ──


def test_rules_verdict_clear_bullish():
    analyses = {
        "a": {"direction": "bullish", "confidence": 0.9},
        "b": {"direction": "bullish", "confidence": 0.8},
    }
    v = make_verdict_rules(analyses)
    assert v.action == "long"
    assert v.confidence > 0.5


def test_rules_verdict_ambiguous_hold():
    analyses = {
        "a": {"direction": "bullish", "confidence": 0.5},
        "b": {"direction": "bearish", "confidence": 0.5},
    }
    v = make_verdict_rules(analyses)
    assert v.action == "hold"


def test_rules_verdict_thesis_fields():
    analyses = {"a": {"direction": "bullish", "confidence": 0.9}}
    v = make_verdict_rules(analyses)
    assert v.thesis == ""  # rules verdict doesn't set thesis
    assert v.invalidation == ""
