"""Tests for parse_applied() — FR-026 attribution parsing.

T038: bare name, prefix form, ambiguous bare name warning, nonexistent prefix warning.
"""

from __future__ import annotations

import logging

from cryptotrader.learning.memory import parse_applied


def test_bare_name_with_originating_agent():
    """Bare 'applied: name' routes to originating_agent when supplied."""
    result = parse_applied("applied: rsi_divergence", originating_agent="tech")
    assert result == {"tech": ["rsi_divergence"]}


def test_prefix_form_routes_to_correct_agent():
    """'applied: chain::whale_accumulation' routes to chain, ignores originating_agent."""
    result = parse_applied("applied: chain::whale_accumulation", originating_agent="tech")
    assert result == {"chain": ["whale_accumulation"]}


def test_ambiguous_bare_name_warns_and_skips(caplog):
    """Bare name with no originating_agent emits warning and is not included in result."""
    with caplog.at_level(logging.WARNING, logger="cryptotrader.learning.memory"):
        result = parse_applied("applied: some_pattern")
    assert result == {}
    assert "bare" in caplog.text.lower() or "skipping" in caplog.text.lower()


def test_unknown_prefix_warns_and_skips(caplog):
    """'applied: unknown_agent::pattern' emits warning and skips."""
    with caplog.at_level(logging.WARNING, logger="cryptotrader.learning.memory"):
        result = parse_applied("applied: ghost::mystery_pattern")
    assert result == {}
    assert "unknown agent prefix" in caplog.text or "ghost" in caplog.text


def test_multiple_patterns_same_agent():
    """Multiple applied lines for same agent are all collected, deduped."""
    text = "applied: tech::rsi_divergence\napplied: tech::rsi_divergence\napplied: tech::macd_cross"
    result = parse_applied(text)
    assert set(result["tech"]) == {"rsi_divergence", "macd_cross"}
    assert len(result["tech"]) == 2  # deduped


def test_mixed_prefix_and_bare():
    """Mix of prefixed and bare patterns in same text."""
    text = "applied: chain::whale_buy\napplied: bullish_flag"
    result = parse_applied(text, originating_agent="news")
    assert result.get("chain") == ["whale_buy"]
    assert result.get("news") == ["bullish_flag"]


def test_empty_string_returns_empty():
    """Empty reasoning produces empty result."""
    assert parse_applied("") == {}
    assert parse_applied("", originating_agent="tech") == {}
