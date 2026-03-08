"""Tests for previously untested pure functions."""

from unittest.mock import patch

from cryptotrader.data.news import _score_headlines, _score_text, _score_texts_finbert
from cryptotrader.journal.search import _within_range

# ── News sentiment scoring ──


def test_score_text_positive():
    assert _score_text("Bitcoin rally surge breakout") > 0


def test_score_text_negative():
    assert _score_text("crypto crash plunge hack ban") < 0


def test_score_text_neutral():
    assert _score_text("the weather is nice today") == 0.0


def test_score_text_mixed():
    score = _score_text("rally crash")
    assert score == 0.0  # 1 pos, 1 neg → balanced


def test_score_text_empty():
    assert _score_text("") == 0.0


# ── FinBERT / headline scoring ──


def test_score_texts_finbert_returns_valid_range():
    """FinBERT returns a score in [-1, 1] range (or 0.0 if unavailable)."""
    score = _score_texts_finbert(["Bitcoin surges to new high"])
    assert -1.0 <= score <= 1.0


def test_score_texts_finbert_empty():
    assert _score_texts_finbert([]) == 0.0


def test_score_headlines_falls_back_to_keywords():
    """When FinBERT unavailable, _score_headlines uses keyword fallback."""
    with (
        patch("cryptotrader.data.news._score_texts_finbert", return_value=0.0),
        patch("cryptotrader.data.news._finbert_available", False),
    ):
        score = _score_headlines(["Bitcoin rally surge breakout"])
        assert score > 0


def test_score_headlines_empty():
    assert _score_headlines([]) == 0.0


def test_score_headlines_uses_finbert_when_available():
    """When FinBERT is available, use its score even if 0.0."""
    with (
        patch("cryptotrader.data.news._finbert_available", True),
        patch("cryptotrader.data.news._score_texts_finbert", return_value=0.3),
    ):
        score = _score_headlines(["some headline"])
        assert score == 0.3


# ── Journal search similarity ──


def test_within_range_both_zero():
    assert _within_range(0.0, 0.0) is True


def test_within_range_both_near_zero():
    assert _within_range(0.0005, 0.0003) is True


def test_within_range_one_zero():
    assert _within_range(0.0, 0.5) is False
    assert _within_range(0.5, 0.0) is False


def test_within_range_similar():
    assert _within_range(0.1, 0.12) is True  # 20% diff < 50%


def test_within_range_too_far():
    assert _within_range(0.1, 0.5) is False  # 400% diff > 50%


def test_within_range_negative():
    assert _within_range(-0.01, -0.012) is True
