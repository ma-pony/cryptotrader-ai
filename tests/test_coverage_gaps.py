"""Tests for previously untested pure functions."""

from cryptotrader.journal.search import _within_range

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
