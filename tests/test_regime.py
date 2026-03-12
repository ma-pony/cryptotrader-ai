"""Tests for regime tagging and overlap."""

import pytest

from cryptotrader.config import RegimeThresholdsConfig
from cryptotrader.learning.regime import regime_overlap, tag_regime


@pytest.fixture
def thresholds():
    return RegimeThresholdsConfig()


class TestTagRegime:
    def test_high_funding(self, thresholds):
        tags = tag_regime({"funding_rate": 0.0005}, thresholds)
        assert "high_funding" in tags

    def test_negative_funding(self, thresholds):
        tags = tag_regime({"funding_rate": -0.0002}, thresholds)
        assert "negative_funding" in tags

    def test_normal_funding(self, thresholds):
        tags = tag_regime({"funding_rate": 0.0001}, thresholds)
        assert "high_funding" not in tags
        assert "negative_funding" not in tags

    def test_high_vol(self, thresholds):
        tags = tag_regime({"volatility": 0.04}, thresholds)
        assert "high_vol" in tags

    def test_low_vol(self, thresholds):
        tags = tag_regime({"volatility": 0.005}, thresholds)
        assert "low_vol" in tags

    def test_trending_up(self, thresholds):
        tags = tag_regime({"price_change_7d": 0.08}, thresholds)
        assert "trending_up" in tags

    def test_trending_down(self, thresholds):
        tags = tag_regime({"price_change_7d": -0.08}, thresholds)
        assert "trending_down" in tags

    def test_extreme_fear(self, thresholds):
        tags = tag_regime({"fear_greed_index": 15}, thresholds)
        assert "extreme_fear" in tags

    def test_extreme_greed(self, thresholds):
        tags = tag_regime({"fear_greed_index": 85}, thresholds)
        assert "extreme_greed" in tags

    def test_no_tags_on_neutral(self, thresholds):
        tags = tag_regime({"funding_rate": 0.0001, "volatility": 0.015}, thresholds)
        assert tags == []

    def test_multiple_tags(self, thresholds):
        tags = tag_regime(
            {
                "funding_rate": 0.0005,
                "volatility": 0.04,
                "price_change_7d": 0.08,
                "fear_greed_index": 85,
            },
            thresholds,
        )
        assert set(tags) == {"high_funding", "high_vol", "trending_up", "extreme_greed"}

    def test_missing_fields(self, thresholds):
        tags = tag_regime({}, thresholds)
        # With all defaults at 0.0, volatility=0.0 < low_vol threshold triggers low_vol
        assert "high_funding" not in tags
        assert "trending_up" not in tags
        assert "extreme_fear" not in tags

    def test_price_change_none(self, thresholds):
        tags = tag_regime({"price_change_7d": None}, thresholds)
        assert "trending_up" not in tags
        assert "trending_down" not in tags


class TestRegimeOverlap:
    def test_identical(self):
        assert regime_overlap(["a", "b"], ["a", "b"]) == 1.0

    def test_disjoint(self):
        assert regime_overlap(["a"], ["b"]) == 0.0

    def test_partial(self):
        assert regime_overlap(["a", "b"], ["a", "c"]) == pytest.approx(1 / 3)

    def test_both_empty(self):
        assert regime_overlap([], []) == 0.0

    def test_one_empty(self):
        assert regime_overlap(["a"], []) == 0.0
