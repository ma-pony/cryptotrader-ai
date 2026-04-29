"""Tests for pure trigger condition functions (T007).

Covers all 4 functions in src/cryptotrader/triggers/conditions.py:
- check_price_threshold
- check_pct_change
- check_candle_pattern
- check_funding_rate
"""

from __future__ import annotations

import pytest

from cryptotrader.triggers.conditions import (
    check_candle_pattern,
    check_funding_rate,
    check_pct_change,
    check_price_threshold,
)

# ---------------------------------------------------------------------------
# check_price_threshold
# ---------------------------------------------------------------------------


class TestCheckPriceThreshold:
    @pytest.mark.parametrize(
        ("current_price", "parameters", "expected"),
        [
            # direction=below: price at threshold => True
            (50_000.0, {"direction": "below", "price": 50_000.0}, True),
            # direction=below: price below threshold => True
            (49_999.0, {"direction": "below", "price": 50_000.0}, True),
            # direction=below: price above threshold => False
            (50_001.0, {"direction": "below", "price": 50_000.0}, False),
            # direction=above: price at threshold => True
            (60_000.0, {"direction": "above", "price": 60_000.0}, True),
            # direction=above: price above threshold => True
            (60_001.0, {"direction": "above", "price": 60_000.0}, True),
            # direction=above: price below threshold => False
            (59_999.0, {"direction": "above", "price": 60_000.0}, False),
            # zero threshold => always False
            (0.0, {"direction": "below", "price": 0}, False),
            # negative threshold => always False
            (100.0, {"direction": "below", "price": -1}, False),
            # unknown direction => False
            (100.0, {"direction": "sideways", "price": 100}, False),
            # default direction (below) used when key absent
            (99.0, {"price": 100.0}, True),
            # current price is zero with threshold > 0, direction below => True
            (0.0, {"direction": "below", "price": 1.0}, True),
        ],
    )
    def test_price_threshold(self, current_price: float, parameters: dict, expected: bool) -> None:
        assert check_price_threshold(current_price, parameters) is expected

    def test_missing_price_key_returns_false(self) -> None:
        # No "price" key -> threshold defaults to 0 -> always False
        assert check_price_threshold(100.0, {"direction": "below"}) is False

    def test_price_as_string_coerced(self) -> None:
        # parameters may contain string values from JSON deserialization
        assert check_price_threshold(50.0, {"direction": "below", "price": "100"}) is True


# ---------------------------------------------------------------------------
# check_pct_change
# ---------------------------------------------------------------------------


class TestCheckPctChange:
    @pytest.mark.parametrize(
        ("current_price", "reference_price", "parameters", "expected"),
        [
            # exactly at threshold => True
            (103.0, 100.0, {"threshold_pct": 3.0}, True),
            # above threshold => True
            (110.0, 100.0, {"threshold_pct": 5.0}, True),
            # below threshold => False
            (101.0, 100.0, {"threshold_pct": 5.0}, False),
            # downward move exceeds threshold => True (abs value)
            (90.0, 100.0, {"threshold_pct": 5.0}, True),
            # zero threshold => always False (threshold <= 0)
            (200.0, 100.0, {"threshold_pct": 0}, False),
            # negative threshold => False
            (200.0, 100.0, {"threshold_pct": -1.0}, False),
            # zero reference price => False (div by zero guard)
            (100.0, 0.0, {"threshold_pct": 1.0}, False),
            # boundary: exactly 0% change, threshold 0.001% => False
            (100.0, 100.0, {"threshold_pct": 0.001}, False),
        ],
    )
    def test_pct_change(
        self,
        current_price: float,
        reference_price: float,
        parameters: dict,
        expected: bool,
    ) -> None:
        assert check_pct_change(current_price, reference_price, parameters) is expected

    def test_missing_threshold_pct_defaults_false(self) -> None:
        # threshold_pct not provided defaults to 0 => always False
        assert check_pct_change(200.0, 100.0, {}) is False


# ---------------------------------------------------------------------------
# check_candle_pattern
# ---------------------------------------------------------------------------


_BEARISH_3 = [
    {"open": 100.0, "close": 98.0},
    {"open": 98.0, "close": 96.0},
    {"open": 96.0, "close": 94.0},
]
_BULLISH_3 = [
    {"open": 94.0, "close": 96.0},
    {"open": 96.0, "close": 98.0},
    {"open": 98.0, "close": 100.0},
]
_MIXED_3 = [
    {"open": 100.0, "close": 98.0},
    {"open": 98.0, "close": 100.0},
    {"open": 100.0, "close": 98.0},
]


class TestCheckCandlePattern:
    @pytest.mark.parametrize(
        ("candles", "parameters", "expected"),
        [
            # 3 bearish candles with count=3 => True
            (_BEARISH_3, {"candle_count": 3, "direction": "bearish"}, True),
            # 3 bullish candles with count=3 => True
            (_BULLISH_3, {"candle_count": 3, "direction": "bullish"}, True),
            # mixed candles bearish check => False
            (_MIXED_3, {"candle_count": 3, "direction": "bearish"}, False),
            # mixed candles bullish check => False
            (_MIXED_3, {"candle_count": 3, "direction": "bullish"}, False),
            # fewer candles than count => False
            (_BEARISH_3[:2], {"candle_count": 3, "direction": "bearish"}, False),
            # empty candles => False
            ([], {"candle_count": 3, "direction": "bearish"}, False),
            # candles >= count: only last N checked (last 2 of 3 bearish => True)
            (
                _BEARISH_3,
                {"candle_count": 2, "direction": "bearish"},
                True,
            ),
            # last N not all bearish (mixed at end) => False
            (
                [*_BULLISH_3[:2], {"open": 100.0, "close": 98.0}],
                {"candle_count": 3, "direction": "bullish"},
                False,
            ),
            # unknown direction => False
            (_BEARISH_3, {"candle_count": 3, "direction": "sideways"}, False),
            # count defaults to 3 when absent
            (_BEARISH_3, {"direction": "bearish"}, True),
            # exactly at boundary: 1 candle count with 1 bearish candle
            ([{"open": 100.0, "close": 99.0}], {"candle_count": 1, "direction": "bearish"}, True),
            # doji (open == close) is NOT bearish (close < open required)
            ([{"open": 100.0, "close": 100.0}], {"candle_count": 1, "direction": "bearish"}, False),
            # doji is NOT bullish (close > open required)
            ([{"open": 100.0, "close": 100.0}], {"candle_count": 1, "direction": "bullish"}, False),
        ],
    )
    def test_candle_pattern(self, candles: list, parameters: dict, expected: bool) -> None:
        assert check_candle_pattern(candles, parameters) is expected

    def test_extra_candles_uses_only_last_n(self) -> None:
        """With 5 candles and count=3, only the last 3 matter."""
        candles = [
            {"open": 100.0, "close": 102.0},  # bullish (ignored)
            {"open": 102.0, "close": 104.0},  # bullish (ignored)
            {"open": 104.0, "close": 103.0},  # bearish
            {"open": 103.0, "close": 102.0},  # bearish
            {"open": 102.0, "close": 101.0},  # bearish
        ]
        assert check_candle_pattern(candles, {"candle_count": 3, "direction": "bearish"}) is True
        assert check_candle_pattern(candles, {"candle_count": 5, "direction": "bearish"}) is False


# ---------------------------------------------------------------------------
# check_funding_rate
# ---------------------------------------------------------------------------


class TestCheckFundingRate:
    @pytest.mark.parametrize(
        ("funding_rate", "parameters", "expected"),
        [
            # positive rate exceeds threshold => True
            (0.003, {"threshold_pct": 0.1}, True),
            # negative rate absolute value exceeds threshold => True
            (-0.003, {"threshold_pct": 0.1}, True),
            # exactly at threshold => True
            (0.001, {"threshold_pct": 0.1}, True),
            # below threshold => False
            (0.0005, {"threshold_pct": 0.1}, False),
            # zero funding rate => False
            (0.0, {"threshold_pct": 0.1}, False),
            # zero threshold => always False (threshold <= 0)
            (0.01, {"threshold_pct": 0}, False),
            # negative threshold => False
            (0.01, {"threshold_pct": -0.1}, False),
            # large positive rate => True
            (0.01, {"threshold_pct": 0.5}, True),
        ],
    )
    def test_funding_rate(self, funding_rate: float, parameters: dict, expected: bool) -> None:
        assert check_funding_rate(funding_rate, parameters) is expected

    def test_missing_threshold_pct_defaults_false(self) -> None:
        # No threshold_pct key -> defaults to 0 -> always False
        assert check_funding_rate(0.01, {}) is False

    def test_threshold_pct_as_string(self) -> None:
        # String "0.1" should be coerced to float
        assert check_funding_rate(0.003, {"threshold_pct": "0.1"}) is True
