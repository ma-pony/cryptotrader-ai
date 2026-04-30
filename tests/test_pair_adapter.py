"""Tests for transient pair_adapter (spec 013, Phase 3a, T012)."""

from __future__ import annotations

import pytest

from cryptotrader.pair import Pair
from cryptotrader.pair_adapter import from_pair, to_pair


class TestToPair:
    def test_str_input_returns_pair(self) -> None:
        p = to_pair("BTC/USDT")
        assert isinstance(p, Pair)
        assert p.canonical() == "BTC/USDT"

    def test_pair_input_idempotent(self) -> None:
        original = Pair.parse("BTC/USDT:USDT")
        assert to_pair(original) is original

    def test_swap_str(self) -> None:
        p = to_pair("BTC/USDT:USDT")
        assert p.market_type == "swap"
        assert p.settle == "USDT"

    def test_invalid_str_raises(self) -> None:
        with pytest.raises(ValueError):
            to_pair("BTCUSDT")  # no slash


class TestFromPair:
    def test_pair_input_returns_canonical(self) -> None:
        assert from_pair(Pair.parse("BTC/USDT")) == "BTC/USDT"
        assert from_pair(Pair.parse("BTC/USDT:USDT")) == "BTC/USDT:USDT"

    def test_str_input_round_trips(self) -> None:
        assert from_pair("BTC/USDT") == "BTC/USDT"
        assert from_pair("BTC/USDT:USDT") == "BTC/USDT:USDT"

    def test_invalid_str_raises(self) -> None:
        with pytest.raises(ValueError):
            from_pair("not-a-pair")


class TestRoundTrip:
    @pytest.mark.parametrize(
        "canonical",
        [
            "BTC/USDT",
            "ETH/USDT",
            "BTC/USDT:USDT",
            "BTC/USD:BTC",
            "ETH/USD:ETH",
        ],
    )
    def test_str_to_pair_to_str(self, canonical: str) -> None:
        assert from_pair(to_pair(canonical)) == canonical

    @pytest.mark.parametrize(
        "canonical",
        [
            "BTC/USDT",
            "BTC/USDT:USDT",
        ],
    )
    def test_pair_to_str_to_pair(self, canonical: str) -> None:
        original = Pair.parse(canonical)
        assert to_pair(from_pair(original)) == original
