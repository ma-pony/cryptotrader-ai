"""Pair value object — round-trip + ccxt symbol shapes + invariant violations.

Spec: specs/013-pair-value-object/spec.md (FR-001~010)
Contract: specs/013-pair-value-object/contracts/pair_api.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cryptotrader.pair import Pair

# ── Constructor invariants ─────────────────────────────────────────────────


class TestInvariants:
    def test_empty_base_raises(self) -> None:
        with pytest.raises(ValueError, match="base"):
            Pair(base="", quote="USDT", ccxt_symbol="/USDT")

    def test_empty_quote_raises(self) -> None:
        with pytest.raises(ValueError, match="quote"):
            Pair(base="BTC", quote="", ccxt_symbol="BTC/")

    def test_missing_slash_in_ccxt_symbol_raises(self) -> None:
        with pytest.raises(ValueError, match="must contain '/'"):
            Pair(base="BTC", quote="USDT", ccxt_symbol="BTCUSDT")

    def test_ccxt_symbol_prefix_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            Pair(base="BTC", quote="USDT", ccxt_symbol="ETH/USDT")

    def test_frozen_dataclass_immutable(self) -> None:
        from dataclasses import FrozenInstanceError

        p = Pair.parse("BTC/USDT")
        with pytest.raises(FrozenInstanceError):
            p.base = "ETH"  # type: ignore[misc]

    def test_pair_is_hashable_dict_key(self) -> None:
        p1 = Pair.parse("BTC/USDT")
        p2 = Pair.parse("BTC/USDT")
        d = {p1: 42}
        assert d[p2] == 42


# ── parse() ────────────────────────────────────────────────────────────────


class TestParse:
    def test_spot(self) -> None:
        p = Pair.parse("BTC/USDT")
        assert p.base == "BTC"
        assert p.quote == "USDT"
        assert p.ccxt_symbol == "BTC/USDT"
        assert p.market_type == "spot"
        assert p.settle is None

    def test_linear_perp(self) -> None:
        p = Pair.parse("BTC/USDT:USDT")
        assert p.base == "BTC"
        assert p.quote == "USDT"
        assert p.market_type == "swap"
        assert p.settle == "USDT"

    def test_inverse_perp(self) -> None:
        p = Pair.parse("BTC/USD:BTC")
        assert p.base == "BTC"
        assert p.quote == "USD"
        assert p.market_type == "swap"
        assert p.settle == "BTC"

    def test_linear_future(self) -> None:
        p = Pair.parse("BTC/USDT:USDT-241227")
        assert p.base == "BTC"
        assert p.quote == "USDT"
        assert p.market_type == "future"
        assert p.settle == "USDT"

    def test_inverse_future(self) -> None:
        p = Pair.parse("BTC/USD:BTC-241227")
        assert p.market_type == "future"
        assert p.settle == "BTC"

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="missing '/'"):
            Pair.parse("")

    def test_no_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="missing '/'"):
            Pair.parse("BTCUSDT")

    def test_empty_quote_raises(self) -> None:
        with pytest.raises(ValueError, match="empty base/quote"):
            Pair.parse("BTC/")


# ── from_ccxt() ─────────────────────────────────────────────────────────────


class TestFromCcxt:
    def test_with_full_metadata_spot(self) -> None:
        ex = MagicMock()
        ex.market.return_value = {
            "base": "BTC",
            "quote": "USDT",
            "spot": True,
        }
        p = Pair.from_ccxt(ex, "BTC/USDT")
        assert p.market_type == "spot"
        assert p.canonical() == "BTC/USDT"

    def test_with_full_metadata_swap(self) -> None:
        ex = MagicMock()
        ex.market.return_value = {
            "base": "BTC",
            "quote": "USDT",
            "settle": "USDT",
            "swap": True,
        }
        p = Pair.from_ccxt(ex, "BTC/USDT:USDT")
        assert p.market_type == "swap"
        assert p.settle == "USDT"

    def test_market_lookup_failure_falls_back_to_parse(self) -> None:
        ex = MagicMock()
        ex.market.side_effect = KeyError("unknown symbol")
        p = Pair.from_ccxt(ex, "DOGE/USDT:USDT")
        assert p.market_type == "swap"
        assert p.canonical() == "DOGE/USDT:USDT"

    def test_empty_market_metadata_falls_back_to_parse(self) -> None:
        ex = MagicMock()
        ex.market.return_value = {}
        p = Pair.from_ccxt(ex, "ETH/USDT")
        assert p.market_type == "spot"

    def test_option_market_raises_not_implemented(self) -> None:
        ex = MagicMock()
        ex.market.return_value = {
            "base": "BTC",
            "quote": "USD",
            "option": True,
        }
        with pytest.raises(NotImplementedError, match="Option markets"):
            Pair.from_ccxt(ex, "BTC/USD:BTC-241227-65000-C")


# ── to_ccxt / canonical / display / __str__ ───────────────────────────────


class TestSerialization:
    def test_to_ccxt_equals_canonical_equals_str(self) -> None:
        p = Pair.parse("BTC/USDT:USDT")
        assert p.to_ccxt() == p.canonical() == str(p) == "BTC/USDT:USDT"

    def test_display_spot(self) -> None:
        assert Pair.parse("BTC/USDT").display() == "BTC/USDT"

    def test_display_swap(self) -> None:
        assert Pair.parse("BTC/USDT:USDT").display() == "BTC/USDT (perp)"

    def test_display_future_with_expiry(self) -> None:
        assert Pair.parse("BTC/USDT:USDT-241227").display() == "BTC/USDT (futures 241227)"

    def test_display_future_no_expiry(self) -> None:
        # corner case: future without dash-suffix (shouldn't happen but defensive)
        # market_type would resolve to 'swap' since no dash, so display = 'perp'
        assert Pair.parse("BTC/USDT:USDT").market_type == "swap"


# ── Round-trip (FR-009) ────────────────────────────────────────────────────


class TestRoundTrip:
    @pytest.mark.parametrize(
        "symbol",
        [
            "BTC/USDT",
            "ETH/USDT",
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "BTC/USD:BTC",  # inverse perp
            "BTC/USDT:USDT-241227",  # linear future
            "BTC/USD:BTC-250328",  # inverse future
        ],
    )
    def test_parse_round_trip(self, symbol: str) -> None:
        """parse → canonical → parse yields the same Pair."""
        p1 = Pair.parse(symbol)
        p2 = Pair.parse(p1.canonical())
        assert p1 == p2
        assert p1.canonical() == symbol


# ── ccxt symbol shapes per exchange (FR-010) ──────────────────────────────


class TestCcxtExchangeShapes:
    """Verify Pair handles symbols from real-world ccxt exchanges."""

    @pytest.mark.parametrize(
        ("exchange", "symbol", "expected_market", "expected_settle"),
        [
            ("okx", "BTC/USDT", "spot", None),
            ("okx", "BTC/USDT:USDT", "swap", "USDT"),
            ("okx", "BTC/USD:BTC", "swap", "BTC"),
            ("binance", "BTC/USDT", "spot", None),
            ("binance", "BTC/USDT:USDT", "swap", "USDT"),  # USDT-M
            ("binance", "BTC/USD:BTC", "swap", "BTC"),  # COIN-M
            ("binance", "BTC/USDT:USDT-241227", "future", "USDT"),  # USDT-M delivery
            ("bybit", "BTC/USDT:USDT", "swap", "USDT"),
            ("bybit", "BTC/USD:BTC", "swap", "BTC"),
            ("dydx", "BTC/USDC:USDC", "swap", "USDC"),
        ],
    )
    def test_symbol_shape(self, exchange: str, symbol: str, expected_market: str, expected_settle: str | None) -> None:
        p = Pair.parse(symbol)
        assert p.market_type == expected_market, f"{exchange} {symbol}: market_type"
        assert p.settle == expected_settle, f"{exchange} {symbol}: settle"
        # round-trip
        assert Pair.parse(p.canonical()) == p
