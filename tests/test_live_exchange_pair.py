"""LiveExchange canonical pair handling at the ccxt boundary.

Regression for 2026-04-30 close-on-flat bug: ccxt reports OKX perp positions
keyed `BTC/USDT:USDT` but the rest of the codebase looks up `BTC/USDT`. The
canonicalization at this single boundary keeps every downstream
`positions.get(pair)` matching.

Tracked properly by spec 013-pair-value-object; this is the band-aid that
spec's Phase 0 documents.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.execution.exchange import LiveExchange


def _make_exchange_with_markets(market_meta: dict) -> LiveExchange:
    """Construct a LiveExchange whose ccxt._exchange is patched with
    a stub returning the given market metadata for any symbol."""
    with patch("ccxt.async_support.okx") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.load_markets = AsyncMock()
        mock_inst.market = MagicMock(side_effect=lambda sym: market_meta.get(sym, {}))
        mock_cls.return_value = mock_inst
        return LiveExchange("okx", "k", "s", sandbox=True, passphrase="p")


@pytest.mark.asyncio
async def test_get_positions_canonicalizes_perp_swap_symbol():
    """OKX perp `BTC/USDT:USDT` -> project canonical `BTC/USDT` via market metadata."""
    market_meta = {
        "BTC/USDT:USDT": {"base": "BTC", "quote": "USDT", "settle": "USDT", "swap": True},
    }
    ex = _make_exchange_with_markets(market_meta)

    raw_positions = [
        {
            "symbol": "BTC/USDT:USDT",
            "contracts": 0.02,
            "side": "long",
            "entryPrice": 84708.8,
            "unrealizedPnl": -1.68,
            "liquidationPrice": None,
        }
    ]
    ex._exchange.fetch_positions = AsyncMock(return_value=raw_positions)
    ex._markets_loaded = True

    positions = await ex.get_positions()
    assert "BTC/USDT" in positions, "perp symbol must be canonicalized to spot form"
    assert "BTC/USDT:USDT" not in positions, "raw ccxt symbol must not leak"
    assert positions["BTC/USDT"]["amount"] == 0.02
    assert positions["BTC/USDT"]["avg_price"] == 84708.8


@pytest.mark.asyncio
async def test_get_positions_keeps_spot_symbol_unchanged():
    """Spot symbols pass through unchanged."""
    market_meta = {
        "ETH/USDT": {"base": "ETH", "quote": "USDT", "spot": True},
    }
    ex = _make_exchange_with_markets(market_meta)
    ex._exchange.fetch_positions = AsyncMock(
        return_value=[{"symbol": "ETH/USDT", "contracts": 1.5, "side": "long", "entryPrice": 2300.0}]
    )
    ex._markets_loaded = True

    positions = await ex.get_positions()
    assert positions["ETH/USDT"]["amount"] == 1.5


@pytest.mark.asyncio
async def test_get_positions_falls_back_to_string_split_when_market_unknown():
    """If ccxt has no market metadata for the symbol (rare), fall back to
    `BASE/QUOTE:SETTLE` split. Never crash."""
    ex = _make_exchange_with_markets({})  # empty -> exchange.market() returns {}
    ex._exchange.fetch_positions = AsyncMock(
        return_value=[{"symbol": "DOGE/USDT:USDT", "contracts": 100.0, "side": "long", "entryPrice": 0.15}]
    )
    ex._markets_loaded = True

    positions = await ex.get_positions()
    assert "DOGE/USDT" in positions
    assert positions["DOGE/USDT"]["amount"] == 100.0


@pytest.mark.asyncio
async def test_get_positions_collision_keeps_larger_size_warns():
    """Two ccxt symbols collapsing to the same canonical pair (e.g. spot +
    perp on the same market) keep the larger absolute size and emit a
    warning. Tests rely on caplog rather than parsing stderr."""
    import logging

    market_meta = {
        "BTC/USDT": {"base": "BTC", "quote": "USDT", "spot": True},
        "BTC/USDT:USDT": {"base": "BTC", "quote": "USDT", "settle": "USDT", "swap": True},
    }
    ex = _make_exchange_with_markets(market_meta)
    ex._exchange.fetch_positions = AsyncMock(
        return_value=[
            {"symbol": "BTC/USDT", "contracts": 0.01, "side": "long", "entryPrice": 80000.0},
            {"symbol": "BTC/USDT:USDT", "contracts": 0.05, "side": "long", "entryPrice": 84000.0},
        ]
    )
    ex._markets_loaded = True

    with patch.object(logging.getLogger("cryptotrader.execution.exchange"), "warning") as mock_warn:
        positions = await ex.get_positions()

    assert positions["BTC/USDT"]["amount"] == 0.05, "larger absolute size wins"
    assert mock_warn.called, "collision must emit a warning"
