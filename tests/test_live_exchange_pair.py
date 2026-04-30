"""LiveExchange position-key contract per spec 013 Phase 3b.

After spec 013, ccxt's unified symbol IS the project canonical pair (spot
``BTC/USDT``, linear perp ``BTC/USDT:USDT``, inverse perp ``BTC/USD:BTC``).
LiveExchange.get_positions returns ccxt symbols verbatim — no
canonicalization translation. The earlier ``_canonical_pair`` band-aid
(commit d05a0bf) is retired by this phase.

T039 will delete this file once the migration is fully verified.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.execution.exchange import LiveExchange


def _make_exchange() -> LiveExchange:
    with patch("ccxt.async_support.okx") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.load_markets = AsyncMock()
        mock_cls.return_value = mock_inst
        return LiveExchange("okx", "k", "s", sandbox=True, passphrase="p")


@pytest.mark.asyncio
async def test_get_positions_returns_ccxt_perp_symbol_verbatim():
    """OKX perp ``BTC/USDT:USDT`` is the canonical key — no translation."""
    ex = _make_exchange()
    ex._exchange.fetch_positions = AsyncMock(
        return_value=[
            {
                "symbol": "BTC/USDT:USDT",
                "contracts": 0.02,
                "side": "long",
                "entryPrice": 84708.8,
                "unrealizedPnl": -1.68,
                "liquidationPrice": None,
            }
        ]
    )
    ex._markets_loaded = True

    positions = await ex.get_positions()
    assert "BTC/USDT:USDT" in positions, "ccxt unified symbol is the canonical key"
    assert "BTC/USDT" not in positions, "spot form must not appear for a perp position"
    assert positions["BTC/USDT:USDT"]["amount"] == 0.02
    assert positions["BTC/USDT:USDT"]["avg_price"] == 84708.8


@pytest.mark.asyncio
async def test_get_positions_keeps_spot_symbol_unchanged():
    ex = _make_exchange()
    ex._exchange.fetch_positions = AsyncMock(
        return_value=[{"symbol": "ETH/USDT", "contracts": 1.5, "side": "long", "entryPrice": 2300.0}]
    )
    ex._markets_loaded = True

    positions = await ex.get_positions()
    assert positions["ETH/USDT"]["amount"] == 1.5


@pytest.mark.asyncio
async def test_get_positions_returns_inverse_perp_symbol_verbatim():
    """Inverse perp ``BTC/USD:BTC`` survives unchanged."""
    ex = _make_exchange()
    ex._exchange.fetch_positions = AsyncMock(
        return_value=[{"symbol": "BTC/USD:BTC", "contracts": 100.0, "side": "long", "entryPrice": 84000.0}]
    )
    ex._markets_loaded = True

    positions = await ex.get_positions()
    assert "BTC/USD:BTC" in positions
    assert positions["BTC/USD:BTC"]["amount"] == 100.0


@pytest.mark.asyncio
async def test_get_positions_skips_zero_contract_entries():
    ex = _make_exchange()
    ex._exchange.fetch_positions = AsyncMock(
        return_value=[
            {"symbol": "BTC/USDT:USDT", "contracts": 0.0, "side": "long"},
            {"symbol": "ETH/USDT:USDT", "contracts": 0.5, "side": "long", "entryPrice": 2300.0},
        ]
    )
    ex._markets_loaded = True

    positions = await ex.get_positions()
    assert "BTC/USDT:USDT" not in positions
    assert "ETH/USDT:USDT" in positions
