"""US1 (spec 013) — perp close flow integration test.

Verifies that with state.metadata.pair = ccxt unified perp symbol
(``BTC/USDT:USDT``), the verdict→execution pipeline:

1. _build_close_order looks up positions keyed by the same canonical
2. The resulting Order.pair is the canonical str (no spot-form translation)
3. LiveExchange.place_order forwards that pair to ccxt.create_order verbatim
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.execution.exchange import LiveExchange
from cryptotrader.models import Order
from cryptotrader.nodes.execution import _build_close_order


def _state_for(pair: str) -> dict[str, Any]:
    return {
        "metadata": {
            "pair": pair,
            "engine": "live",
            "exchange_id": "okx",
            "database_url": "postgresql+asyncpg://stub",
        },
        "data": {"snapshot_summary": {"price": 84500.0}},
    }


@pytest.mark.asyncio
async def test_build_close_order_uses_perp_canonical_key():
    """Position lookup must match the perp ccxt unified symbol."""
    state = _state_for("BTC/USDT:USDT")
    exchange_portfolio = {
        "positions": {
            "BTC/USDT:USDT": {"amount": 0.02, "avg_price": 84708.8},
        },
        "total_value": 1700.0,
        "cash": 1500.0,
    }

    with patch(
        "cryptotrader.nodes.execution.read_portfolio_from_exchange",
        AsyncMock(return_value=exchange_portfolio),
    ):
        order = await _build_close_order("BTC/USDT:USDT", price=84500.0, state=state)

    assert order is not None, "perp position must produce a close order"
    assert isinstance(order, Order)
    assert order.pair == "BTC/USDT:USDT", "Order.pair must stay canonical (ccxt full symbol)"
    assert order.side == "sell", "long position closes via sell"
    assert order.amount == 0.02


@pytest.mark.asyncio
async def test_build_close_order_returns_none_when_no_perp_position():
    """If the canonical key is absent, no order is built (no spot-form fallback)."""
    state = _state_for("BTC/USDT:USDT")
    exchange_portfolio = {
        # Stale spot-form key (the Phase 0 band-aid era leftover) must NOT
        # satisfy a perp pair lookup after Phase 3b.
        "positions": {"BTC/USDT": {"amount": 0.02, "avg_price": 84708.8}},
        "total_value": 1700.0,
        "cash": 1500.0,
    }

    with patch(
        "cryptotrader.nodes.execution.read_portfolio_from_exchange",
        AsyncMock(return_value=exchange_portfolio),
    ):
        order = await _build_close_order("BTC/USDT:USDT", price=84500.0, state=state)

    assert order is None, "spot-form key must not satisfy perp canonical lookup"


@pytest.mark.asyncio
async def test_live_exchange_place_order_forwards_perp_symbol_to_ccxt():
    """LiveExchange.place_order calls create_order with the perp canonical."""
    with patch("ccxt.async_support.okx") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.load_markets = AsyncMock()
        mock_inst.markets = {
            "BTC/USDT:USDT": {"limits": {"amount": {"min": 0.001}}},
        }
        mock_inst.amount_to_precision = MagicMock(side_effect=lambda _sym, a: a)
        mock_inst.price_to_precision = MagicMock(side_effect=lambda _sym, p: p)
        mock_inst.fetch_balance = AsyncMock(
            return_value={"total": {"USDT": 1500.0}, "free": {"USDT": 1500.0}, "used": {}}
        )
        mock_inst.create_order = AsyncMock(
            return_value={"id": "abc", "status": "closed", "filled": 0.02, "price": 84500.0}
        )
        mock_cls.return_value = mock_inst

        ex = LiveExchange("okx", "k", "s", sandbox=True, passphrase="p")
        order = Order(pair="BTC/USDT:USDT", side="sell", amount=0.02, price=84500.0, order_type="market")
        result = await ex.place_order(order)

    assert result["status"] == "closed"
    args, _kwargs = mock_inst.create_order.call_args
    # ccxt create_order positional args: (symbol, type, side, amount, price)
    assert args[0] == "BTC/USDT:USDT", "ccxt receives the perp canonical symbol verbatim"
    assert args[2] == "sell"
    assert args[3] == 0.02


@pytest.mark.asyncio
async def test_live_exchange_perp_balance_check_uses_settle_currency():
    """Perp balance pre-check verifies USDT margin (settle), not BTC asset."""
    with patch("ccxt.async_support.okx") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.load_markets = AsyncMock()
        mock_inst.markets = {"BTC/USDT:USDT": {"limits": {"amount": {"min": 0.001}}}}
        mock_inst.amount_to_precision = MagicMock(side_effect=lambda _sym, a: a)
        mock_inst.price_to_precision = MagicMock(side_effect=lambda _sym, p: p)
        # Zero USDT margin — perp order must reject regardless of side
        mock_inst.fetch_balance = AsyncMock(return_value={"total": {"USDT": 0.0}, "free": {"USDT": 0.0}, "used": {}})
        mock_inst.create_order = AsyncMock()
        mock_cls.return_value = mock_inst

        ex = LiveExchange("okx", "k", "s", sandbox=True, passphrase="p")
        order = Order(pair="BTC/USDT:USDT", side="sell", amount=0.02, price=84500.0, order_type="market")
        with pytest.raises(ValueError, match="USDT margin"):
            await ex.place_order(order)

    mock_inst.create_order.assert_not_called()
