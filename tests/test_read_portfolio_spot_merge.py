"""Tests for read_portfolio_from_exchange spot-balance merging.

Production bug (2026-05-01): after a spot ETH long fill, ``positions``
returned by ``read_portfolio_from_exchange`` was empty even though the
exchange held ETH and the DB row was correct. Cause: the function only
unioned ``exchange.get_positions()`` (ccxt fetchPositions = perps only),
silently dropping non-USDT spot balances. API ``/portfolio/snapshot``
showed cash drained from $8540 to $4697 with positions=[], making it
look like the bot lost cash for no reason.

These tests pin down:
  - Spot non-USDT balances become positions, priced via exchange.fetch_ticker
  - USDT remains cash (not a position)
  - If a perp position already exists for the same pair, perp wins (not overwritten by spot)
  - Ticker unavailable → spot row included with avg_price=0 (visible but not valued)
  - total_value reflects spot value at ticker price
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


def _build_state(pair: str = "ETH/USDT") -> dict:
    return {
        "metadata": {"engine": "live", "exchange_id": "okx", "pair": pair},
        "data": {"snapshot_summary": {"price": 2300.0}},
    }


def _make_exchange(
    *,
    balances: dict[str, float],
    perps: dict[str, dict] | None = None,
    ticker_prices: dict[str, float] | None = None,
    ticker_raises: bool = False,
):
    ex = AsyncMock()
    ex.get_balance = AsyncMock(return_value=dict(balances))
    ex.get_positions = AsyncMock(return_value=dict(perps or {}))

    async def _fetch(symbol: str) -> dict[str, Any]:
        if ticker_raises:
            raise RuntimeError("network")
        if ticker_prices is None or symbol not in ticker_prices:
            raise ValueError(f"unknown {symbol}")
        return {"last": ticker_prices[symbol], "symbol": symbol}

    ex.fetch_ticker = _fetch
    return ex


@pytest.mark.asyncio
async def test_spot_eth_balance_appears_as_position():
    """ETH spot balance is surfaced as a position with ticker-derived avg_price."""
    from cryptotrader.portfolio.manager import read_portfolio_from_exchange

    ex = _make_exchange(
        balances={"USDT": 4697.51, "ETH": 1.68},
        perps={},
        ticker_prices={"ETH/USDT": 2285.0},
    )
    with patch("cryptotrader.nodes.execution._get_exchange", AsyncMock(return_value=(ex, None))):
        result = await read_portfolio_from_exchange(_build_state("ETH/USDT"))

    assert result is not None
    assert result["cash"] == pytest.approx(4697.51)
    assert "ETH/USDT" in result["positions"]
    eth = result["positions"]["ETH/USDT"]
    assert eth["amount"] == pytest.approx(1.68)
    assert eth["avg_price"] == pytest.approx(2285.0)
    assert eth["side"] == "long"
    # total = cash + 1.68 * 2285
    assert result["total_value"] == pytest.approx(4697.51 + 1.68 * 2285.0)


@pytest.mark.asyncio
async def test_usdt_balance_stays_as_cash_not_position():
    from cryptotrader.portfolio.manager import read_portfolio_from_exchange

    ex = _make_exchange(balances={"USDT": 10000.0}, perps={}, ticker_prices={})
    with patch("cryptotrader.nodes.execution._get_exchange", AsyncMock(return_value=(ex, None))):
        result = await read_portfolio_from_exchange(_build_state("BTC/USDT"))

    assert result is not None
    assert result["cash"] == pytest.approx(10000.0)
    assert result["positions"] == {}
    assert result["total_value"] == pytest.approx(10000.0)


@pytest.mark.asyncio
async def test_perp_position_not_overwritten_by_spot_balance():
    """If a perp position exists for ETH/USDT:USDT, do NOT clobber it with spot ETH/USDT."""
    from cryptotrader.portfolio.manager import read_portfolio_from_exchange

    perps = {"ETH/USDT:USDT": {"amount": 1.0, "avg_price": 2300.0, "side": "long"}}
    ex = _make_exchange(
        balances={"USDT": 5000.0, "ETH": 0.5},  # spot ETH coexists
        perps=perps,
        ticker_prices={"ETH/USDT": 2285.0},
    )
    with patch("cryptotrader.nodes.execution._get_exchange", AsyncMock(return_value=(ex, None))):
        result = await read_portfolio_from_exchange(_build_state("ETH/USDT:USDT"))

    assert result is not None
    # Both spot ETH/USDT and perp ETH/USDT:USDT should appear (different pairs, both real)
    assert result["positions"]["ETH/USDT:USDT"]["avg_price"] == 2300.0  # perp untouched
    assert result["positions"]["ETH/USDT"]["amount"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_spot_balance_without_ticker_uses_zero_price():
    """Ticker fails → row still appears so caller sees the balance, but priced at 0."""
    from cryptotrader.portfolio.manager import read_portfolio_from_exchange

    ex = _make_exchange(
        balances={"USDT": 1000.0, "OBSCURE": 100.0},
        perps={},
        ticker_raises=True,
    )
    with patch("cryptotrader.nodes.execution._get_exchange", AsyncMock(return_value=(ex, None))):
        result = await read_portfolio_from_exchange(_build_state("BTC/USDT"))

    assert result is not None
    assert "OBSCURE/USDT" in result["positions"]
    assert result["positions"]["OBSCURE/USDT"]["avg_price"] == 0.0
    # total_value excludes the unpriced spot
    assert result["total_value"] == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_multiple_spot_balances_all_appear():
    from cryptotrader.portfolio.manager import read_portfolio_from_exchange

    ex = _make_exchange(
        balances={"USDT": 1000.0, "ETH": 1.0, "BTC": 0.05},
        perps={},
        ticker_prices={"ETH/USDT": 2200.0, "BTC/USDT": 77000.0},
    )
    with patch("cryptotrader.nodes.execution._get_exchange", AsyncMock(return_value=(ex, None))):
        result = await read_portfolio_from_exchange(_build_state("ETH/USDT"))

    assert result is not None
    assert "ETH/USDT" in result["positions"]
    assert "BTC/USDT" in result["positions"]
    assert result["total_value"] == pytest.approx(1000.0 + 1.0 * 2200.0 + 0.05 * 77000.0)


@pytest.mark.asyncio
async def test_zero_amount_balance_is_skipped():
    from cryptotrader.portfolio.manager import read_portfolio_from_exchange

    ex = _make_exchange(
        balances={"USDT": 1000.0, "ETH": 0.0},
        perps={},
        ticker_prices={"ETH/USDT": 2200.0},
    )
    with patch("cryptotrader.nodes.execution._get_exchange", AsyncMock(return_value=(ex, None))):
        result = await read_portfolio_from_exchange(_build_state("ETH/USDT"))

    assert result is not None
    assert "ETH/USDT" not in result["positions"]


@pytest.mark.asyncio
async def test_dust_balance_is_skipped():
    """Post-close exchange residue (e.g. ETH=2.91e-07) must NOT appear in API.

    Production observation 2026-05-02: OKX kept dust on the balance after a
    successful market sell. Without this filter, /api/portfolio/snapshot
    surfaced microscopic phantom long positions to the AI.
    """
    from cryptotrader.portfolio.manager import read_portfolio_from_exchange

    ex = _make_exchange(
        balances={"USDT": 1000.0, "ETH": 2.91e-07, "BTC": 3.19e-09},
        perps={},
        ticker_prices={"ETH/USDT": 2200.0, "BTC/USDT": 77000.0},
    )
    with patch("cryptotrader.nodes.execution._get_exchange", AsyncMock(return_value=(ex, None))):
        result = await read_portfolio_from_exchange(_build_state("BTC/USDT"))

    assert result is not None
    assert "ETH/USDT" not in result["positions"]
    assert "BTC/USDT" not in result["positions"]
    assert result["total_value"] == pytest.approx(1000.0)
