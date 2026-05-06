"""Regression tests for portfolio sync price-source selection.

Production bug (2026-04-30): when an exchange returns a non-traded asset
balance with no prior cost basis in our DB, the sync code fell back to the
*traded pair's* current_price. So a BTC trade at $77k caused ETH/OKB rows
to be persisted with avg_price=$77k, inflating total_value to ~$7.3M and
poisoning portfolio_snapshots / drawdown for weeks.

These tests pin down the four price-source branches in
``_sync_spot_from_balances`` and ``_sync_derivatives_from_positions``:

  1. pair == traded_pair                  -> current_price
  2. pair != traded_pair, old_avg > 0     -> old_avg (preserve cost basis)
  3. pair != traded_pair, old_avg == 0,
     ticker available                     -> ticker.last
  4. pair != traded_pair, old_avg == 0,
     ticker fails / unavailable           -> 0.0 (DO NOT pollute with current_price)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from cryptotrader.nodes.execution import (
    _sync_derivatives_from_positions,
    _sync_spot_from_balances,
)
from cryptotrader.portfolio.manager import PortfolioManager


@pytest.fixture
def pm():
    return PortfolioManager(None)


def _make_exchange(ticker_responses: dict[str, float] | None = None, raises: bool = False):
    """Build a stub exchange with a fetch_ticker that returns mapped prices.

    ``ticker_responses`` maps pair -> last price. Pairs not in the map cause
    ``fetch_ticker`` to raise ValueError (simulating an unknown symbol).
    """
    ex = AsyncMock()

    async def _fetch(pair: str) -> dict[str, Any]:
        if raises:
            raise RuntimeError("network down")
        if ticker_responses is None or pair not in ticker_responses:
            raise ValueError(f"unknown symbol {pair}")
        return {"last": ticker_responses[pair], "symbol": pair}

    ex.fetch_ticker = _fetch
    return ex


# ── spot path ──


@pytest.mark.asyncio
async def test_spot_traded_pair_uses_current_price(pm):
    ex = _make_exchange()
    balances = {"BTC": 0.5}
    total, seen = await _sync_spot_from_balances(
        pm, balances, old_positions={}, traded_pair="BTC/USDT", current_price=77000.0, exchange=ex
    )
    p = await pm.get_portfolio()
    assert p["positions"]["BTC/USDT"]["avg_price"] == 77000.0
    assert seen == {"BTC/USDT"}
    assert total == pytest.approx(0.5 * 77000.0)


@pytest.mark.asyncio
async def test_spot_non_traded_pair_uses_old_avg_when_present(pm):
    ex = _make_exchange()
    balances = {"BTC": 0.5, "ETH": 1.0}
    old = {"ETH/USDT": {"amount": 1.0, "avg_price": 3500.0}}
    total, _ = await _sync_spot_from_balances(
        pm, balances, old_positions=old, traded_pair="BTC/USDT", current_price=77000.0, exchange=ex
    )
    p = await pm.get_portfolio()
    assert p["positions"]["ETH/USDT"]["avg_price"] == 3500.0  # preserved
    assert p["positions"]["BTC/USDT"]["avg_price"] == 77000.0
    assert total == pytest.approx(0.5 * 77000.0 + 1.0 * 3500.0)


@pytest.mark.asyncio
async def test_spot_non_traded_pair_fetches_ticker_when_old_missing(pm):
    """Plan A core fix: missing cost basis -> fetch real market price, NOT traded pair's price."""
    ex = _make_exchange({"ETH/USDT": 2200.0})
    balances = {"BTC": 0.5, "ETH": 1.0}
    total, _ = await _sync_spot_from_balances(
        pm, balances, old_positions={}, traded_pair="BTC/USDT", current_price=77000.0, exchange=ex
    )
    p = await pm.get_portfolio()
    assert p["positions"]["ETH/USDT"]["avg_price"] == 2200.0  # real ticker, not 77000
    assert total == pytest.approx(0.5 * 77000.0 + 1.0 * 2200.0)


@pytest.mark.asyncio
async def test_spot_non_traded_pair_writes_zero_when_ticker_fails(pm):
    """Ticker raises -> avg_price=0, NOT traded pair's price (the production bug)."""
    ex = _make_exchange(raises=True)
    balances = {"BTC": 0.5, "OKB": 100.0}
    total, _ = await _sync_spot_from_balances(
        pm, balances, old_positions={}, traded_pair="BTC/USDT", current_price=77000.0, exchange=ex
    )
    p = await pm.get_portfolio()
    assert p["positions"]["OKB/USDT"]["avg_price"] == 0.0  # safe sentinel
    # OKB contributes 0 to total (we don't know its value)
    assert total == pytest.approx(0.5 * 77000.0)


@pytest.mark.asyncio
async def test_spot_non_traded_pair_writes_zero_when_exchange_has_no_fetch_ticker(pm):
    """PaperExchange has no fetch_ticker — must fall through cleanly to 0, not blow up."""

    class NoTicker:
        pass

    balances = {"BTC": 0.5, "OKB": 100.0}
    total, _ = await _sync_spot_from_balances(
        pm,
        balances,
        old_positions={},
        traded_pair="BTC/USDT",
        current_price=77000.0,
        exchange=NoTicker(),
    )
    p = await pm.get_portfolio()
    assert p["positions"]["OKB/USDT"]["avg_price"] == 0.0
    assert total == pytest.approx(0.5 * 77000.0)


# ── derivatives path ──


@pytest.mark.asyncio
async def test_derivs_uses_entry_price_when_present(pm):
    """Persistence still records entryPrice; equity contribution = unrealized_pnl."""
    ex = _make_exchange()
    derivs = {"BTC/USDT:USDT": {"amount": 0.1, "avg_price": 70000.0, "unrealized_pnl": 12.5}}
    total, _ = await _sync_derivatives_from_positions(
        pm, derivs, traded_pair="BTC/USDT:USDT", current_price=77000.0, exchange=ex
    )
    p = await pm.get_portfolio()
    # traded perp pair uses current_price for persistence
    assert p["positions"]["BTC/USDT:USDT"]["avg_price"] == 77000.0
    # equity contribution is unrealized_pnl, NOT notional (notional is not an asset)
    assert total == pytest.approx(12.5)


@pytest.mark.asyncio
async def test_derivs_non_traded_uses_old_entry_price(pm):
    ex = _make_exchange()
    derivs = {
        "BTC/USDT:USDT": {"amount": 0.1, "avg_price": 0.0, "unrealized_pnl": 5.0},  # traded
        "ETH/USDT:USDT": {"amount": 1.0, "avg_price": 2300.0, "unrealized_pnl": -3.0},  # non-traded
    }
    total, _ = await _sync_derivatives_from_positions(
        pm, derivs, traded_pair="BTC/USDT:USDT", current_price=77000.0, exchange=ex
    )
    p = await pm.get_portfolio()
    assert p["positions"]["ETH/USDT:USDT"]["avg_price"] == 2300.0
    assert total == pytest.approx(5.0 - 3.0)


@pytest.mark.asyncio
async def test_derivs_non_traded_fetches_ticker_when_entry_missing(pm):
    ex = _make_exchange({"ETH/USDT:USDT": 2200.0})
    derivs = {
        "BTC/USDT:USDT": {"amount": 0.1, "avg_price": 0.0},  # no unrealized -> 0
        "ETH/USDT:USDT": {"amount": 1.0, "avg_price": 0.0, "unrealized_pnl": 17.0},
    }
    total, _ = await _sync_derivatives_from_positions(
        pm, derivs, traded_pair="BTC/USDT:USDT", current_price=77000.0, exchange=ex
    )
    p = await pm.get_portfolio()
    assert p["positions"]["ETH/USDT:USDT"]["avg_price"] == 2200.0
    assert total == pytest.approx(17.0)  # only ETH contributes; BTC missing field -> 0


@pytest.mark.asyncio
async def test_derivs_non_traded_writes_zero_when_ticker_fails(pm):
    ex = _make_exchange(raises=True)
    derivs = {
        "BTC/USDT:USDT": {"amount": 0.1, "avg_price": 0.0, "unrealized_pnl": 8.0},
        "ETH/USDT:USDT": {"amount": 1.0, "avg_price": 0.0, "unrealized_pnl": 4.0},
    }
    total, _ = await _sync_derivatives_from_positions(
        pm, derivs, traded_pair="BTC/USDT:USDT", current_price=77000.0, exchange=ex
    )
    p = await pm.get_portfolio()
    assert p["positions"]["ETH/USDT:USDT"]["avg_price"] == 0.0
    assert total == pytest.approx(8.0 + 4.0)
