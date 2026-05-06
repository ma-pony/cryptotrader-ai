"""Dust-threshold tests for portfolio sync.

A close fill that doesn't perfectly match cost-basis often leaves a
sub-microscopic residue on the exchange — e.g. ``ETH = 2.91e-07``,
``BTC = 3.19e-09`` — observed in production after a successful close.
The sync helpers used to write those amounts straight into ``portfolios``
because ``amount != 0``, leaving stale dust rows that the orphan sweep
also wouldn't clean (the pair is still ``seen``). The AI then treated the
microscopic position as a real long.

These tests pin down the contract: amounts below ``_DUST_THRESHOLD`` are
treated as zero — the row does NOT get persisted as an active position,
and is allowed to fall through to the orphan sweep so any prior DB row
gets zeroed out.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from cryptotrader.nodes.execution import (
    _sweep_orphaned_positions,
    _sync_derivatives_from_positions,
    _sync_spot_from_balances,
)
from cryptotrader.portfolio.manager import PortfolioManager


@pytest.fixture
def pm():
    return PortfolioManager(None)


def _exchange_with_ticker(prices: dict[str, float] | None = None):
    ex = AsyncMock()

    async def _ticker(symbol):
        if prices is None or symbol not in prices:
            raise ValueError(f"unknown {symbol}")
        return {"last": prices[symbol]}

    ex.fetch_ticker = _ticker
    return ex


# ── spot dust ──


@pytest.mark.asyncio
async def test_spot_dust_amount_is_treated_as_zero(pm):
    """ETH=2.91e-07 (production value) must NOT be persisted as a position."""
    ex = _exchange_with_ticker({"ETH/USDT": 2300.0})
    balances = {"BTC": 0.5, "ETH": 2.91e-07}  # ETH dust
    total, seen = await _sync_spot_from_balances(
        pm, balances, old_positions={}, traded_pair="BTC/USDT", current_price=77000.0, exchange=ex
    )
    p = await pm.get_portfolio()
    # BTC stays, ETH dust dropped
    assert "BTC/USDT" in p["positions"]
    assert "ETH/USDT" not in p["positions"]
    # Dust pair NOT in seen — sweep can clean any leftover DB row
    assert "ETH/USDT" not in seen
    # Total only counts the real position
    assert total == pytest.approx(0.5 * 77000.0)


@pytest.mark.asyncio
async def test_spot_dust_lets_sweep_clear_old_db_row(pm):
    """If portfolios DB still has an old ETH row, dust amount → orphan → swept to 0."""
    # Seed a pre-existing ETH row in the DB-equivalent memory store.
    await pm.update_position("default", "ETH/USDT", 2.68, 2284.95)
    assert (await pm.get_portfolio())["positions"]["ETH/USDT"]["amount"] == 2.68

    ex = _exchange_with_ticker({"ETH/USDT": 2300.0})
    balances = {"BTC": 0.5, "ETH": 2.91e-07}
    old = (await pm.get_portfolio())["positions"]
    _, seen = await _sync_spot_from_balances(
        pm, balances, old_positions=old, traded_pair="BTC/USDT", current_price=77000.0, exchange=ex
    )
    # ETH not in seen — sweep treats it as orphaned
    await _sweep_orphaned_positions(pm, old, seen, derivatives_observed=True)

    p = await pm.get_portfolio()
    # ETH amount zeroed by sweep
    assert p["positions"]["ETH/USDT"]["amount"] == 0


# ── derivatives dust ──


@pytest.mark.asyncio
async def test_perp_dust_amount_is_treated_as_zero(pm):
    ex = _exchange_with_ticker()
    derivs = {
        "BTC/USDT:USDT": {"amount": 0.5, "avg_price": 70000.0, "unrealized_pnl": 100.0},
        "ETH/USDT:USDT": {"amount": 1e-08, "avg_price": 2300.0, "unrealized_pnl": 999.0},  # dust skipped
    }
    total, seen = await _sync_derivatives_from_positions(
        pm, derivs, traded_pair="BTC/USDT:USDT", current_price=77000.0, exchange=ex
    )
    p = await pm.get_portfolio()
    assert "BTC/USDT:USDT" in p["positions"]
    assert "ETH/USDT:USDT" not in p["positions"]
    assert "ETH/USDT:USDT" not in seen
    # Equity contribution = unrealized_pnl of NON-dust positions
    assert total == pytest.approx(100.0)


# ── sentinel: real positions just above threshold still survive ──


@pytest.mark.asyncio
async def test_amount_just_above_dust_threshold_is_kept(pm):
    """Threshold is 1e-6; an amount of 1e-5 is 10x larger and must be kept."""
    ex = _exchange_with_ticker({"ETH/USDT": 2300.0})
    balances = {"ETH": 1e-5}
    _, seen = await _sync_spot_from_balances(
        pm, balances, old_positions={}, traded_pair="BTC/USDT", current_price=77000.0, exchange=ex
    )
    p = await pm.get_portfolio()
    assert "ETH/USDT" in p["positions"]
    assert p["positions"]["ETH/USDT"]["amount"] == pytest.approx(1e-5)
    assert "ETH/USDT" in seen


@pytest.mark.asyncio
async def test_negative_dust_is_also_cleaned(pm):
    """Short perp leftover with amount=-1e-09 should also fall through."""
    ex = _exchange_with_ticker()
    derivs = {"ETH/USDT:USDT": {"amount": -1e-09, "avg_price": 2300.0}}
    total, seen = await _sync_derivatives_from_positions(
        pm, derivs, traded_pair="BTC/USDT:USDT", current_price=77000.0, exchange=ex
    )
    assert seen == set()
    assert total == 0.0
