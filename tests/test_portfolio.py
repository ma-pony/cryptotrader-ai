"""Portfolio manager tests."""

import pytest
from cryptotrader.portfolio.manager import PortfolioManager


@pytest.fixture
def pm():
    return PortfolioManager(None)


@pytest.mark.asyncio
async def test_get_portfolio_default(pm):
    p = await pm.get_portfolio()
    assert p["total_value"] >= 10000
    assert p["positions"] == {}


@pytest.mark.asyncio
async def test_update_and_get_position(pm):
    await pm.update_position("default", "BTC/USDT", 0.5, 50000)
    p = await pm.get_portfolio()
    assert "BTC/USDT" in p["positions"]
    assert p["positions"]["BTC/USDT"]["amount"] == 0.5


@pytest.mark.asyncio
async def test_daily_pnl_no_snapshots(pm):
    pnl = await pm.get_daily_pnl()
    assert pnl == 0.0


@pytest.mark.asyncio
async def test_drawdown(pm):
    await pm.snapshot(total_value=10000)
    await pm.snapshot(total_value=9500)
    dd = await pm.get_drawdown()
    assert dd < 0


@pytest.mark.asyncio
async def test_returns(pm):
    await pm.snapshot(total_value=10000)
    await pm.snapshot(total_value=10100)
    await pm.snapshot(total_value=10200)
    r = await pm.get_returns(days=60)
    assert len(r) == 2
    assert r[0] > 0
