"""Portfolio manager tests."""

import pytest

from cryptotrader.portfolio.manager import PortfolioManager


@pytest.fixture
def pm():
    return PortfolioManager(None)


@pytest.mark.asyncio
async def test_get_portfolio_default(pm):
    p = await pm.get_portfolio()
    assert p["total_value"] == 0
    assert p["positions"] == {}


@pytest.mark.asyncio
async def test_update_and_get_position(pm):
    await pm.update_position("default", "BTC/USDT", 0.5, 50000)
    p = await pm.get_portfolio()
    assert "BTC/USDT" in p["positions"]
    assert p["positions"]["BTC/USDT"]["amount"] == 0.5


@pytest.mark.asyncio
async def test_daily_pnl_no_snapshots(pm):
    """Empty snapshot history -- 'unknown', not synthesized 0.0.

    Returning 0.0 here masked the 2026-04-29 bug where DailyLossLimit divided
    a stale snapshot diff by a tiny equity value and printed -9.42% as a real
    loss. None forces the loss check to skip rather than fabricate a number.
    """
    pnl = await pm.get_daily_pnl()
    assert pnl is None


@pytest.mark.asyncio
async def test_daily_pnl_only_snapshot_today_returns_zero(pm):
    """A single snapshot in today's UTC window is the baseline -- PnL is 0."""
    await pm.snapshot(total_value=10000)
    pnl = await pm.get_daily_pnl()
    assert pnl == 0.0


@pytest.mark.asyncio
async def test_daily_pnl_two_snapshots_today_returns_diff(pm):
    """Two snapshots in today's window -> latest minus earliest_in_window."""
    await pm.snapshot(total_value=10000)
    await pm.snapshot(total_value=9700)
    pnl = await pm.get_daily_pnl()
    assert pnl == -300.0


@pytest.mark.asyncio
async def test_daily_pnl_ignores_stale_yesterday_snapshots(pm):
    """Snapshots before today's UTC midnight must NOT count toward daily PnL.

    Regression: previously snaps[-1]-snaps[-2] would diff arbitrary historical
    snapshots, producing fake 'daily losses' on the first cycle of a new day.
    """
    from datetime import datetime, timedelta

    from cryptotrader._compat import UTC

    yesterday = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=4)
    pm._snapshots.append({"account_id": "default", "total_value": 9000.0, "cash": 0.0, "timestamp": yesterday})
    pnl = await pm.get_daily_pnl()
    assert pnl is None  # zero entries in today's window after yesterday-only history


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
