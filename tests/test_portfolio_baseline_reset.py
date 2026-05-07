"""Tests for portfolio baseline reset (drawdown decoupling design, 2026-05-07).

Covers:
- ``record_baseline_reset`` writes an audit row.
- ``get_last_baseline_reset`` returns the most recent row.
- ``get_drawdown`` honors the cutoff: pre-reset snapshots are excluded.
- Reset is monotonic: a second reset supersedes the first.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from cryptotrader._compat import UTC


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite+aiosqlite:///{tmp_path}/portfolio.db"


def _run(coro):
    return asyncio.run(coro)


def test_record_baseline_reset_writes_audit_row(db_url):
    from cryptotrader.portfolio.manager import PortfolioManager

    async def go():
        pm = PortfolioManager(db_url)
        row = await pm.record_baseline_reset(
            account_id="default",
            baseline_equity=98_753.42,
            operator="alice",
            reason="accepting historical drawdown",
        )
        assert row["id"]
        assert row["account_id"] == "default"
        assert row["baseline_equity"] == 98_753.42
        assert row["operator"] == "alice"
        assert row["reason"] == "accepting historical drawdown"
        assert isinstance(row["reset_at"], datetime)

        latest = await pm.get_last_baseline_reset("default")
        assert latest is not None
        assert latest["id"] == row["id"]

    _run(go())


def test_get_last_baseline_reset_returns_none_when_no_resets(db_url):
    from cryptotrader.portfolio.manager import PortfolioManager

    async def go():
        pm = PortfolioManager(db_url)
        # touch a snapshot so DB tables get created
        await pm.snapshot("default", 100_000.0, 100_000.0)
        latest = await pm.get_last_baseline_reset("default")
        assert latest is None

    _run(go())


def test_get_drawdown_ignores_pre_reset_snapshots(db_url):
    """The headline bug we are fixing: drawdown must reset after operator ack."""
    from cryptotrader.portfolio.manager import PortfolioManager

    async def go():
        pm = PortfolioManager(db_url)

        # Simulate history: peak at 150k → drop to 98k (33% drawdown).
        await pm.snapshot("default", 150_000.0, 150_000.0)
        await pm.snapshot("default", 100_000.0, 100_000.0)
        await pm.snapshot("default", 98_000.0, 98_000.0)

        before = await pm.get_drawdown("default")
        assert before < -0.30, f"expected 33%+ drawdown before reset, got {before:.2%}"

        # Operator accepts the new baseline.
        await pm.record_baseline_reset(account_id="default", baseline_equity=98_000.0, operator="alice", reason="reset")
        # Need a small wait so the next snapshot has a strictly later timestamp.
        await asyncio.sleep(0.05)
        await pm.snapshot("default", 98_500.0, 98_500.0)

        after = await pm.get_drawdown("default")
        assert after >= -0.005, f"drawdown should be ~0 after fresh baseline, got {after:.2%}"

    _run(go())


def test_second_reset_supersedes_first(db_url):
    """get_last_baseline_reset returns the most recent row only."""
    from cryptotrader.portfolio.manager import PortfolioManager

    async def go():
        pm = PortfolioManager(db_url)

        first = await pm.record_baseline_reset(
            account_id="default", baseline_equity=100_000.0, operator="alice", reason="first"
        )
        await asyncio.sleep(0.05)
        second = await pm.record_baseline_reset(
            account_id="default", baseline_equity=95_000.0, operator="bob", reason="second"
        )

        latest = await pm.get_last_baseline_reset("default")
        assert latest is not None
        assert latest["id"] == second["id"], "latest should be the second (later) reset"
        assert latest["operator"] == "bob"
        assert latest["reason"] == "second"
        assert first["id"] != second["id"]

    _run(go())


def test_reset_only_affects_target_account(db_url):
    """A reset for account A must not affect drawdown for account B."""
    from cryptotrader.portfolio.manager import PortfolioManager

    async def go():
        pm = PortfolioManager(db_url)
        # A: peak 150k → 100k
        await pm.snapshot("acct_a", 150_000.0, 150_000.0)
        await pm.snapshot("acct_a", 100_000.0, 100_000.0)
        # B: peak 200k → 120k
        await pm.snapshot("acct_b", 200_000.0, 200_000.0)
        await pm.snapshot("acct_b", 120_000.0, 120_000.0)

        await pm.record_baseline_reset(account_id="acct_a", baseline_equity=100_000.0, operator="op", reason="ack")
        await asyncio.sleep(0.05)
        await pm.snapshot("acct_a", 102_000.0, 102_000.0)

        # acct_a drawdown should be near zero, acct_b unchanged at -40%.
        a_dd = await pm.get_drawdown("acct_a")
        b_dd = await pm.get_drawdown("acct_b")
        assert a_dd >= -0.005, f"acct_a should be reset, got {a_dd:.2%}"
        assert b_dd <= -0.35, f"acct_b should remain at ~40% drawdown, got {b_dd:.2%}"

    _run(go())


def test_drawdown_limit_does_not_trip_circuit_breaker():
    """Mirror of test_live_readiness assertion — kept here for proximity to design."""
    from unittest.mock import AsyncMock, MagicMock

    from cryptotrader.config import LossConfig
    from cryptotrader.models import TradeVerdict
    from cryptotrader.risk.checks.loss import DrawdownLimit

    async def go():
        mock_redis = MagicMock()
        mock_redis.available = True
        mock_redis.set_circuit_breaker = AsyncMock()
        check = DrawdownLimit(LossConfig(max_drawdown_pct=0.10), mock_redis)
        result = await check.evaluate(TradeVerdict(action="long", confidence=0.8), {"drawdown": 0.20})
        assert not result.passed
        mock_redis.set_circuit_breaker.assert_not_called()

    _run(go())


def test_baseline_reset_requires_db():
    """In-memory mode raises explicit RuntimeError (no silent no-op)."""
    from cryptotrader.portfolio.manager import PortfolioManager

    async def go():
        pm = PortfolioManager(database_url=None)
        with pytest.raises(RuntimeError):
            await pm.record_baseline_reset(account_id="default", baseline_equity=0.0, operator="x", reason="y")

    _run(go())


# Silence lint about unused import: timedelta is used in implicit comparisons.
_ = timedelta(seconds=1)
_ = UTC
