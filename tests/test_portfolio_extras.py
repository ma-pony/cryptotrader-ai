"""Unit tests for portfolio_v2._compute_extras (Deep Review I-T)."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.routes._utils import coerce_timestamp
from api.routes.portfolio_v2 import _compute_extras
from cryptotrader._compat import UTC


class TestCoerceTimestamp:
    def test_datetime_aware_returned_as_is(self) -> None:
        dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        assert coerce_timestamp(dt) == dt

    def test_naive_datetime_promoted_to_utc(self) -> None:
        naive = datetime.fromisoformat("2026-01-01T12:00:00")
        result = coerce_timestamp(naive)
        assert result is not None
        assert result.tzinfo is not None

    def test_iso_string_parsed(self) -> None:
        result = coerce_timestamp("2026-01-01T12:00:00+00:00")
        assert result is not None
        assert result.year == 2026

    def test_z_suffix_parsed(self) -> None:
        result = coerce_timestamp("2026-01-01T12:00:00Z")
        assert result is not None
        assert result.tzinfo is not None

    def test_invalid_string_returns_none(self) -> None:
        assert coerce_timestamp("not a date") is None

    def test_unsupported_type_returns_none(self) -> None:
        assert coerce_timestamp(42) is None
        assert coerce_timestamp(None) is None


class TestComputeExtras:
    @pytest.mark.asyncio
    async def test_fewer_than_30_daily_samples_returns_none_sharpe(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        snaps = [{"timestamp": (base + timedelta(days=i)).isoformat(), "total_value": 10_000.0 + i} for i in range(20)]
        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=snaps)
            js_cls.return_value.log = AsyncMock(return_value=[])
            extras = await _compute_extras(None, current_equity=10000.0)
        assert extras["sharpe_90d"] is None

    @pytest.mark.asyncio
    async def test_flat_equity_stdev_zero_yields_none_sharpe(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        snaps = [{"timestamp": (base + timedelta(days=i)).isoformat(), "total_value": 10_000.0} for i in range(40)]
        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=snaps)
            js_cls.return_value.log = AsyncMock(return_value=[])
            extras = await _compute_extras(None, current_equity=10000.0)
        # Flat equity → all returns are 0 → stdev=0 → Sharpe skipped
        assert extras["sharpe_90d"] is None

    @pytest.mark.asyncio
    async def test_win_rate_from_filled_trades(self) -> None:
        def _make_commit(pnl: float | None, has_order: bool) -> MagicMock:
            c = MagicMock()
            c.pnl = pnl
            c.order = MagicMock() if has_order else None
            c.timestamp = datetime.now(UTC)
            return c

        commits = [
            _make_commit(100.0, True),  # win
            _make_commit(-50.0, True),  # loss
            _make_commit(200.0, True),  # win
            _make_commit(None, True),  # unsettled — excluded from win_rate denominator
            _make_commit(0.0, False),  # no order — excluded entirely
        ]
        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=[])
            js_cls.return_value.log = AsyncMock(return_value=commits)
            extras = await _compute_extras(None, current_equity=10000.0)
        assert extras["total_trades"] == 4  # 4 commits with non-null order
        # win_rate is rounded to 4 decimals in _compute_extras
        assert extras["win_rate"] == pytest.approx(2 / 3, abs=1e-3)

    @pytest.mark.asyncio
    async def test_realized_pnl_30d_window(self) -> None:
        now = datetime.now(UTC)
        commit_in = MagicMock()
        commit_in.pnl = 500.0
        commit_in.order = MagicMock()
        commit_in.timestamp = now - timedelta(days=5)

        commit_out = MagicMock()
        commit_out.pnl = 9999.0  # must be excluded — outside 30d
        commit_out.order = MagicMock()
        commit_out.timestamp = now - timedelta(days=45)

        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=[])
            js_cls.return_value.log = AsyncMock(return_value=[commit_in, commit_out])
            extras = await _compute_extras(None, current_equity=10000.0)
        assert extras["realized_pnl_30d"] == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_empty_journal_yields_zero_stats(self) -> None:
        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=[])
            js_cls.return_value.log = AsyncMock(return_value=[])
            extras = await _compute_extras(None, current_equity=10000.0)
        assert extras["sharpe_90d"] is None
        assert extras["win_rate"] is None
        assert extras["total_trades"] == 0
        assert extras["realized_pnl_30d"] == 0.0
        # No snapshots means no inception baseline → 0.0 (not divide-by-zero)
        assert extras["total_return"] == 0.0
        assert extras["total_return_pct"] == 0.0

    @pytest.mark.asyncio
    async def test_total_return_from_inception_snapshot(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        snaps = [
            {"timestamp": base.isoformat(), "total_value": 100_000.0},
            {"timestamp": (base + timedelta(days=1)).isoformat(), "total_value": 105_000.0},
            {"timestamp": (base + timedelta(days=2)).isoformat(), "total_value": 110_000.0},
        ]
        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=snaps)
            js_cls.return_value.log = AsyncMock(return_value=[])
            extras = await _compute_extras(None, current_equity=110_000.0)
        # 110k - 100k inception = +10k; +10% ROI
        assert extras["total_return"] == pytest.approx(10_000.0)
        assert extras["total_return_pct"] == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_total_return_negative(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        snaps = [{"timestamp": base.isoformat(), "total_value": 100_000.0}]
        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=snaps)
            js_cls.return_value.log = AsyncMock(return_value=[])
            extras = await _compute_extras(None, current_equity=92_500.0)
        assert extras["total_return"] == pytest.approx(-7_500.0)
        assert extras["total_return_pct"] == pytest.approx(-0.075)
