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
        # 2026-05-07 contract change: stats only count close-action commits.
        # Open-action commits' .pnl is calibration / unrealized, not realized PnL.
        def _make_commit(pnl: float | None, has_order: bool, action: str = "close") -> MagicMock:
            c = MagicMock()
            c.pnl = pnl
            c.order = MagicMock() if has_order else None
            c.timestamp = datetime.now(UTC)
            c.verdict = MagicMock()
            c.verdict.action = action
            return c

        commits = [
            _make_commit(100.0, True),  # close, win
            _make_commit(-50.0, True),  # close, loss
            _make_commit(200.0, True),  # close, win
            _make_commit(None, True),  # close, unsettled — counted in total but not in win/avg
            _make_commit(0.0, False),  # no order — excluded
            _make_commit(999.0, True, action="long"),  # open — excluded (calibration value)
            _make_commit(-888.0, True, action="short"),  # open — excluded
        ]
        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=[])
            js_cls.return_value.log = AsyncMock(return_value=commits)
            extras = await _compute_extras(None, current_equity=10000.0)
        assert extras["total_trades"] == 4  # 4 close-action commits with order
        assert extras["win_rate"] == pytest.approx(2 / 3, abs=1e-3)
        # avg_trade_pnl = mean over settled close commits: (100 - 50 + 200) / 3 ≈ 83.33
        assert extras["avg_trade_pnl"] == pytest.approx(83.33, abs=1e-2)

    @pytest.mark.asyncio
    async def test_realized_pnl_30d_window(self) -> None:
        now = datetime.now(UTC)
        commit_in = MagicMock()
        commit_in.pnl = 500.0
        commit_in.order = MagicMock()
        commit_in.timestamp = now - timedelta(days=5)
        commit_in.verdict = MagicMock()
        commit_in.verdict.action = "close"

        commit_out = MagicMock()
        commit_out.pnl = 9999.0  # must be excluded — outside 30d
        commit_out.order = MagicMock()
        commit_out.timestamp = now - timedelta(days=45)
        commit_out.verdict = MagicMock()
        commit_out.verdict.action = "close"

        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=[])
            js_cls.return_value.log = AsyncMock(return_value=[commit_in, commit_out])
            extras = await _compute_extras(None, current_equity=10000.0)
        assert extras["realized_pnl_30d"] == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_pre_inception_trades_excluded(self) -> None:
        """spec(B) 2026-05-07: trades older than first snapshot are excluded so
        commit-level stats align with snapshot-based ``total_return``.
        """
        now = datetime.now(UTC)
        first_snap_ts = now - timedelta(days=10)

        def _close(pnl: float, days_ago: float) -> MagicMock:
            c = MagicMock()
            c.pnl = pnl
            c.order = MagicMock()
            c.timestamp = now - timedelta(days=days_ago)
            c.verdict = MagicMock()
            c.verdict.action = "close"
            return c

        snaps = [{"timestamp": first_snap_ts, "total_value": 10_000.0}]
        commits = [
            _close(-5_000.0, days_ago=30),  # PRE-INCEPTION, excluded
            _close(100.0, days_ago=5),  # post-inception, win
            _close(-200.0, days_ago=2),  # post-inception, loss
        ]
        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=snaps)
            js_cls.return_value.log = AsyncMock(return_value=commits)
            extras = await _compute_extras(None, current_equity=9_900.0)
        assert extras["total_trades"] == 2  # the -5_000 pre-inception is excluded
        assert extras["realized_pnl_30d"] == pytest.approx(-100.0)  # 100 + (-200)
        assert extras["avg_trade_pnl"] == pytest.approx(-50.0)
        assert extras["win_rate"] == pytest.approx(0.5)

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
        assert extras["avg_trade_pnl"] is None

    @pytest.mark.asyncio
    async def test_total_return_realized_plus_unrealized(self) -> None:
        """total_return = sum(realized close-action pnl since inception) + sum(unrealized).

        2026-05-11 redesign: ``total_return`` is *trading* P&L, not ``equity -
        baseline``. The old formula counted USDT deposits as profit; the new
        formula counts only closed-trade P&L plus current mark-to-market.
        """
        base = datetime(2026, 1, 1, tzinfo=UTC)
        snaps = [{"timestamp": base.isoformat(), "total_value": 100_000.0}]
        # 3 closed trades since inception: +500, -200, +300 → realized 600
        commits = []
        for pnl in (500.0, -200.0, 300.0):
            c = MagicMock()
            c.pnl = pnl
            c.order = MagicMock()
            c.timestamp = base + timedelta(days=1)
            c.verdict = MagicMock()
            c.verdict.action = "close"
            commits.append(c)
        # Current open position with $400 unrealized profit
        raw_positions = {"BTC/USDT:USDT": {"unrealized_pnl": 400.0}}
        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=snaps)
            js_cls.return_value.log = AsyncMock(return_value=commits)
            extras = await _compute_extras(None, current_equity=110_000.0, raw_positions=raw_positions)
        # realized 600 + unrealized 400 = 1000; pct = 1000 / baseline 100_000 = 0.01
        assert extras["total_return"] == pytest.approx(1000.0)
        assert extras["total_return_pct"] == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_total_return_pct_uses_configured_initial_capital_baseline(self) -> None:
        """When config.portfolio.initial_capital > 0, it overrides the first-snapshot
        baseline as the denominator for ``total_return_pct`` (numerator stays
        realized + unrealized — the new trading-P&L definition)."""
        from unittest.mock import MagicMock as _MagicMock

        base = datetime(2026, 1, 1, tzinfo=UTC)
        snaps = [{"timestamp": base.isoformat(), "total_value": 50_000.0}]
        cfg = _MagicMock()
        cfg.portfolio.initial_capital = 100_000.0
        # 1 closed trade since inception: +1000 realized
        c = MagicMock()
        c.pnl = 1000.0
        c.order = MagicMock()
        c.timestamp = base + timedelta(days=1)
        c.verdict = MagicMock()
        c.verdict.action = "close"
        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
            patch("cryptotrader.config.load_config", return_value=cfg),
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=snaps)
            js_cls.return_value.log = AsyncMock(return_value=[c])
            extras = await _compute_extras(None, current_equity=110_000.0, raw_positions=None)
        # realized 1000 + unrealized 0 = 1000; pct uses configured baseline 100K → 0.01
        assert extras["total_return"] == pytest.approx(1000.0)
        assert extras["total_return_pct"] == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_total_return_negative_when_realized_plus_unrealized_negative(self) -> None:
        """Net loss case: realized losses + unrealized losses sum to negative
        total_return. Matches '总收益' definition; previously this would have
        been wrong if the user funded $0 → +$3500 deposit → showed +$3500 gain
        even with negative trading P&L."""
        base = datetime(2026, 1, 1, tzinfo=UTC)
        snaps = [{"timestamp": base.isoformat(), "total_value": 100_000.0}]
        # 2 close trades, both losses: -500, -200 → realized -700
        commits = []
        for pnl in (-500.0, -200.0):
            c = MagicMock()
            c.pnl = pnl
            c.order = MagicMock()
            c.timestamp = base + timedelta(days=1)
            c.verdict = MagicMock()
            c.verdict.action = "close"
            commits.append(c)
        # Current position underwater by -50
        raw_positions = {"ETH/USDT:USDT": {"unrealized_pnl": -50.0}}
        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm_cls.return_value.load_snapshots = AsyncMock(return_value=snaps)
            js_cls.return_value.log = AsyncMock(return_value=commits)
            extras = await _compute_extras(None, current_equity=92_500.0, raw_positions=raw_positions)
        # realized -700 + unrealized -50 = -750
        assert extras["total_return"] == pytest.approx(-750.0)
        assert extras["total_return_pct"] == pytest.approx(-0.0075)
