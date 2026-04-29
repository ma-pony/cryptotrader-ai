"""Unit tests for risk-route compute + build helpers (Deep Review C8)."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.routes.risk import (
    _build_correlation_groups,
    _build_recent_blocks,
    _compute_cvar_95,
    _compute_daily_loss_pct,
    _compute_drawdown_pct,
    _compute_total_exposure_pct,
    _known_pairs,
)
from cryptotrader._compat import UTC


class TestComputeDailyLossPct:
    @pytest.mark.asyncio
    async def test_loss_yields_positive_pct(self) -> None:
        portfolio = {"total_value": 10_000.0}
        result = await _compute_daily_loss_pct(None, portfolio=portfolio, pnl_24h=-300.0)
        assert result == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_profit_yields_negative_pct(self) -> None:
        """Design contract: negative return when profitable (frontend shows as 'profit day')."""
        portfolio = {"total_value": 10_000.0}
        result = await _compute_daily_loss_pct(None, portfolio=portfolio, pnl_24h=200.0)
        assert result == pytest.approx(-2.0)

    @pytest.mark.asyncio
    async def test_zero_equity_returns_none(self) -> None:
        portfolio = {"total_value": 0.0}
        result = await _compute_daily_loss_pct(None, portfolio=portfolio, pnl_24h=-100.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_negative_equity_returns_none(self) -> None:
        portfolio = {"total_value": -50.0}
        result = await _compute_daily_loss_pct(None, portfolio=portfolio, pnl_24h=-100.0)
        assert result is None


class TestComputeDrawdownPct:
    @pytest.mark.asyncio
    async def test_from_snaps_computes_percent(self) -> None:
        snaps = [
            {"total_value": 10_000.0},
            {"total_value": 12_000.0},  # peak
            {"total_value": 11_000.0},  # 8.33% drawdown from peak
        ]
        result = await _compute_drawdown_pct(None, snaps=snaps)
        assert result == pytest.approx(8.33, abs=0.01)

    @pytest.mark.asyncio
    async def test_empty_snaps_returns_zero(self) -> None:
        result = await _compute_drawdown_pct(None, snaps=[])
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_all_new_highs_returns_zero(self) -> None:
        snaps = [{"total_value": v} for v in [1000, 1100, 1200, 1300]]
        result = await _compute_drawdown_pct(None, snaps=snaps)
        assert result == pytest.approx(0.0)


class TestComputeTotalExposurePct:
    @pytest.mark.asyncio
    async def test_long_and_short_both_count_absolute(self) -> None:
        """Short positions contribute via abs(amount) * avg_price."""
        portfolio = {
            "total_value": 10_000.0,
            "positions": {
                "BTC/USDT": {"amount": 0.1, "avg_price": 90_000.0},  # 9000 notional
                "ETH/USDT": {"amount": -2.0, "avg_price": 3000.0},  # 6000 notional (short)
            },
        }
        result = await _compute_total_exposure_pct(None, portfolio=portfolio)
        assert result == pytest.approx(150.0)  # 15k/10k

    @pytest.mark.asyncio
    async def test_zero_equity_returns_none(self) -> None:
        portfolio = {"total_value": 0.0, "positions": {}}
        assert await _compute_total_exposure_pct(None, portfolio=portfolio) is None

    @pytest.mark.asyncio
    async def test_empty_positions_yields_zero(self) -> None:
        portfolio = {"total_value": 10_000.0, "positions": {}}
        assert await _compute_total_exposure_pct(None, portfolio=portfolio) == 0.0


class TestComputeCVaR95:
    @pytest.mark.asyncio
    async def test_fewer_than_30_samples_returns_none(self) -> None:
        snaps = [
            {"timestamp": f"2026-01-{(i % 28) + 1:02d}T00:00:00+00:00", "total_value": 1000 + i} for i in range(25)
        ]
        assert await _compute_cvar_95(None, snaps=snaps) is None

    @pytest.mark.asyncio
    async def test_30_samples_produces_positive_cvar_for_losses(self) -> None:
        """Declining series — CVaR must be positive (loss magnitude)."""
        base = datetime(2026, 1, 1, tzinfo=UTC)
        snaps = []
        val = 10_000.0
        for i in range(40):
            val *= 0.98  # -2% per day, synthetic drawdown
            snaps.append(
                {
                    "timestamp": (base + timedelta(days=i)).isoformat(),
                    "total_value": val,
                }
            )
        result = await _compute_cvar_95(None, snaps=snaps)
        assert result is not None
        assert result > 0  # always-down series → positive CVaR (loss magnitude)

    @pytest.mark.asyncio
    async def test_flat_equity_stdev_zero_path(self) -> None:
        """Flat equity (all returns 0) → CVaR of 0% (tail is all zeros)."""
        base = datetime(2026, 1, 1, tzinfo=UTC)
        snaps = [{"timestamp": (base + timedelta(days=i)).isoformat(), "total_value": 10_000.0} for i in range(40)]
        result = await _compute_cvar_95(None, snaps=snaps)
        assert result == pytest.approx(0.0)


class TestBuildCorrelationGroups:
    @pytest.mark.asyncio
    async def test_open_position_counted(self) -> None:
        portfolio = {"positions": {"BTC/USDT": {"amount": 0.1, "avg_price": 90_000.0}}}
        groups = await _build_correlation_groups(None, portfolio=portfolio)
        btc_group = next(g for g in groups if g.name == "BTC-correlated")
        assert btc_group.open == 1

    @pytest.mark.asyncio
    async def test_zero_amount_not_counted(self) -> None:
        portfolio = {"positions": {"BTC/USDT": {"amount": 0.0, "avg_price": 90_000.0}}}
        groups = await _build_correlation_groups(None, portfolio=portfolio)
        btc_group = next(g for g in groups if g.name == "BTC-correlated")
        assert btc_group.open == 0

    @pytest.mark.asyncio
    async def test_all_5_groups_emitted(self) -> None:
        groups = await _build_correlation_groups(None, portfolio={"positions": {}})
        names = {g.name for g in groups}
        assert {"BTC-correlated", "ETH-correlated", "L1-alt", "DeFi", "meme"}.issubset(names)


class TestKnownPairs:
    @pytest.mark.asyncio
    async def test_union_of_positions_and_recent_commits(self) -> None:
        portfolio = {"positions": {"BTC/USDT": {"amount": 0.1}}}
        recent = [MagicMock(pair="ETH/USDT"), MagicMock(pair="BTC/USDT")]
        with patch("cryptotrader.journal.store.JournalStore") as js_cls:
            js_cls.return_value.log = AsyncMock(return_value=recent)
            pairs = await _known_pairs(None, portfolio=portfolio)
        assert set(pairs) == {"BTC/USDT", "ETH/USDT"}

    @pytest.mark.asyncio
    async def test_zero_amount_position_excluded(self) -> None:
        portfolio = {"positions": {"BTC/USDT": {"amount": 0.0}}}
        with patch("cryptotrader.journal.store.JournalStore") as js_cls:
            js_cls.return_value.log = AsyncMock(return_value=[])
            pairs = await _known_pairs(None, portfolio=portfolio)
        assert pairs == []


class TestBuildRecentBlocks:
    @pytest.mark.asyncio
    async def test_only_rejected_commits_included(self) -> None:
        passed = MagicMock()
        passed.risk_gate = MagicMock(passed=True, rejected_by="", reason="")
        rejected = MagicMock()
        rejected.risk_gate = MagicMock(passed=False, rejected_by="CooldownCheck", reason="same-pair")
        rejected.hash = "abc123"
        rejected.timestamp = datetime(2026, 1, 1, tzinfo=UTC)

        with patch("cryptotrader.journal.store.JournalStore") as js_cls:
            js_cls.return_value.log = AsyncMock(return_value=[passed, rejected])
            blocks = await _build_recent_blocks(None)
        assert len(blocks) == 1
        assert blocks[0].rule == "CooldownCheck"

    @pytest.mark.asyncio
    async def test_capped_at_10(self) -> None:
        many = []
        for i in range(20):
            c = MagicMock()
            c.risk_gate = MagicMock(passed=False, rejected_by="X", reason="y")
            c.hash = f"h{i}"
            c.timestamp = datetime(2026, 1, 1, tzinfo=UTC)
            many.append(c)
        with patch("cryptotrader.journal.store.JournalStore") as js_cls:
            js_cls.return_value.log = AsyncMock(return_value=many)
            blocks = await _build_recent_blocks(None)
        assert len(blocks) == 10

    @pytest.mark.asyncio
    async def test_journal_error_returns_empty(self) -> None:
        with patch("cryptotrader.journal.store.JournalStore") as js_cls:
            js_cls.return_value.log = AsyncMock(side_effect=RuntimeError("db down"))
            blocks = await _build_recent_blocks(None)
        assert blocks == []
