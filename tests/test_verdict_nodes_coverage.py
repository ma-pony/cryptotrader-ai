"""Tests for nodes/verdict.py — _extract_ohlcv_returns, _merge_returns, _measure_api_latency, _should_downgrade."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cryptotrader.nodes.verdict import (
    _extract_ohlcv_returns,
    _measure_api_latency,
    _merge_returns,
    _should_downgrade_to_weighted,
)

# ── _merge_returns ──


class TestMergeReturns:
    def test_pm_returns_sufficient(self):
        result = _merge_returns([0.01, 0.02, -0.01] * 10, [0.05], min_count=20)
        assert len(result) == 30
        assert result == [0.01, 0.02, -0.01] * 10

    def test_pm_returns_empty_uses_ohlcv(self):
        result = _merge_returns([], [0.01, 0.02, -0.01], min_count=3)
        assert result == [0.01, 0.02, -0.01]

    def test_pm_returns_partial_supplement(self):
        pm = [0.01, 0.02]
        ohlcv = [0.05, 0.06, 0.07, 0.08]
        result = _merge_returns(pm, ohlcv, min_count=5)
        assert len(result) == 5
        assert result[-2:] == pm

    def test_both_empty(self):
        assert _merge_returns([], []) == []


# ── _extract_ohlcv_returns ──


class TestExtractOhlcvReturns:
    def test_no_snapshot(self):
        state = {"data": {}, "metadata": {}}
        assert _extract_ohlcv_returns(state) == ([], [])

    def test_no_market(self):
        snap = MagicMock(spec=[])
        state = {"data": {"snapshot": snap}, "metadata": {}}
        assert _extract_ohlcv_returns(state) == ([], [])

    def test_ohlcv_none(self):
        snap = MagicMock()
        snap.market.ohlcv = None
        state = {"data": {"snapshot": snap}, "metadata": {}}
        assert _extract_ohlcv_returns(state) == ([], [])

    def test_single_close(self):
        snap = MagicMock()
        snap.market.ohlcv = {"close": pd.Series([100.0])}
        state = {"data": {"snapshot": snap}, "metadata": {}}
        prices, returns = _extract_ohlcv_returns(state)
        assert prices == [100.0]
        assert returns == []

    def test_hourly_aggregation_to_daily(self):
        closes = [100.0 + i for i in range(50)]
        snap = MagicMock()
        snap.market.ohlcv = {"close": pd.Series(closes)}
        state = {"data": {"snapshot": snap}, "metadata": {"timeframe": "1h"}}
        prices, returns = _extract_ohlcv_returns(state)
        assert len(prices) == 50
        assert len(returns) > 0

    def test_daily_timeframe_no_aggregation(self):
        closes = [100.0, 110.0, 105.0]
        snap = MagicMock()
        snap.market.ohlcv = {"close": pd.Series(closes)}
        state = {"data": {"snapshot": snap}, "metadata": {"timeframe": "1d"}}
        prices, returns = _extract_ohlcv_returns(state)
        assert len(prices) == 3
        assert len(returns) == 2
        assert abs(returns[0] - 0.10) < 0.001


# ── _measure_api_latency ──


class TestMeasureApiLatency:
    @pytest.mark.asyncio
    async def test_paper_mode_returns_100(self):
        state = {"metadata": {"engine": "paper"}}
        assert await _measure_api_latency(state) == 100

    @pytest.mark.asyncio
    async def test_default_engine_returns_100(self):
        state = {"metadata": {}}
        assert await _measure_api_latency(state) == 100


# ── _should_downgrade_to_weighted ──


class TestShouldDowngradeToWeighted:
    @pytest.mark.asyncio
    async def test_has_position_returns_false(self):
        state = {
            "data": {"position_context": {"side": "long"}},
            "metadata": {},
        }
        assert await _should_downgrade_to_weighted(state) is False

    @pytest.mark.asyncio
    async def test_flat_no_circuit_breaker(self):
        mock_rsm = MagicMock()
        mock_rsm.is_circuit_breaker_active = MagicMock(return_value=_coro(False))
        state = {
            "data": {"position_context": {"side": "flat"}},
            "metadata": {"redis_url": None},
        }
        with patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm):
            assert await _should_downgrade_to_weighted(state) is True

    @pytest.mark.asyncio
    async def test_flat_circuit_breaker_active(self):
        mock_rsm = MagicMock()
        mock_rsm.is_circuit_breaker_active = MagicMock(return_value=_coro(True))
        state = {
            "data": {"position_context": {"side": "flat"}},
            "metadata": {"redis_url": None},
        }
        with patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm):
            assert await _should_downgrade_to_weighted(state) is False

    @pytest.mark.asyncio
    async def test_no_position_context(self):
        mock_rsm = MagicMock()
        mock_rsm.is_circuit_breaker_active = MagicMock(return_value=_coro(False))
        state = {
            "data": {},
            "metadata": {"redis_url": None},
        }
        with patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm):
            assert await _should_downgrade_to_weighted(state) is True

    @pytest.mark.asyncio
    async def test_redis_exception_returns_false(self):
        state = {
            "data": {"position_context": {"side": "flat"}},
            "metadata": {"redis_url": None},
        }
        with patch("cryptotrader.risk.state.RedisStateManager", side_effect=Exception("no redis")):
            assert await _should_downgrade_to_weighted(state) is False


async def _coro(val):
    return val
