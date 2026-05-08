"""Tests for data/macro.py — _fetch_fred, _fetch_fear_greed, _fetch_btc_dominance, MacroCollector."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.data.macro import (
    MacroCollector,
    _fetch_btc_dominance,
    _fetch_fear_greed,
    _fetch_fred,
    _is_plausible_fred,
)


def _mock_httpx_client(response=None, error=None):
    """Create a mock httpx.AsyncClient context manager."""
    mock_client = AsyncMock()
    if error:
        mock_client.get.side_effect = error
    else:
        mock_client.get.return_value = response
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_client)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _mock_response(json_data):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


# ── _fetch_fred ──


class TestFetchFred:
    @pytest.mark.asyncio
    async def test_cached_hit_returns_float(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value=2.5):
            assert await _fetch_fred("DFF", "key123") == 2.5

    @pytest.mark.asyncio
    async def test_cached_hit_int(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value=3):
            assert await _fetch_fred("DFF", "key123") == 3.0

    @pytest.mark.asyncio
    async def test_cached_hit_non_numeric(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value="bad"):
            assert await _fetch_fred("DFF", "key123") == 0.0

    @pytest.mark.asyncio
    async def test_backtest_mode_no_cache(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value=None):
            assert await _fetch_fred("DFF", "key123", date="2025-01-01") == 0.0

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value=None):
            assert await _fetch_fred("DFF", "") == 0.0

    @pytest.mark.asyncio
    async def test_api_success(self):
        resp = _mock_response({"observations": [{"value": "4.33", "date": "2025-01-15"}]})
        client = _mock_httpx_client(response=resp)
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("cryptotrader.data.macro.cache_result") as mock_cache,
            patch("httpx.AsyncClient", return_value=client),
        ):
            result = await _fetch_fred("DFF", "key123")
        assert result == 4.33
        mock_cache.assert_called_once_with("fred_DFF", 4.33, date="2025-01-15")

    @pytest.mark.asyncio
    async def test_api_empty_observations(self):
        resp = _mock_response({"observations": []})
        client = _mock_httpx_client(response=resp)
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("httpx.AsyncClient", return_value=client),
        ):
            assert await _fetch_fred("DFF", "key123") == 0.0

    @pytest.mark.asyncio
    async def test_api_exception(self):
        client = _mock_httpx_client(error=Exception("network error"))
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("httpx.AsyncClient", return_value=client),
        ):
            assert await _fetch_fred("DFF", "key123") == 0.0

    # ── sanity-check guard against poisoned values ──

    @pytest.mark.asyncio
    async def test_api_value_below_range_returns_zero_and_skips_cache(self):
        # DTWEXBGS plausible band is 85–145; 50 is below.
        resp = _mock_response({"observations": [{"value": "50.0", "date": "2026-05-07"}]})
        client = _mock_httpx_client(response=resp)
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("cryptotrader.data.macro.cache_result") as mock_cache,
            patch("httpx.AsyncClient", return_value=client),
        ):
            result = await _fetch_fred("DTWEXBGS", "key123")
        assert result == 0.0
        mock_cache.assert_not_called()  # poisoned values must NOT be cached

    @pytest.mark.asyncio
    async def test_api_value_above_range_returns_zero(self):
        resp = _mock_response({"observations": [{"value": "200.0", "date": "2026-05-07"}]})
        client = _mock_httpx_client(response=resp)
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("cryptotrader.data.macro.cache_result") as mock_cache,
            patch("httpx.AsyncClient", return_value=client),
        ):
            assert await _fetch_fred("DTWEXBGS", "key123") == 0.0
        mock_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_api_value_inside_range_caches_normally(self):
        # 118.39 is the actual current DTWEXBGS reading — must pass.
        resp = _mock_response({"observations": [{"value": "118.39", "date": "2026-05-07"}]})
        client = _mock_httpx_client(response=resp)
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("cryptotrader.data.macro.cache_result") as mock_cache,
            patch("httpx.AsyncClient", return_value=client),
        ):
            assert await _fetch_fred("DTWEXBGS", "key123") == 118.39
        mock_cache.assert_called_once_with("fred_DTWEXBGS", 118.39, date="2026-05-07")

    @pytest.mark.asyncio
    async def test_cached_value_outside_range_returns_zero(self):
        # Stale poisoned cache (e.g. someone manually inserted 999) must not leak.
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value=999.0):
            assert await _fetch_fred("DTWEXBGS", "key123") == 0.0

    @pytest.mark.asyncio
    async def test_cached_value_inside_range_passes_through(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value=100.5):
            assert await _fetch_fred("DTWEXBGS", "key123") == 100.5

    @pytest.mark.asyncio
    async def test_unknown_series_passes_through_unchecked(self):
        # Series with no entry in the range table shouldn't be policed.
        resp = _mock_response({"observations": [{"value": "99999.0", "date": "2026-05-07"}]})
        client = _mock_httpx_client(response=resp)
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("cryptotrader.data.macro.cache_result"),
            patch("httpx.AsyncClient", return_value=client),
        ):
            assert await _fetch_fred("MADEUP_SERIES", "key123") == 99999.0


class TestIsPlausibleFred:
    def test_known_series_in_range(self):
        assert _is_plausible_fred("DTWEXBGS", 100.0) is True
        assert _is_plausible_fred("DTWEXBGS", 118.39) is True
        assert _is_plausible_fred("DTWEXBGS", 85.0) is True  # boundary inclusive
        assert _is_plausible_fred("DTWEXBGS", 145.0) is True

    def test_known_series_out_of_range(self):
        assert _is_plausible_fred("DTWEXBGS", 84.99) is False
        assert _is_plausible_fred("DTWEXBGS", 145.01) is False
        assert _is_plausible_fred("DFF", -1.0) is False
        assert _is_plausible_fred("DFF", 16.0) is False
        assert _is_plausible_fred("VIXCLS", 4.0) is False
        assert _is_plausible_fred("T10Y2Y", -3.5) is False

    def test_unknown_series_passes(self):
        assert _is_plausible_fred("FOOBAR", 1e9) is True
        assert _is_plausible_fred("FOOBAR", -1e9) is True


# ── _fetch_fear_greed ──


class TestFetchFearGreed:
    @pytest.mark.asyncio
    async def test_cached_int(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value=75):
            assert await _fetch_fear_greed() == (75, [])

    @pytest.mark.asyncio
    async def test_cached_float(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value=80.0):
            assert await _fetch_fear_greed() == (80, [])

    @pytest.mark.asyncio
    async def test_cached_non_numeric(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value="bad"):
            assert await _fetch_fear_greed() == (50, [])

    @pytest.mark.asyncio
    async def test_backtest_mode(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value=None):
            assert await _fetch_fear_greed(date="2025-01-01") == (50, [])

    @pytest.mark.asyncio
    async def test_api_success(self):
        resp = _mock_response(
            {
                "data": [
                    {"value": "72", "timestamp": "1700000000"},
                    {"value": "65", "timestamp": "1699900000"},
                    {"value": "60", "timestamp": "1699800000"},
                ]
            }
        )
        client = _mock_httpx_client(response=resp)
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("cryptotrader.data.macro.cache_result") as mock_cache,
            patch("httpx.AsyncClient", return_value=client),
        ):
            val, history = await _fetch_fear_greed()
        assert val == 72
        assert history == [72, 65, 60]
        mock_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_empty_data(self):
        resp = _mock_response({"data": []})
        client = _mock_httpx_client(response=resp)
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("httpx.AsyncClient", return_value=client),
        ):
            assert await _fetch_fear_greed() == (50, [])

    @pytest.mark.asyncio
    async def test_api_no_timestamp(self):
        resp = _mock_response({"data": [{"value": "55"}]})
        client = _mock_httpx_client(response=resp)
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("cryptotrader.data.macro.cache_result"),
            patch("httpx.AsyncClient", return_value=client),
        ):
            val, history = await _fetch_fear_greed()
        assert val == 55
        assert history == [55]

    @pytest.mark.asyncio
    async def test_api_exception(self):
        client = _mock_httpx_client(error=Exception("timeout"))
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("httpx.AsyncClient", return_value=client),
        ):
            assert await _fetch_fear_greed() == (50, [])


# ── _fetch_btc_dominance ──


class TestFetchBtcDominance:
    @pytest.mark.asyncio
    async def test_cached_float(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value=55.2):
            assert await _fetch_btc_dominance() == 55.2

    @pytest.mark.asyncio
    async def test_cached_int(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value=58):
            assert await _fetch_btc_dominance() == 58.0

    @pytest.mark.asyncio
    async def test_cached_non_numeric(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value="bad"):
            assert await _fetch_btc_dominance() == 0.0

    @pytest.mark.asyncio
    async def test_backtest_mode(self):
        with patch("cryptotrader.data.macro.get_cached_or_none", return_value=None):
            assert await _fetch_btc_dominance(date="2025-01-01") == 0.0

    @pytest.mark.asyncio
    async def test_api_success(self):
        resp = _mock_response({"data": {"market_cap_percentage": {"btc": 56.78}}})
        client = _mock_httpx_client(response=resp)
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("cryptotrader.data.macro.cache_result") as mock_cache,
            patch("httpx.AsyncClient", return_value=client),
        ):
            assert await _fetch_btc_dominance() == 56.78
        mock_cache.assert_called_once_with("btc_dominance", 56.78)

    @pytest.mark.asyncio
    async def test_api_exception(self):
        client = _mock_httpx_client(error=Exception("network"))
        with (
            patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
            patch("httpx.AsyncClient", return_value=client),
        ):
            assert await _fetch_btc_dominance() == 0.0


# ── MacroCollector ──


class TestMacroCollector:
    @pytest.mark.asyncio
    async def test_collect_all_disabled(self):
        cfg = MagicMock()
        cfg.fred_enabled = False
        cfg.coingecko_enabled = False
        cfg.sosovalue_enabled = False
        cfg.fred_api_key = ""
        cfg.sosovalue_api_key = ""
        collector = MacroCollector(cfg)
        with (
            patch("cryptotrader.data.macro._fetch_fear_greed", return_value=(50, [])),
            patch.object(
                MacroCollector,
                "_load_store_supplements",
                return_value={
                    "vix": 0.0,
                    "sp500": 0.0,
                    "stablecoin": 0.0,
                    "hashrate": 0.0,
                    "yield_curve": 0.0,
                    "m2_supply": 0.0,
                    "cpi": 0.0,
                },
            ),
        ):
            result = await collector.collect()
        assert result.fed_rate == 0.0
        assert result.dxy == 0.0
        assert result.btc_dominance == 0.0
        assert result.fear_greed_index == 50

    @pytest.mark.asyncio
    async def test_collect_with_fred_and_coingecko(self):
        cfg = MagicMock()
        cfg.fred_enabled = True
        cfg.coingecko_enabled = True
        cfg.sosovalue_enabled = False
        cfg.fred_api_key = "test-key"
        cfg.sosovalue_api_key = ""
        collector = MacroCollector(cfg)
        with (
            patch("cryptotrader.data.macro._fetch_fred", return_value=4.5),
            patch("cryptotrader.data.macro._fetch_fear_greed", return_value=(65, [65, 60])),
            patch("cryptotrader.data.macro._fetch_btc_dominance", return_value=55.0),
            patch.object(
                MacroCollector,
                "_load_store_supplements",
                return_value={
                    "vix": 15.0,
                    "sp500": 5000.0,
                    "stablecoin": 0.0,
                    "hashrate": 0.0,
                    "yield_curve": 0.0,
                    "m2_supply": 0.0,
                    "cpi": 0.0,
                },
            ),
        ):
            result = await collector.collect()
        assert result.fed_rate == 4.5
        assert result.dxy == 4.5
        assert result.btc_dominance == 55.0
        assert result.fear_greed_index == 65
        assert result.fear_greed_history == [65, 60]
        assert result.vix == 15.0
        assert result.sp500 == 5000.0

    @pytest.mark.asyncio
    async def test_collect_no_config(self):
        collector = MacroCollector(None)
        with (
            patch("cryptotrader.data.macro._fetch_fear_greed", return_value=(50, [])),
            patch.object(
                MacroCollector,
                "_load_store_supplements",
                return_value={
                    "vix": 0.0,
                    "sp500": 0.0,
                    "stablecoin": 0.0,
                    "hashrate": 0.0,
                    "yield_curve": 0.0,
                    "m2_supply": 0.0,
                    "cpi": 0.0,
                },
            ),
        ):
            result = await collector.collect()
        assert result.fed_rate == 0.0
        assert result.fear_greed_index == 50

    @pytest.mark.asyncio
    async def test_collect_with_sosovalue(self):
        cfg = MagicMock()
        cfg.fred_enabled = False
        cfg.coingecko_enabled = False
        cfg.sosovalue_enabled = True
        cfg.fred_api_key = ""
        cfg.sosovalue_api_key = "soso-key"
        collector = MacroCollector(cfg)

        async def fake_etf(*_a, **_kw):
            return {"dailyNetInflow": 100.0, "totalNetAssets": 5000.0, "cumNetInflow": 2000.0, "topEtfFlows": []}

        with (
            patch("cryptotrader.data.macro._fetch_fear_greed", return_value=(50, [])),
            patch("cryptotrader.data.providers.sosovalue.fetch_etf_metrics", side_effect=fake_etf),
            patch.object(
                MacroCollector,
                "_load_store_supplements",
                return_value={
                    "vix": 0.0,
                    "sp500": 0.0,
                    "stablecoin": 0.0,
                    "hashrate": 0.0,
                    "yield_curve": 0.0,
                    "m2_supply": 0.0,
                    "cpi": 0.0,
                },
            ),
        ):
            result = await collector.collect()
        assert result.etf_daily_net_inflow == 100.0
        assert result.etf_total_net_assets == 5000.0

    def test_load_store_supplements_empty(self):
        with patch("cryptotrader.data.macro.get_latest", return_value=[]):
            result = MacroCollector._load_store_supplements()
        assert result["vix"] == 0.0
        assert result["sp500"] == 0.0

    def test_load_store_supplements_with_data(self):
        def fake_latest(source, limit=1):
            data = {
                "fred_VIXCLS": [(None, 18.5)],
                "fred_SP500": [(None, 5200.0)],
                "stablecoin_total_supply": [(None, {"total_supply": 150e9})],
                "btc_hashrate": [(None, 600.0)],
                "fred_T10Y2Y": [(None, -0.5)],
                "fred_WM2NS": [(None, 21000.0)],
                "fred_CPIAUCSL": [(None, 310.0)],
            }
            return data.get(source, [])

        with patch("cryptotrader.data.macro.get_latest", side_effect=fake_latest):
            result = MacroCollector._load_store_supplements()
        assert result["vix"] == 18.5
        assert result["sp500"] == 5200.0
        assert result["stablecoin"] == 150e9
        assert result["hashrate"] == 600.0
        assert result["yield_curve"] == -0.5

    def test_load_store_supplements_dict_non_supply(self):
        with patch("cryptotrader.data.macro.get_latest", return_value=[(None, {"some": "dict"})]):
            result = MacroCollector._load_store_supplements()
        assert result["vix"] == 0

    def test_load_store_supplements_exception(self):
        with patch("cryptotrader.data.macro.get_latest", side_effect=Exception("db error")):
            result = MacroCollector._load_store_supplements()
        assert result["vix"] == 0.0

    @pytest.mark.asyncio
    async def test_collect_fg_result_non_tuple(self):
        cfg = MagicMock()
        cfg.fred_enabled = False
        cfg.coingecko_enabled = False
        cfg.sosovalue_enabled = False
        cfg.fred_api_key = ""
        cfg.sosovalue_api_key = ""
        collector = MacroCollector(cfg)
        with (
            patch("cryptotrader.data.macro._fetch_fear_greed", return_value=72),
            patch.object(
                MacroCollector,
                "_load_store_supplements",
                return_value={
                    "vix": 0.0,
                    "sp500": 0.0,
                    "stablecoin": 0.0,
                    "hashrate": 0.0,
                    "yield_curve": 0.0,
                    "m2_supply": 0.0,
                    "cpi": 0.0,
                },
            ),
        ):
            result = await collector.collect()
        assert result.fear_greed_index == 72
        assert result.fear_greed_history == []
