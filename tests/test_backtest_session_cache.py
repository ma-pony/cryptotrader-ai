"""Legacy OHLCV cache checks; file-session migration is tested in test_backtest_import.py."""

import pytest


@pytest.mark.usefixtures("_patch_cache_db")
class TestOHLCVCache:
    """OHLCV cache get/store operations with isolated SQLite databases."""

    @pytest.fixture
    def _patch_cache_db(self, tmp_path, monkeypatch):
        """Redirect the OHLCV cache database to a temp directory."""
        from cryptotrader.backtest import cache as cache_module

        db_path = tmp_path / "ohlcv_cache.db"
        monkeypatch.setattr(cache_module, "CACHE_DB", db_path)

    def test_get_cached_returns_empty_for_empty_db(self) -> None:
        from cryptotrader.backtest.cache import get_cached

        result = get_cached("BTC/USDT", "1h", 0, 1_000_000_000)
        assert result == []

    def test_store_and_retrieve_ohlcv(self) -> None:
        from cryptotrader.backtest.cache import get_cached, store_ohlcv

        candles = [
            [1_000_000, 50000.0, 51000.0, 49000.0, 50500.0, 100.0],
            [1_001_000, 50500.0, 52000.0, 50000.0, 51000.0, 120.0],
        ]
        store_ohlcv("BTC/USDT", "1h", candles)

        result = get_cached("BTC/USDT", "1h", 0, 2_000_000)
        assert len(result) == 2
        assert result[0][0] == 1_000_000
        assert result[1][0] == 1_001_000

    def test_get_cached_filters_by_time_range(self) -> None:
        from cryptotrader.backtest.cache import get_cached, store_ohlcv

        candles = [
            [100, 100.0, 110.0, 90.0, 105.0, 1.0],
            [200, 105.0, 115.0, 95.0, 110.0, 1.0],
            [300, 110.0, 120.0, 100.0, 115.0, 1.0],
        ]
        store_ohlcv("ETH/USDT", "1h", candles)

        # Only fetch ts in [150, 250]
        result = get_cached("ETH/USDT", "1h", 150, 250)
        assert len(result) == 1
        assert result[0][0] == 200

    def test_get_cached_filters_by_pair(self) -> None:
        from cryptotrader.backtest.cache import get_cached, store_ohlcv

        candles = [[100, 1.0, 1.1, 0.9, 1.05, 10.0]]
        store_ohlcv("SOL/USDT", "1h", candles)

        # Query for different pair — should return empty
        result = get_cached("BTC/USDT", "1h", 0, 1000)
        assert result == []

    def test_get_cached_filters_by_timeframe(self) -> None:
        from cryptotrader.backtest.cache import get_cached, store_ohlcv

        candles = [[100, 1.0, 1.1, 0.9, 1.05, 10.0]]
        store_ohlcv("BTC/USDT", "4h", candles)

        # Query 1h timeframe — should return empty
        result = get_cached("BTC/USDT", "1h", 0, 1000)
        assert result == []

    def test_store_ohlcv_upserts_on_duplicate(self) -> None:
        """Duplicate (pair, timeframe, ts) entries should be replaced, not duplicated."""
        from cryptotrader.backtest.cache import get_cached, store_ohlcv

        candles = [[100, 50000.0, 51000.0, 49000.0, 50500.0, 100.0]]
        store_ohlcv("BTC/USDT", "1h", candles)

        # Store same ts with updated values
        updated = [[100, 51000.0, 52000.0, 50000.0, 51500.0, 200.0]]
        store_ohlcv("BTC/USDT", "1h", updated)

        result = get_cached("BTC/USDT", "1h", 0, 200)
        # Should still be only 1 entry
        assert len(result) == 1
        # Value should be the updated one
        assert result[0][1] == pytest.approx(51000.0)

    def test_store_empty_candles_does_nothing(self) -> None:
        from cryptotrader.backtest.cache import get_cached, store_ohlcv

        store_ohlcv("BTC/USDT", "1h", [])
        result = get_cached("BTC/USDT", "1h", 0, 1_000_000)
        assert result == []


# ---------------------------------------------------------------------------
# fetch_historical() in cache.py — mocked ccxt network
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_cache_db")
class TestFetchHistorical:
    """fetch_historical() uses cache when available, fetches from ccxt otherwise."""

    @pytest.fixture
    def _patch_cache_db(self, tmp_path, monkeypatch):
        from cryptotrader.backtest import cache as cache_module

        db_path = tmp_path / "ohlcv_cache.db"
        monkeypatch.setattr(cache_module, "CACHE_DB", db_path)

    @pytest.mark.asyncio
    async def test_fetch_historical_uses_cache_when_available(self) -> None:
        """fetch_historical returns cached data when cache covers the full range."""
        from cryptotrader.backtest.cache import fetch_historical, store_ohlcv

        # Pre-populate cache with data starting well before the query range
        since_ms = 1_000_000
        until_ms = 2_000_000
        candles = [
            [since_ms - 10, 100.0, 110.0, 90.0, 105.0, 1.0],
            [since_ms + 100, 105.0, 115.0, 95.0, 110.0, 1.0],
            [until_ms - 100, 110.0, 120.0, 100.0, 115.0, 1.0],
        ]
        store_ohlcv("BTC/USDT", "1h", candles)

        result = await fetch_historical("BTC/USDT", "1h", since_ms, until_ms)

        # Should return from cache (first candle ts <= since_ms + 86_400_000)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fetch_historical_fetches_from_ccxt_when_cache_miss(
        self,
    ) -> None:
        """fetch_historical calls ccxt when cache does not cover the range."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from cryptotrader.backtest.cache import fetch_historical

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv = AsyncMock(
            side_effect=[
                [
                    [5_000_000, 200.0, 210.0, 190.0, 205.0, 50.0],
                    [5_001_000, 205.0, 215.0, 195.0, 210.0, 60.0],
                ],
                [],  # second call returns empty to stop pagination loop
            ]
        )
        mock_exchange.close = AsyncMock()

        mock_binance_cls = MagicMock(return_value=mock_exchange)

        with patch("ccxt.async_support.binance", mock_binance_cls):
            result = await fetch_historical("SOL/USDT", "4h", 5_000_000, 5_002_000)

        # ccxt was called and results returned
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fetch_historical_returns_empty_when_ccxt_returns_empty(
        self,
    ) -> None:
        """fetch_historical returns empty list when ccxt returns no data."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from cryptotrader.backtest.cache import fetch_historical

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=[])
        mock_exchange.close = AsyncMock()

        mock_binance_cls = MagicMock(return_value=mock_exchange)

        with patch("ccxt.async_support.binance", mock_binance_cls):
            result = await fetch_historical("DOGE/USDT", "1d", 10_000_000, 20_000_000)

        assert result == []
