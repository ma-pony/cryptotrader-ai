"""Market data collector using ccxt async.

OHLCV data is cached in the unified store. Ticker and orderbook are always fetched live.
"""

from __future__ import annotations

import logging
import time

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd

from cryptotrader.data.store import cache_result, get_cached_or_none
from cryptotrader.models import MarketData

logger = logging.getLogger(__name__)


class MarketCollector:
    async def collect(
        self,
        pair: str,
        exchange_id: str = "binance",
        timeframe: str = "1h",
        limit: int = 100,
        date: str | None = None,
    ) -> MarketData:
        """Collect market data.

        Args:
            date: If provided, use date-specific store lookup (backtest mode).
        """
        # Check OHLCV cache first
        ohlcv_key = f"ohlcv_{pair.replace('/', '_')}_{timeframe}"
        cached_ohlcv = get_cached_or_none(ohlcv_key, date=date)

        exchange: ccxt.Exchange = getattr(ccxt, exchange_id)()
        try:
            await exchange.load_markets()

            # Validate OHLCV cache: must have enough bars AND last bar must be recent
            use_cache = False
            if cached_ohlcv is not None and isinstance(cached_ohlcv, list) and len(cached_ohlcv) >= limit * 0.8:
                if date is not None:
                    # Backtest mode: skip wall-clock staleness check — cached data is correct for this date
                    use_cache = True
                else:
                    # Live mode: check if the last candle timestamp is within 2x the timeframe
                    _tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
                    tf_sec = _tf_seconds.get(timeframe, 3600)
                    last_ts_ms = cached_ohlcv[-1][0] if cached_ohlcv[-1] else 0
                    age_sec = time.time() - last_ts_ms / 1000
                    use_cache = age_sec < tf_sec * 2

            if use_cache:
                logger.debug("Using cached OHLCV for %s %s (%d bars)", pair, timeframe, len(cached_ohlcv))
                df = pd.DataFrame(cached_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            else:
                ohlcv_raw = await exchange.fetch_ohlcv(pair, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv_raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                # Cache raw OHLCV data
                cache_result(ohlcv_key, ohlcv_raw)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Ticker — always fetch live for current price
            ticker = await exchange.fetch_ticker(pair)

            # Funding rate — use cache if available
            funding_key = f"funding_rate_{pair.replace('/', '_')}"
            cached_fr = get_cached_or_none(funding_key, date=date)
            if cached_fr is not None:
                funding_rate = float(cached_fr) if isinstance(cached_fr, int | float) else 0.0
            else:
                try:
                    funding = await exchange.fetch_funding_rate(pair)
                    funding_rate = float(funding.get("fundingRate", 0.0) or 0.0)
                    cache_result(funding_key, funding_rate)
                except Exception:
                    logger.debug("Funding rate fetch failed for %s", pair, exc_info=True)
                    funding_rate = 0.0

            # Orderbook — always fetch live
            orderbook = await exchange.fetch_order_book(pair, limit=10)
            bid_vol = sum(b[1] for b in orderbook["bids"][:10])
            ask_vol = sum(a[1] for a in orderbook["asks"][:10])
            total = bid_vol + ask_vol
            orderbook_imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0

            returns = df["close"].pct_change().dropna()
            volatility = float(np.std(returns)) if len(returns) > 0 else 0.0

            return MarketData(
                pair=pair,
                ohlcv=df,
                ticker=ticker,
                funding_rate=funding_rate,
                orderbook_imbalance=orderbook_imbalance,
                volatility=volatility,
            )
        finally:
            await exchange.close()
