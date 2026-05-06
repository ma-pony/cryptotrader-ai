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
        exchange_id: str = "",
        timeframe: str = "1h",
        limit: int = 100,
        date: str | None = None,
    ) -> MarketData:
        """Collect market data.

        Args:
            date: If provided, use date-specific store lookup (backtest mode).
        """
        if not exchange_id:
            from cryptotrader.config import load_config

            exchange_id = load_config().exchange_id

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
                    logger.warning("Funding rate fetch failed for %s", pair, exc_info=True)
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


class MarketDataService:
    """Thin aggregator over ccxt for the /api/market/{pair} route.

    Returns funding rate + open interest in a single dict. Liquidation totals
    are surfaced as zeros until a CoinGlass-style provider is wired in — the
    route falls back gracefully on missing data.
    """

    async def get_market_snapshot(self, pair: str, exchange_id: str) -> dict:
        snapshot: dict = {
            "funding_rate": None,
            "open_interest": None,
            "liquidations_long_24h": 0.0,
            "liquidations_short_24h": 0.0,
        }
        exchange_cls = getattr(ccxt, exchange_id, None)
        if exchange_cls is None:
            return snapshot
        ex = exchange_cls({"enableRateLimit": True})
        try:
            try:
                fr = await ex.fetch_funding_rate(pair)
                snapshot["funding_rate"] = fr.get("fundingRate") if isinstance(fr, dict) else None
            except Exception:
                logger.warning("funding_rate fetch failed for %s on %s", pair, exchange_id, exc_info=True)
            try:
                oi = await ex.fetch_open_interest(pair)
                snapshot["open_interest"] = (
                    oi.get("openInterestAmount") or oi.get("openInterest") if isinstance(oi, dict) else None
                )
            except Exception:
                logger.warning("open_interest fetch failed for %s on %s", pair, exchange_id, exc_info=True)
        finally:
            await ex.close()
        return snapshot


async def fetch_klines_binance(symbol: str = "BTC", interval: str = "1h", limit: int = 100) -> dict:
    """Fetch K-line data via ccxt Binance. Returns {"klines": [{"t", "o", "h", "l", "c", "v"}, ...]}."""
    result: dict = {"klines": []}
    pair = f"{symbol}/USDT"
    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        ohlcv = await exchange.fetch_ohlcv(pair, timeframe=interval, limit=limit)
        result["klines"] = [
            {"t": row[0], "o": row[1], "h": row[2], "l": row[3], "c": row[4], "v": row[5]} for row in ohlcv
        ]
    except Exception:
        logger.warning("Binance klines fetch failed for %s", pair, exc_info=True)
    finally:
        await exchange.close()
    return result
