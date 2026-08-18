"""Market data collector using ccxt async.

OHLCV data is cached in the unified store. Ticker and orderbook are always fetched live.
"""

from __future__ import annotations

import logging
import time

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd

from cryptotrader.ccxt_options import fetch_market_types
from cryptotrader.data.store import cache_result, get_cached_or_none
from cryptotrader.models import MarketData

logger = logging.getLogger(__name__)

_TIMEFRAME_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}


def _timeframe_ms(timeframe: str) -> int:
    try:
        return int(timeframe[:-1]) * _TIMEFRAME_UNIT_MS[timeframe[-1]]
    except (KeyError, TypeError, ValueError):
        return 3_600_000


def _closed_ohlcv(rows: list, timeframe: str, now_ms: int) -> list:
    """Return only candles whose full interval has elapsed."""
    tf_ms = _timeframe_ms(timeframe)
    current_open_ms = now_ms - now_ms % tf_ms
    return [row for row in rows if row and row[0] < current_open_ms]


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

        # fetchMarkets restricts load_markets to types we actually use — see
        # comment in execution/exchange.py for why future/option are excluded.
        exchange: ccxt.Exchange = getattr(ccxt, exchange_id)(
            {"options": {"fetchMarkets": fetch_market_types(exchange_id)}}
        )
        try:
            await exchange.load_markets()

            # Validate OHLCV cache against the latest fully closed candle. A candle
            # that is still forming must never become strategy input or cache state.
            use_cache = False
            live_now_ms = int(time.time() * 1000)
            if isinstance(cached_ohlcv, list):
                if date is not None:
                    # Backtest mode: skip wall-clock staleness check — cached data is correct for this date
                    use_cache = len(cached_ohlcv) >= limit
                else:
                    cached_ohlcv = _closed_ohlcv(cached_ohlcv, timeframe, live_now_ms)[-limit:]
                    expected_last_open = live_now_ms - live_now_ms % _timeframe_ms(timeframe) - _timeframe_ms(timeframe)
                    last_ts_ms = cached_ohlcv[-1][0] if cached_ohlcv else 0
                    use_cache = len(cached_ohlcv) >= limit and last_ts_ms >= expected_last_open

            if use_cache:
                logger.debug("Using cached OHLCV for %s %s (%d bars)", pair, timeframe, len(cached_ohlcv))
                df = pd.DataFrame(cached_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            else:
                fetch_limit = limit if date is not None else limit + 1
                fetch_params = {"paginate": True} if fetch_limit > 300 else {}
                ohlcv_raw = await exchange.fetch_ohlcv(pair, timeframe, limit=fetch_limit, params=fetch_params)
                if date is None:
                    ohlcv_raw = _closed_ohlcv(ohlcv_raw, timeframe, live_now_ms)[-limit:]
                df = pd.DataFrame(ohlcv_raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                # Live cache contains closed candles only.
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
        ex = exchange_cls({"enableRateLimit": True, "options": {"fetchMarkets": fetch_market_types(exchange_id)}})
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
    exchange = ccxt.binance({"enableRateLimit": True, "options": {"fetchMarkets": fetch_market_types("binance")}})
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
