"""Market data collector using ccxt async."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt

from cryptotrader.models import MarketData


class MarketCollector:

    async def collect(
        self,
        pair: str,
        exchange_id: str = "binance",
        timeframe: str = "1h",
        limit: int = 100,
    ) -> MarketData:
        exchange: ccxt.Exchange = getattr(ccxt, exchange_id)()
        try:
            await exchange.load_markets()

            ohlcv_raw = await exchange.fetch_ohlcv(pair, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv_raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            ticker = await exchange.fetch_ticker(pair)

            try:
                funding = await exchange.fetch_funding_rate(pair)
                funding_rate = float(funding.get("fundingRate", 0.0) or 0.0)
            except Exception:
                funding_rate = 0.0

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
