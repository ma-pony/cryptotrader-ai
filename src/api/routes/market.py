"""Market data endpoints (FR-700~719)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/market")


class Liquidations24h(BaseModel):
    long: float = 0.0
    short: float = 0.0


class MarketDataResponse(BaseModel):
    funding_rate: float | None = None
    open_interest: float | None = None
    liquidations_24h: Liquidations24h = Liquidations24h()


@router.get("/{pair}", response_model=MarketDataResponse)
async def get_market_data(pair: str, exchange: str = Query(default="binance")):
    """Return funding rate, OI, and 24h liquidations for a pair."""
    pair_symbol = pair.replace("-", "/")
    try:
        from cryptotrader.data.market import MarketDataService

        service = MarketDataService()
        snapshot = await service.get_market_snapshot(pair_symbol, exchange)
        return MarketDataResponse(
            funding_rate=snapshot.get("funding_rate"),
            open_interest=snapshot.get("open_interest"),
            liquidations_24h=Liquidations24h(
                long=snapshot.get("liquidations_long_24h", 0),
                short=snapshot.get("liquidations_short_24h", 0),
            ),
        )
    except Exception:
        logger.debug("Market data fetch failed for %s on %s", pair_symbol, exchange, exc_info=True)
        return MarketDataResponse()
