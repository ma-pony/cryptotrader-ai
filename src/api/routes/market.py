"""Market data endpoints (FR-700~719)."""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/market")

# Security (I-S1/I-S2 · Phase 4, 2026-04-24):
#
# - Trading pairs are validated via ``_PAIR_RE``: uppercase alnum + optional `/` + 1-6
#   digits max. This blocks `?`, `&`, `=`, `%`, path traversal, and any unicode
#   normalisation attack against the Binance FAPI URL template.
# - Exchange identifiers come from a hard allowlist — ``getattr(ccxt, user_input)``
#   is no longer trusted to only resolve to exchange classes.
_PAIR_RE = re.compile(r"^[A-Z0-9]{2,10}/[A-Z0-9]{2,10}$")
_ALLOWED_EXCHANGES: frozenset[str] = frozenset({"binance", "okx", "bybit", "coinbase"})


def _validate_pair(pair: str) -> str:
    """Normalise ``pair`` (accepting ``BTC-USDT`` for URL safety) and reject anything else.

    Returns the canonical ``BTC/USDT`` form. Raises HTTPException 400 on invalid input.
    """
    normalized = pair.replace("-", "/").upper()
    if not _PAIR_RE.match(normalized):
        raise HTTPException(status_code=400, detail=f"Invalid pair format: {pair!r}")
    return normalized


def _validate_exchange(exchange: str) -> str:
    """Reject any exchange not in the hard allowlist."""
    low = exchange.lower()
    if low not in _ALLOWED_EXCHANGES:
        raise HTTPException(status_code=400, detail=f"Unsupported exchange: {exchange!r}")
    return low


class Liquidations24h(BaseModel):
    long: float = 0.0
    short: float = 0.0


class MarketDataResponse(BaseModel):
    funding_rate: float | None = None
    open_interest: float | None = None
    liquidations_24h: Liquidations24h = Liquidations24h()
    # Alignment with frontend prototype (2026-04-24):
    long_short_ratio: float | None = None
    top_traders_long_share: float | None = None  # fraction of top traders net long


async def _fetch_long_short(pair_symbol: str, exchange: str) -> tuple[float | None, float | None]:
    """Pull Binance long/short ratio endpoints for the given pair.

    Binance futures exposes two ratios under /futures/data:
      - globalLongShortAccountRatio → `long_short_ratio`
      - topLongShortPositionRatio   → top-trader net long share

    Other exchanges return ``(None, None)``. ``pair_symbol`` must already be
    validated by :func:`_validate_pair` — the allow-listed alphabet guarantees
    httpx query-string safety even under f-string construction.
    """
    if exchange != "binance":
        return None, None
    base_quote = pair_symbol.replace("/", "")
    # Defence in depth: even though the route layer validated the pair, we re-enforce
    # the alphabet here so any future caller can't bypass the URL safety guarantee.
    if not re.fullmatch(r"[A-Z0-9]{4,20}", base_quote):
        logger.warning("long/short: rejecting non-alphanumeric symbol %r", base_quote)
        return None, None
    params = {"symbol": base_quote, "period": "5m", "limit": "1"}
    ls_ratio: float | None = None
    top_share: float | None = None
    try:
        import httpx

        async with httpx.AsyncClient(timeout=6.0) as client:
            g_resp, t_resp = await __import__("asyncio").gather(
                client.get(
                    "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                    params=params,
                ),
                client.get(
                    "https://fapi.binance.com/futures/data/topLongShortPositionRatio",
                    params=params,
                ),
                return_exceptions=True,
            )
            if not isinstance(g_resp, Exception) and g_resp.status_code == 200:
                try:
                    data = g_resp.json()
                    if isinstance(data, list) and data:
                        ls_ratio = float(data[0].get("longShortRatio", 0.0) or 0.0)
                except (ValueError, KeyError, TypeError):
                    logger.debug("long/short global parse failed", exc_info=True)
            if not isinstance(t_resp, Exception) and t_resp.status_code == 200:
                try:
                    data = t_resp.json()
                    if isinstance(data, list) and data:
                        long_share = float(data[0].get("longAccount", 0.0) or 0.0)
                        top_share = long_share if long_share > 0 else None
                except (ValueError, KeyError, TypeError):
                    logger.debug("long/short top parse failed", exc_info=True)
    except Exception:
        logger.debug("long/short ratio fetch failed for %s", pair_symbol, exc_info=True)
    return ls_ratio, top_share


@router.get("/{pair}", response_model=MarketDataResponse)
async def get_market_data(pair: str, exchange: str = Query(default="binance")):
    """Return funding rate, OI, 24h liquidations, and long/short ratios."""
    pair_symbol = _validate_pair(pair)
    exchange = _validate_exchange(exchange)
    try:
        from cryptotrader.data.market import MarketDataService

        service = MarketDataService()
        snapshot = await service.get_market_snapshot(pair_symbol, exchange)
        ls_ratio, top_share = await _fetch_long_short(pair_symbol, exchange)
        return MarketDataResponse(
            funding_rate=snapshot.get("funding_rate"),
            open_interest=snapshot.get("open_interest"),
            liquidations_24h=Liquidations24h(
                long=snapshot.get("liquidations_long_24h", 0),
                short=snapshot.get("liquidations_short_24h", 0),
            ),
            long_short_ratio=ls_ratio,
            top_traders_long_share=top_share,
        )
    except Exception:
        logger.debug("Market data fetch failed for %s on %s", pair_symbol, exchange, exc_info=True)
        return MarketDataResponse()


class OHLCVBar(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class OHLCVResponse(BaseModel):
    bars: list[OHLCVBar] = Field(default_factory=list)


_ALLOWED_TIMEFRAMES: frozenset[str] = frozenset({"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"})


@router.get("/{pair}/ohlcv", response_model=OHLCVResponse)
async def get_ohlcv(
    pair: str,
    timeframe: str = Query(default="1h"),
    limit: int = Query(default=100, ge=1, le=1000),
    exchange: str = Query(default="binance"),
):
    """Return OHLCV candlestick data for the given pair."""
    pair_symbol = _validate_pair(pair)
    exchange = _validate_exchange(exchange)
    if timeframe not in _ALLOWED_TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {timeframe!r}")
    try:
        import ccxt.async_support as ccxt

        # Even though _validate_exchange bounds the input, re-check via hasattr
        # so a ccxt version that drops a listed exchange yields 404-equivalent empty
        # response rather than AttributeError.
        exchange_cls = getattr(ccxt, exchange, None)
        if exchange_cls is None:
            return OHLCVResponse()
        ex = exchange_cls()
        try:
            raw = await ex.fetch_ohlcv(pair_symbol, timeframe, limit=limit)
        finally:
            await ex.close()
        bars = [OHLCVBar(time=int(c[0]), open=c[1], high=c[2], low=c[3], close=c[4], volume=c[5]) for c in raw]
        return OHLCVResponse(bars=bars)
    except Exception:
        logger.debug("OHLCV fetch failed for %s %s on %s", pair_symbol, timeframe, exchange, exc_info=True)
        return OHLCVResponse()
