"""Kronos gate_v21 auxiliary feature fetchers.

Three market features required by the Kronos gate_v21 signal that are not
present in the existing DataSnapshot:

1. lsr_top_count      - Binance futures top-trader long/short ACCOUNT ratio
2. premium_close_5d   - 5-day mean of the BTC perp premium index (mark vs index)
3. spy_btc_corr       - 30-day rolling SPY/BTC daily-return correlation

All public functions are async, match the httpx/ccxt async style used by the
rest of cryptotrader.data, and return None (never raise) on any network or
parse failure.  Callers should substitute gate_v21 median fallbacks when None
is returned:
    lsr_top_count    -> 1.526
    premium_close_5d -> 0.0
    spy_btc_corr     -> 0.312

These fetchers are designed for 4-hour cadence (one call per cycle); no
aggressive polling or caching layer is needed.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BINANCE_FAPI = "https://fapi.binance.com"
_RETRY_DELAYS = (1.0, 3.0)  # two retries, 1 s then 3 s backoff
_HTTP_TIMEOUT = 12.0  # seconds
_FOUR_HOURS_MS = 4 * 60 * 60 * 1000
_ONE_DAY_MS = 24 * 60 * 60 * 1000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _get_json(url: str, params: dict | None = None) -> dict | list | None:
    """GET *url* with optional query *params*, return parsed JSON or None."""
    last_exc: Exception | None = None
    delays = list(_RETRY_DELAYS)
    for attempt in range(len(delays) + 1):
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < len(delays):
                await asyncio.sleep(delays[attempt])
    logger.warning("HTTP GET failed for %s after retries: %s", url, last_exc)
    return None


def _closed_interval_rows(rows: list[list], interval_ms: int, now_ms: int) -> list[list]:
    """Drop the currently forming interval from open-time-based rows."""
    return [row for row in rows if row and row[0] + interval_ms <= now_ms]


# ---------------------------------------------------------------------------
# 1. lsr_top_count - Binance top-trader long/short ACCOUNT ratio
# ---------------------------------------------------------------------------


async def fetch_lsr_top_count(symbol: str = "BTCUSDT") -> float | None:
    """Return the latest Binance top-trader long/short ACCOUNT ratio.

    Endpoint:
        GET https://fapi.binance.com/futures/data/topLongShortAccountRatio
        ?symbol=BTCUSDT&period=4h&limit=1

    Returns:
        longShortRatio as float, or None on failure (median fallback: 1.526).
    """
    url = f"{_BINANCE_FAPI}/futures/data/topLongShortAccountRatio"
    params = {"symbol": symbol, "period": "4h", "limit": "1"}
    data = await _get_json(url, params)
    try:
        if isinstance(data, list) and data:
            return float(data[0]["longShortRatio"])
        logger.warning("lsr_top_count: unexpected response shape: %r", data)
    except Exception as exc:
        logger.warning("lsr_top_count: parse error: %s", exc)
    return None


# ---------------------------------------------------------------------------
# 2. premium_close_5d - 5-day mean of perp premium index (stateless, single call)
# ---------------------------------------------------------------------------


async def fetch_premium_close_5d(symbol: str = "BTCUSDT") -> float | None:
    """Return the 5-day mean premium index close via premiumIndexKlines.

    Fetches 30 4-hour klines (= 5 days) and computes mean(close).
    Premium per bar = (markPrice - indexPrice) / indexPrice; the kline close
    field in premiumIndexKlines already encodes this directly as the premium
    index value.

    Endpoint:
        GET https://fapi.binance.com/fapi/v1/premiumIndexKlines
        ?symbol=BTCUSDT&interval=4h&limit=30

    Returns:
        Mean premium-index close as float, or None on failure (median fallback ~0.0).
    """
    url = f"{_BINANCE_FAPI}/fapi/v1/premiumIndexKlines"
    params = {"symbol": symbol, "interval": "4h", "limit": "31"}
    data = await _get_json(url, params)
    try:
        if isinstance(data, list) and data:
            # kline format: [openTime, open, high, low, close, ...]
            import time

            closed = _closed_interval_rows(data, _FOUR_HOURS_MS, int(time.time() * 1000))[-30:]
            closes = [float(row[4]) for row in closed if len(row) >= 5]
            if closes:
                return sum(closes) / len(closes)
        logger.warning("premium_close_5d: unexpected response shape: %r", data)
    except Exception as exc:
        logger.warning("premium_close_5d: parse error: %s", exc)
    return None


# ---------------------------------------------------------------------------
# 3. spy_btc_corr - 30-day rolling SPY/BTC daily-return correlation
# ---------------------------------------------------------------------------


async def fetch_spy_btc_corr_30d() -> float | None:
    """Return the latest 30-day rolling correlation between SPY and BTC daily returns.

    Data sources:
    - BTC/USDT daily OHLCV: ccxt Binance (async), 65 days to seed 30-day window
    - SPY daily closes: yfinance (lazy import) with 65-day lookback

    If yfinance is not installed the function logs a warning and returns None.
    Median fallback: 0.312.

    Returns:
        Latest 30-day rolling Pearson correlation as float, or None on failure.
    """
    try:
        import yfinance as yf  # lazy - not in base deps; add 'yfinance' to pyproject.toml
    except ImportError:
        logger.warning(
            "spy_btc_corr: yfinance not installed - add 'yfinance' to project deps. "
            "Returning None; caller should use median fallback 0.312."
        )
        return None

    import ccxt.async_support as ccxt
    import numpy as np
    import pandas as pd

    lookback_days = 65  # 65 calendar days >= 30 trading days to seed window

    # --- BTC daily via ccxt (async) ---
    btc_rows: list[list] = []
    # ccxt ≥4.5 rejects "swap" in fetchMarkets list. Spot daily OHLCV suffices
    # for the SPY/BTC correlation feature.
    exchange = ccxt.binance({"enableRateLimit": True, "options": {"fetchMarkets": ["spot"]}})
    try:
        import time as _time

        now_ms = int(_time.time() * 1000)
        since_ms = now_ms - lookback_days * _ONE_DAY_MS
        btc_raw = await exchange.fetch_ohlcv("BTC/USDT", timeframe="1d", since=since_ms, limit=lookback_days + 5)
        btc_rows = _closed_interval_rows(btc_raw or [], _ONE_DAY_MS, now_ms)
    except Exception as exc:
        logger.warning("spy_btc_corr: ccxt BTC fetch failed: %s", exc)
        return None
    finally:
        await exchange.close()

    if not btc_rows:
        logger.warning("spy_btc_corr: empty BTC OHLCV data")
        return None

    # --- SPY daily via yfinance (sync, run in executor thread) ---
    from datetime import UTC, datetime, timedelta

    end_dt = datetime.now(tz=UTC)
    start_dt = end_dt - timedelta(days=lookback_days)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")  # exclusive: never include today's partial session

    try:
        spy_raw = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: yf.download("SPY", start=start_str, end=end_str, interval="1d", auto_adjust=True, progress=False),
        )
    except Exception as exc:
        logger.warning("spy_btc_corr: yfinance SPY fetch failed: %s", exc)
        return None

    if spy_raw is None or (hasattr(spy_raw, "empty") and spy_raw.empty):
        logger.warning("spy_btc_corr: yfinance returned empty DataFrame for SPY")
        return None

    try:
        # Flatten multi-level columns if present (yfinance >= 0.2 returns MultiIndex)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = [str(c[0]).lower() for c in spy_raw.columns]
        else:
            spy_raw.columns = [str(c).lower() for c in spy_raw.columns]

        spy = spy_raw[["close"]].rename(columns={"close": "spy_close"}).copy()
        spy.index = pd.to_datetime(spy.index, utc=True).floor("1D")

        # Build BTC DataFrame
        btc_df = pd.DataFrame(btc_rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
        btc_df["timestamp"] = pd.to_datetime(btc_df["ts_ms"], unit="ms", utc=True).dt.floor("1D")
        btc_df = btc_df.set_index("timestamp")[["close"]].rename(columns={"close": "btc_close"})

        # Align and merge on calendar date
        merged = spy.join(btc_df, how="inner")
        if len(merged) < 20:
            logger.warning("spy_btc_corr: insufficient aligned rows (%d < 20)", len(merged))
            return None

        spy_ret = merged["spy_close"].pct_change()
        btc_ret = merged["btc_close"].pct_change()

        rolling_corr = spy_ret.rolling(window=30, min_periods=20).corr(btc_ret)
        non_nan = rolling_corr.dropna()
        last_val = non_nan.iloc[-1] if not non_nan.empty else None

        if last_val is None or np.isnan(last_val):
            logger.warning("spy_btc_corr: rolling correlation is NaN")
            return None

        return float(last_val)

    except Exception as exc:
        logger.warning("spy_btc_corr: computation error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


async def fetch_kronos_aux(symbol: str = "BTCUSDT") -> dict[str, float | None]:
    """Fetch all three Kronos gate_v21 auxiliary features concurrently.

    Returns a dict with keys:
        "lsr_top_count"    - Binance top-trader L/S account ratio  (fallback: 1.526)
        "premium_close_5d" - 5-day mean perp premium index close   (fallback: 0.0)
        "spy_btc_corr"     - 30-day SPY/BTC daily-return corr      (fallback: 0.312)

    Any value that is None means the fetch failed; callers should substitute
    the gate_v21 median fallback for that feature.
    """
    lsr, premium, corr = await asyncio.gather(
        fetch_lsr_top_count(symbol),
        fetch_premium_close_5d(symbol),
        fetch_spy_btc_corr_30d(),
        return_exceptions=False,
    )
    return {
        "lsr_top_count": lsr,
        "premium_close_5d": premium,
        "spy_btc_corr": corr,
    }


# ---------------------------------------------------------------------------
# Smoke test (manual, requires live network)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    async def _smoke() -> None:
        print("Fetching Kronos aux features (live Binance + yfinance)...")  # noqa: T201
        result = await fetch_kronos_aux()
        pprint.pprint(result)  # noqa: T203
        medians = {"lsr_top_count": 1.526, "premium_close_5d": 0.0, "spy_btc_corr": 0.312}
        print("\nWith fallbacks applied:")  # noqa: T201
        pprint.pprint({k: (v if v is not None else medians[k]) for k, v in result.items()})  # noqa: T203

    asyncio.run(_smoke())
