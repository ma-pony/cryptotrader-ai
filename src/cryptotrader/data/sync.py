"""Data sync — fetch and persist historical data from all sources.

Fetches historical data once, stores in SQLite, then only fetches incremental updates.
Respects rate limits to avoid being blocked.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import httpx

from cryptotrader.data.store import (
    _record_fetch,
    _should_fetch,
    count_records,
    get_latest,
    store_batch,
    store_data,
)

logger = logging.getLogger(__name__)


async def sync_sosovalue_etf_history(api_key: str) -> int:
    """Fetch up to 300 days of ETF historical inflow data and persist."""
    if not api_key:
        return 0
    if not _should_fetch("sosovalue_etf_history"):
        existing = count_records("sosovalue_etf")
        logger.debug("SoSoValue ETF history rate-limited, %d records cached", existing)
        return existing

    from cryptotrader.data.providers.sosovalue import fetch_etf_history

    history = await fetch_etf_history(api_key)
    if not history:
        return 0

    records = []
    for item in history:
        date = item.get("date", "")
        if not date:
            continue
        records.append(
            (
                date,
                {
                    "totalNetInflow": item.get("totalNetInflow", 0),
                    "totalValueTraded": item.get("totalValueTraded", 0),
                    "totalNetAssets": item.get("totalNetAssets", 0),
                    "cumNetInflow": item.get("cumNetInflow", 0),
                },
            )
        )

    store_batch("sosovalue_etf", records, forward_fill=True)
    _record_fetch("sosovalue_etf_history")
    logger.info("Synced %d days of SoSoValue ETF history", len(records))
    return len(records)


async def sync_sosovalue_etf_current(api_key: str) -> dict:
    """Fetch current ETF metrics and persist today's data."""
    if not api_key:
        return {}
    if not _should_fetch("sosovalue_etf_metrics"):
        latest = get_latest("sosovalue_etf", limit=1)
        if latest:
            return latest[0][1]
        return {}

    from cryptotrader.data.providers.sosovalue import fetch_etf_metrics

    data = await fetch_etf_metrics(api_key)
    if data:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        store_data("sosovalue_etf", today, data)
        _record_fetch("sosovalue_etf_metrics")
    return data


async def sync_fear_greed_history() -> int:
    """Fetch historical Fear & Greed index (up to 365 days) and persist."""
    if not _should_fetch("fear_greed"):
        return count_records("fear_greed")

    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get("https://api.alternative.me/fng/?limit=365")
            r.raise_for_status()
            items = r.json().get("data", [])

        records = []
        for item in items:
            ts = int(item.get("timestamp", 0))
            if ts > 0:
                date = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                records.append((date, int(item["value"])))

        store_batch("fear_greed", records)
        _record_fetch("fear_greed")
        logger.info("Synced %d days of Fear & Greed history", len(records))
        return len(records)
    except Exception:
        logger.warning("Fear & Greed history sync failed", exc_info=True)
        return 0


async def sync_fred_history(api_key: str, series: str = "DFF", limit: int = 365) -> int:
    """Fetch historical FRED data and persist."""
    source_key = f"fred_{series}"
    if not api_key:
        return 0
    if not _should_fetch(source_key):
        return count_records(source_key)

    try:
        start = (datetime.now(UTC) - timedelta(days=limit)).strftime("%Y-%m-%d")
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series,
                    "api_key": api_key,
                    "observation_start": start,
                    "file_type": "json",
                },
            )
            r.raise_for_status()
            obs = r.json().get("observations", [])

        records = []
        for o in obs:
            date = o.get("date", "")
            val = o.get("value", ".")
            if date and val != ".":
                records.append((date, float(val)))

        store_batch(source_key, records, forward_fill=True)
        _record_fetch(source_key)
        logger.info("Synced %d observations for FRED/%s", len(records), series)
        return len(records)
    except Exception:
        logger.warning("FRED %s history sync failed", series, exc_info=True)
        return 0


async def sync_sosovalue_news(api_key: str) -> int:
    """Fetch latest news and persist."""
    if not api_key:
        return 0
    if not _should_fetch("sosovalue_news"):
        return count_records("sosovalue_news")

    from cryptotrader.data.providers.sosovalue import fetch_news

    news = await fetch_news(api_key, page_size=50)
    if not news:
        return 0

    records = []
    for item in news:
        ts = item.get("releaseTime", 0)
        if ts > 0:
            date = datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d %H:%M")
            records.append((date, item))

    store_batch("sosovalue_news", records)
    _record_fetch("sosovalue_news")
    logger.info("Synced %d news items", len(records))
    return len(records)


async def sync_binance_derivatives(symbol: str = "BTC", days: int = 365) -> int:
    """Fetch Binance futures historical data: OI, long/short ratio, taker ratio."""
    if not _should_fetch("binance_derivatives"):
        return count_records(f"binance_oi_{symbol}")

    pair = f"{symbol}USDT"
    base = "https://fapi.binance.com/futures/data"
    total_stored = 0

    async with httpx.AsyncClient(timeout=15) as c:
        # OI history
        try:
            records = []
            r = await c.get(
                f"{base}/openInterestHist",
                params={"symbol": pair, "period": "1d", "limit": 500},
            )
            r.raise_for_status()
            for item in r.json():
                date = datetime.fromtimestamp(item["timestamp"] / 1000, UTC).strftime("%Y-%m-%d")
                records.append(
                    (
                        date,
                        {
                            "openInterest": float(item.get("sumOpenInterest", 0)),
                            "openInterestValue": float(item.get("sumOpenInterestValue", 0)),
                        },
                    )
                )
            store_batch(f"binance_oi_{symbol}", records)
            total_stored += len(records)
            logger.info("Synced %d days of Binance OI for %s", len(records), symbol)
        except Exception:
            logger.warning("Binance OI history sync failed", exc_info=True)

        await asyncio.sleep(0.3)

        # Long/Short ratio history
        try:
            records = []
            r = await c.get(
                f"{base}/globalLongShortAccountRatio",
                params={"symbol": pair, "period": "1d", "limit": 500},
            )
            r.raise_for_status()
            for item in r.json():
                date = datetime.fromtimestamp(item["timestamp"] / 1000, UTC).strftime("%Y-%m-%d")
                records.append(
                    (
                        date,
                        {
                            "longShortRatio": float(item.get("longShortRatio", 1)),
                            "longAccount": float(item.get("longAccount", 0.5)),
                            "shortAccount": float(item.get("shortAccount", 0.5)),
                        },
                    )
                )
            store_batch(f"binance_ls_ratio_{symbol}", records)
            total_stored += len(records)
            logger.info("Synced %d days of Binance long/short ratio", len(records))
        except Exception:
            logger.warning("Binance long/short ratio sync failed", exc_info=True)

        await asyncio.sleep(0.3)

        # Top trader ratio history
        try:
            records = []
            r = await c.get(
                f"{base}/topLongShortPositionRatio",
                params={"symbol": pair, "period": "1d", "limit": 500},
            )
            r.raise_for_status()
            for item in r.json():
                date = datetime.fromtimestamp(item["timestamp"] / 1000, UTC).strftime("%Y-%m-%d")
                records.append(
                    (
                        date,
                        {
                            "topTraderRatio": float(item.get("longShortRatio", 1)),
                            "longAccount": float(item.get("longAccount", 0.5)),
                            "shortAccount": float(item.get("shortAccount", 0.5)),
                        },
                    )
                )
            store_batch(f"binance_top_trader_{symbol}", records)
            total_stored += len(records)
            logger.info("Synced %d days of Binance top trader ratio", len(records))
        except Exception:
            logger.warning("Binance top trader ratio sync failed", exc_info=True)

        await asyncio.sleep(0.3)

        # Taker buy/sell ratio history
        try:
            records = []
            r = await c.get(
                f"{base}/takerlongshortRatio",
                params={"symbol": pair, "period": "1d", "limit": 500},
            )
            r.raise_for_status()
            for item in r.json():
                date = datetime.fromtimestamp(item["timestamp"] / 1000, UTC).strftime("%Y-%m-%d")
                records.append(
                    (
                        date,
                        {
                            "buySellRatio": float(item.get("buySellRatio", 1)),
                            "buyVol": float(item.get("buyVol", 0)),
                            "sellVol": float(item.get("sellVol", 0)),
                        },
                    )
                )
            store_batch(f"binance_taker_{symbol}", records)
            total_stored += len(records)
            logger.info("Synced %d days of Binance taker buy/sell ratio", len(records))
        except Exception:
            logger.warning("Binance taker ratio sync failed", exc_info=True)

    _record_fetch("binance_derivatives")
    return total_stored


async def sync_defillama_tvl(days: int = 365) -> int:
    """Fetch DeFi TVL history from DefiLlama (free, no key)."""
    if not _should_fetch("defillama"):
        return count_records("defillama_tvl")

    try:
        async with httpx.AsyncClient(timeout=15, verify=False) as c:
            r = await c.get("https://api.llama.fi/v2/historicalChainTvl/Ethereum")
            r.raise_for_status()
            data = r.json()

        cutoff = (datetime.now(UTC) - timedelta(days=days)).timestamp()
        records = []
        for item in data:
            ts = item.get("date", 0)
            if ts >= cutoff:
                date = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                records.append((date, {"tvl": item.get("tvl", 0)}))

        store_batch("defillama_tvl", records)
        _record_fetch("defillama")
        logger.info("Synced %d days of DefiLlama TVL", len(records))
        return len(records)
    except Exception:
        logger.warning("DefiLlama TVL sync failed", exc_info=True)
        return 0


async def sync_coingecko_market(days: int = 365) -> int:
    """Fetch BTC market data (price, market cap, volume) from CoinGecko (free)."""
    if not _should_fetch("coingecko"):
        return count_records("coingecko_btc")

    try:
        async with httpx.AsyncClient(timeout=15, verify=False) as c:
            r = await c.get(
                "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                params={"vs_currency": "usd", "days": days, "interval": "daily"},
            )
            r.raise_for_status()
            data = r.json()

        prices = data.get("prices", [])
        mcaps = data.get("market_caps", [])
        volumes = data.get("total_volumes", [])

        # Build date-indexed records
        by_date: dict[str, dict] = {}
        for ts, price in prices:
            date = datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d")
            by_date.setdefault(date, {})["price"] = price
        for ts, mcap in mcaps:
            date = datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d")
            by_date.setdefault(date, {})["market_cap"] = mcap
        for ts, vol in volumes:
            date = datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d")
            by_date.setdefault(date, {})["total_volume"] = vol

        records = sorted(by_date.items())
        store_batch("coingecko_btc", records)
        _record_fetch("coingecko")
        logger.info("Synced %d days of CoinGecko BTC market data", len(records))
        return len(records)
    except Exception:
        logger.warning("CoinGecko BTC market sync failed", exc_info=True)
        return 0


async def sync_fred_multi(api_key: str) -> int:
    """Fetch multiple FRED macro series: yield curve, VIX, S&P500, M2, CPI."""
    if not api_key:
        return 0
    source_key = "fred_multi"
    if not _should_fetch(source_key):
        return sum(count_records(f"fred_{s}") for s in ("T10Y2Y", "VIXCLS", "SP500", "WM2NS", "CPIAUCSL"))

    series_list = {
        "T10Y2Y": "yield_curve_spread",
        "VIXCLS": "vix",
        "SP500": "sp500",
        "WM2NS": "m2_money_supply",
        "CPIAUCSL": "cpi",
    }
    total = 0
    start = (datetime.now(UTC) - timedelta(days=365)).strftime("%Y-%m-%d")

    async with httpx.AsyncClient(timeout=15) as c:
        for series_id, label in series_list.items():
            try:
                r = await c.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": api_key,
                        "observation_start": start,
                        "file_type": "json",
                    },
                )
                r.raise_for_status()
                obs = r.json().get("observations", [])
                records = [(o["date"], float(o["value"])) for o in obs if o.get("value", ".") != "."]
                store_batch(f"fred_{series_id}", records, forward_fill=True)
                total += len(records)
                logger.info("Synced %d observations for FRED/%s (%s)", len(records), series_id, label)
                await asyncio.sleep(0.2)
            except Exception:
                logger.warning("FRED %s sync failed", series_id, exc_info=True)

    _record_fetch(source_key)
    return total


async def _sync_defillama_tvl(c, cutoff: float) -> int:
    """Sync total TVL across all chains."""
    try:
        r = await c.get("https://api.llama.fi/v2/historicalChainTvl")
        r.raise_for_status()
        records = [
            (datetime.fromtimestamp(item["date"], UTC).strftime("%Y-%m-%d"), {"tvl": item.get("tvl", 0)})
            for item in r.json()
            if item.get("date", 0) >= cutoff
        ]
        store_batch("defillama_total_tvl", records)
        logger.info("Synced %d days of DefiLlama total TVL", len(records))
        return len(records)
    except Exception:
        logger.warning("DefiLlama total TVL sync failed", exc_info=True)
        return 0


async def _sync_defillama_stablecoin(c, cutoff: float) -> int:
    """Sync stablecoin total supply (USDT = id 1)."""
    try:
        r = await c.get("https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1")
        r.raise_for_status()
        records = []
        for item in r.json():
            ts = int(item.get("date", 0))
            if ts >= cutoff:
                date = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                circ = item.get("totalCirculatingUSD", {}).get("peggedUSD", 0)
                records.append((date, {"usdt_supply": circ}))
        store_batch("defillama_stablecoin", records)
        logger.info("Synced %d days of stablecoin supply", len(records))
        return len(records)
    except Exception:
        logger.warning("DefiLlama stablecoin sync failed", exc_info=True)
        return 0


async def _sync_defillama_dex_volume(c, cutoff: float) -> int:
    """Sync DEX volume."""
    try:
        r = await c.get(
            "https://api.llama.fi/overview/dexs",
            params={"excludeTotalDataChart": "false", "excludeTotalDataChartBreakdown": "true"},
        )
        r.raise_for_status()
        records = [
            (datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d"), {"dex_volume": vol})
            for ts, vol in r.json().get("totalDataChart", [])
            if ts >= cutoff
        ]
        store_batch("defillama_dex_vol", records)
        logger.info("Synced %d days of DEX volume", len(records))
        return len(records)
    except Exception:
        logger.warning("DefiLlama DEX volume sync failed", exc_info=True)
        return 0


async def sync_defillama_extra() -> int:
    """Fetch DefiLlama extra data: total TVL (all chains), stablecoin supply, DEX volume."""
    if not _should_fetch("defillama_extra"):
        return count_records("defillama_total_tvl")

    cutoff = (datetime.now(UTC) - timedelta(days=365)).timestamp()

    async with httpx.AsyncClient(timeout=15, verify=False) as c:
        total = await _sync_defillama_tvl(c, cutoff)
        await asyncio.sleep(0.3)
        total += await _sync_defillama_stablecoin(c, cutoff)
        await asyncio.sleep(0.3)
        total += await _sync_defillama_dex_volume(c, cutoff)

    _record_fetch("defillama_extra")
    return total


async def sync_blockchain_info() -> int:
    """Fetch BTC on-chain data from blockchain.info (free, no key)."""
    if not _should_fetch("blockchain_info"):
        return count_records("btc_hashrate")

    total = 0
    charts = {
        "btc_hashrate": ("hash-rate", "365days"),
        "btc_tx_count": ("n-transactions", "365days"),
        "btc_difficulty": ("difficulty", "365days"),
        "btc_mempool": ("mempool-size", "30days"),
        "btc_avg_fee": ("transaction-fees-usd", "365days"),
        "btc_active_addresses": ("n-unique-addresses", "365days"),
    }

    async with httpx.AsyncClient(timeout=15, verify=False) as c:
        for source, (chart_name, timespan) in charts.items():
            try:
                r = await c.get(
                    f"https://api.blockchain.info/charts/{chart_name}",
                    params={"timespan": timespan, "format": "json", "sampled": "true"},
                )
                r.raise_for_status()
                values = r.json().get("values", [])
                records = []
                for item in values:
                    date = datetime.fromtimestamp(item["x"], UTC).strftime("%Y-%m-%d")
                    records.append((date, item["y"]))
                store_batch(source, records)
                total += len(records)
                logger.info("Synced %d points for %s", len(records), source)
                await asyncio.sleep(0.3)
            except Exception:
                logger.warning("Blockchain.info %s sync failed", chart_name, exc_info=True)

    _record_fetch("blockchain_info")
    return total


async def sync_coingecko_eth() -> int:
    """Fetch ETH market data from CoinGecko."""
    if not _should_fetch("coingecko_eth"):
        return count_records("coingecko_eth")

    try:
        async with httpx.AsyncClient(timeout=15, verify=False) as c:
            r = await c.get(
                "https://api.coingecko.com/api/v3/coins/ethereum/market_chart",
                params={"vs_currency": "usd", "days": 365, "interval": "daily"},
            )
            r.raise_for_status()
            data = r.json()

        by_date: dict[str, dict] = {}
        for ts, price in data.get("prices", []):
            date = datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d")
            by_date.setdefault(date, {})["price"] = price
        for ts, mcap in data.get("market_caps", []):
            date = datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d")
            by_date.setdefault(date, {})["market_cap"] = mcap
        for ts, vol in data.get("total_volumes", []):
            date = datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d")
            by_date.setdefault(date, {})["total_volume"] = vol

        records = sorted(by_date.items())
        store_batch("coingecko_eth", records)
        _record_fetch("coingecko_eth")
        logger.info("Synced %d days of CoinGecko ETH market data", len(records))
        return len(records)
    except Exception:
        logger.warning("CoinGecko ETH sync failed", exc_info=True)
        return 0


async def sync_sosovalue_eth_etf(api_key: str) -> int:
    """Fetch ETH ETF historical inflow data from SoSoValue."""
    if not api_key:
        return 0
    if not _should_fetch("sosovalue_eth_etf"):
        return count_records("sosovalue_eth_etf")

    from cryptotrader.data.providers.sosovalue import fetch_etf_history

    history = await fetch_etf_history(api_key, etf_type="us-eth-spot")
    if not history:
        return 0

    records = []
    for item in history:
        date = item.get("date", "")
        if not date:
            continue
        records.append(
            (
                date,
                {
                    "totalNetInflow": item.get("totalNetInflow", 0),
                    "totalValueTraded": item.get("totalValueTraded", 0),
                    "totalNetAssets": item.get("totalNetAssets", 0),
                    "cumNetInflow": item.get("cumNetInflow", 0),
                },
            )
        )

    store_batch("sosovalue_eth_etf", records, forward_fill=True)
    _record_fetch("sosovalue_eth_etf")
    logger.info("Synced %d days of SoSoValue ETH ETF history", len(records))
    return len(records)


async def sync_binance_funding_history(symbols: list[str] | None = None) -> int:
    """Fetch funding rate history for multiple symbols from Binance."""
    if not _should_fetch("binance_funding"):
        return count_records("binance_funding_BTC")

    if symbols is None:
        symbols = ["BTC", "ETH", "SOL", "BNB", "XRP"]

    total = 0
    start_ms = int((datetime.now(UTC) - timedelta(days=90)).timestamp() * 1000)

    async with httpx.AsyncClient(timeout=15) as c:
        for symbol in symbols:
            try:
                pair = f"{symbol}USDT"
                all_records: list[dict] = []
                cursor = start_ms

                while True:
                    r = await c.get(
                        "https://fapi.binance.com/fapi/v1/fundingRate",
                        params={"symbol": pair, "startTime": cursor, "limit": 1000},
                    )
                    r.raise_for_status()
                    batch = r.json()
                    if not batch:
                        break
                    all_records.extend(batch)
                    cursor = batch[-1]["fundingTime"] + 1
                    if len(batch) < 1000:
                        break
                    await asyncio.sleep(0.1)

                # Aggregate to daily average
                daily: dict[str, list[float]] = {}
                for rec in all_records:
                    date = datetime.fromtimestamp(rec["fundingTime"] / 1000, UTC).strftime("%Y-%m-%d")
                    daily.setdefault(date, []).append(float(rec["fundingRate"]))

                records = [
                    (date, {"avg_rate": sum(rates) / len(rates), "count": len(rates)}) for date, rates in daily.items()
                ]
                store_batch(f"binance_funding_{symbol}", records)
                total += len(records)
                logger.info("Synced %d days of funding rate for %s", len(records), symbol)
                await asyncio.sleep(0.2)
            except Exception:
                logger.warning("Binance funding %s sync failed", symbol, exc_info=True)

    _record_fetch("binance_funding")
    return total


async def sync_binance_eth_derivatives() -> int:
    """Fetch ETH derivatives data from Binance."""
    if not _should_fetch("binance_eth_derivatives"):
        return count_records("binance_oi_ETH")

    pair = "ETHUSDT"
    base = "https://fapi.binance.com/futures/data"
    total = 0

    async with httpx.AsyncClient(timeout=15) as c:
        for endpoint, source_name in [
            ("openInterestHist", "binance_oi_ETH"),
            ("globalLongShortAccountRatio", "binance_ls_ratio_ETH"),
            ("topLongShortPositionRatio", "binance_top_trader_ETH"),
            ("takerlongshortRatio", "binance_taker_ETH"),
        ]:
            try:
                r = await c.get(f"{base}/{endpoint}", params={"symbol": pair, "period": "1d", "limit": 500})
                r.raise_for_status()
                records = []
                for item in r.json():
                    date = datetime.fromtimestamp(item["timestamp"] / 1000, UTC).strftime("%Y-%m-%d")
                    records.append((date, {k: v for k, v in item.items() if k != "timestamp"}))
                store_batch(source_name, records)
                total += len(records)
                logger.info("Synced %d records for %s", len(records), source_name)
                await asyncio.sleep(0.3)
            except Exception:
                logger.warning("Binance ETH %s sync failed", endpoint, exc_info=True)

    _record_fetch("binance_eth_derivatives")
    return total


async def sync_stablecoin_total_supply() -> int:
    """Fetch total stablecoin market supply history from DefiLlama (all stablecoins combined)."""
    if not _should_fetch("stablecoin_total"):
        return count_records("stablecoin_total_supply")

    try:
        async with httpx.AsyncClient(timeout=15, verify=False) as c:
            r = await c.get("https://stablecoins.llama.fi/stablecoincharts/all")
            r.raise_for_status()
            data = r.json()

        cutoff = (datetime.now(UTC) - timedelta(days=365)).timestamp()
        records = []
        for item in data:
            ts = int(item.get("date", 0))
            if ts >= cutoff:
                date = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                total_circ = item.get("totalCirculatingUSD", {}).get("peggedUSD", 0)
                records.append((date, {"total_supply": total_circ}))

        store_batch("stablecoin_total_supply", records)
        _record_fetch("stablecoin_total")
        logger.info("Synced %d days of total stablecoin supply", len(records))
        return len(records)
    except Exception:
        logger.warning("Total stablecoin supply sync failed", exc_info=True)
        return 0


async def sync_blockchain_extra() -> int:
    """Fetch additional BTC on-chain metrics: miner revenue, tx volume USD, block size, fees."""
    if not _should_fetch("blockchain_extra"):
        return count_records("btc_miners_revenue")

    total = 0
    charts = {
        "btc_miners_revenue": ("miners-revenue", "365days"),
        "btc_tx_volume_usd": ("estimated-transaction-volume-usd", "365days"),
        "btc_block_size": ("avg-block-size", "365days"),
        "btc_fees_usd": ("transaction-fees-usd", "365days"),
        "btc_market_price": ("market-price", "365days"),
    }

    async with httpx.AsyncClient(timeout=15, verify=False) as c:
        for source, (chart_name, timespan) in charts.items():
            try:
                r = await c.get(
                    f"https://api.blockchain.info/charts/{chart_name}",
                    params={"timespan": timespan, "format": "json", "sampled": "true"},
                )
                r.raise_for_status()
                values = r.json().get("values", [])
                records = [(datetime.fromtimestamp(item["x"], UTC).strftime("%Y-%m-%d"), item["y"]) for item in values]
                store_batch(source, records)
                total += len(records)
                logger.info("Synced %d points for %s", len(records), source)
                await asyncio.sleep(0.3)
            except Exception:
                logger.warning("Blockchain.info %s sync failed", chart_name, exc_info=True)

    _record_fetch("blockchain_extra")
    return total


async def sync_blockchain_extended() -> int:
    """Fetch extended on-chain data: UTXO count, total BTC supply, market cap, output volume, cost/tx."""
    if not _should_fetch("blockchain_extended"):
        return count_records("btc_utxo_count")

    total = 0
    charts = {
        "btc_utxo_count": ("utxo-count", "365days"),
        "btc_total_supply": ("total-bitcoins", "365days"),
        "btc_total_market_cap": ("market-cap", "365days"),
        "btc_output_volume": ("output-volume", "365days"),
        "btc_cost_per_tx": ("cost-per-transaction", "365days"),
        "btc_trade_volume": ("trade-volume", "365days"),
    }

    async with httpx.AsyncClient(timeout=15, verify=False) as c:
        for source, (chart_name, timespan) in charts.items():
            try:
                r = await c.get(
                    f"https://api.blockchain.info/charts/{chart_name}",
                    params={"timespan": timespan, "format": "json", "sampled": "true"},
                )
                r.raise_for_status()
                values = r.json().get("values", [])
                records = [(datetime.fromtimestamp(item["x"], UTC).strftime("%Y-%m-%d"), item["y"]) for item in values]
                store_batch(source, records)
                total += len(records)
                logger.info("Synced %d points for %s", len(records), source)
                await asyncio.sleep(0.3)
            except Exception:
                logger.warning("Blockchain.info %s sync failed", chart_name, exc_info=True)

    _record_fetch("blockchain_extended")
    return total


async def sync_mempool_space() -> int:
    """Fetch BTC mining data from mempool.space: hashrate, difficulty adjustments, fee rates."""
    if not _should_fetch("mempool_space"):
        return count_records("mempool_hashrate")

    total = 0

    async with httpx.AsyncClient(timeout=15, verify=False) as c:
        # Hashrate history (365 daily points)
        try:
            r = await c.get("https://mempool.space/api/v1/mining/hashrate/1y")
            r.raise_for_status()
            data = r.json()
            hashrates = data.get("hashrates", [])
            records = [
                (
                    datetime.fromtimestamp(h["timestamp"], UTC).strftime("%Y-%m-%d"),
                    {"hashrate_eh": h["avgHashrate"] / 1e18},
                )
                for h in hashrates
            ]
            store_batch("mempool_hashrate", records)
            total += len(records)
            logger.info("Synced %d days of mempool.space hashrate", len(records))

            # Difficulty adjustments
            diffs = data.get("difficulty", [])
            diff_records = [
                (
                    datetime.fromtimestamp(d["time"], UTC).strftime("%Y-%m-%d"),
                    {"difficulty": d["difficulty"], "adjustment": d.get("adjustment", 0)},
                )
                for d in diffs
            ]
            store_batch("mempool_difficulty_adj", diff_records)
            total += len(diff_records)
            logger.info("Synced %d difficulty adjustments", len(diff_records))
        except Exception:
            logger.warning("Mempool.space hashrate sync failed", exc_info=True)

        await asyncio.sleep(0.3)

        # Block fee rates history
        try:
            r = await c.get("https://mempool.space/api/v1/mining/blocks/fee-rates/1y")
            r.raise_for_status()
            data = r.json()
            records = [
                (
                    datetime.fromtimestamp(item["timestamp"], UTC).strftime("%Y-%m-%d %H:%M"),
                    {"avg_fee": item.get("avgFee_50", 0), "avg_fee_90": item.get("avgFee_90", 0)},
                )
                for item in data
                if "timestamp" in item
            ]
            store_batch("mempool_fee_rates", records)
            total += len(records)
            logger.info("Synced %d fee rate entries", len(records))
        except Exception:
            logger.warning("Mempool.space fee rates sync failed", exc_info=True)

    _record_fetch("mempool_space")
    return total


async def sync_defillama_chains_tvl() -> int:
    """Fetch TVL history for major chains: Solana, BSC, Bitcoin, Arbitrum, Base, Tron."""
    if not _should_fetch("defillama_chains"):
        return count_records("defillama_tvl_Solana")

    total = 0
    chains = ["Solana", "BSC", "Bitcoin", "Arbitrum", "Base", "Tron"]
    cutoff = (datetime.now(UTC) - timedelta(days=365)).timestamp()

    async with httpx.AsyncClient(timeout=15, verify=False) as c:
        for chain in chains:
            try:
                r = await c.get(f"https://api.llama.fi/v2/historicalChainTvl/{chain}")
                r.raise_for_status()
                records = []
                for item in r.json():
                    ts = item.get("date", 0)
                    if ts >= cutoff:
                        date = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                        records.append((date, {"tvl": item.get("tvl", 0)}))
                store_batch(f"defillama_tvl_{chain}", records)
                total += len(records)
                logger.info("Synced %d days of %s TVL", len(records), chain)
                await asyncio.sleep(0.2)
            except Exception:
                logger.warning("DefiLlama %s TVL sync failed", chain, exc_info=True)

    _record_fetch("defillama_chains")
    return total


async def sync_defillama_derivatives_volume() -> int:
    """Fetch perpetual derivatives and options volume history from DefiLlama."""
    if not _should_fetch("defillama_perps"):
        return count_records("defillama_perps_vol")

    total = 0
    cutoff = (datetime.now(UTC) - timedelta(days=365)).timestamp()

    async with httpx.AsyncClient(timeout=15, verify=False) as c:
        # Perps volume
        try:
            r = await c.get(
                "https://api.llama.fi/overview/derivatives",
                params={"excludeTotalDataChart": "false", "excludeTotalDataChartBreakdown": "true"},
            )
            r.raise_for_status()
            chart = r.json().get("totalDataChart", [])
            records = []
            for ts, vol in chart:
                if ts >= cutoff:
                    date = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                    records.append((date, {"perps_volume": vol}))
            store_batch("defillama_perps_vol", records)
            total += len(records)
            logger.info("Synced %d days of perps volume", len(records))
        except Exception:
            logger.warning("DefiLlama perps volume sync failed", exc_info=True)

        await asyncio.sleep(0.3)

        # Options volume
        try:
            r = await c.get(
                "https://api.llama.fi/overview/options",
                params={"excludeTotalDataChart": "false", "excludeTotalDataChartBreakdown": "true"},
            )
            r.raise_for_status()
            chart = r.json().get("totalDataChart", [])
            records = []
            for ts, vol in chart:
                if ts >= cutoff:
                    date = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                    records.append((date, {"options_volume": vol}))
            store_batch("defillama_options_vol", records)
            total += len(records)
            logger.info("Synced %d days of options volume", len(records))
        except Exception:
            logger.warning("DefiLlama options volume sync failed", exc_info=True)

    _record_fetch("defillama_perps")
    return total


async def sync_binance_funding_full(symbols: list[str] | None = None) -> int:
    """Fetch full funding rate history (up to 2 years) via pagination."""
    if not _should_fetch("binance_funding_full"):
        return count_records("binance_funding_full_BTC")

    if symbols is None:
        symbols = ["BTC", "ETH"]

    total = 0
    start_ms = int((datetime.now(UTC) - timedelta(days=730)).timestamp() * 1000)

    async with httpx.AsyncClient(timeout=15) as c:
        for symbol in symbols:
            try:
                pair = f"{symbol}USDT"
                all_records: list[dict] = []
                cursor = start_ms

                while True:
                    r = await c.get(
                        "https://fapi.binance.com/fapi/v1/fundingRate",
                        params={"symbol": pair, "startTime": cursor, "limit": 1000},
                    )
                    r.raise_for_status()
                    batch = r.json()
                    if not batch:
                        break
                    all_records.extend(batch)
                    cursor = batch[-1]["fundingTime"] + 1
                    if len(batch) < 1000:
                        break
                    await asyncio.sleep(0.1)

                # Aggregate to daily average
                daily: dict[str, list[float]] = {}
                for rec in all_records:
                    date = datetime.fromtimestamp(rec["fundingTime"] / 1000, UTC).strftime("%Y-%m-%d")
                    daily.setdefault(date, []).append(float(rec["fundingRate"]))

                records = [
                    (date, {"avg_rate": sum(rates) / len(rates), "count": len(rates)}) for date, rates in daily.items()
                ]
                store_batch(f"binance_funding_full_{symbol}", records)
                total += len(records)
                logger.info("Synced %d days of full funding rate for %s (2yr)", len(records), symbol)
                await asyncio.sleep(0.2)
            except Exception:
                logger.warning("Binance full funding %s sync failed", symbol, exc_info=True)

    _record_fetch("binance_funding_full")
    return total


async def sync_coinpaprika_global() -> int:
    """Fetch current global crypto market data from Coinpaprika (free, no key)."""
    if not _should_fetch("coinpaprika"):
        return count_records("coinpaprika_global")

    try:
        async with httpx.AsyncClient(timeout=15, verify=False) as c:
            r = await c.get("https://api.coinpaprika.com/v1/global")
            r.raise_for_status()
            data = r.json()

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        store_data(
            "coinpaprika_global",
            today,
            {
                "market_cap_usd": data.get("market_cap_usd", 0),
                "volume_24h_usd": data.get("volume_24h_usd", 0),
                "bitcoin_dominance": data.get("bitcoin_dominance_percentage", 0),
                "cryptocurrencies_number": data.get("cryptocurrencies_number", 0),
                "market_cap_change_24h": data.get("market_cap_change_24h", 0),
                "volume_24h_change_24h": data.get("volume_24h_change_24h", 0),
            },
        )
        _record_fetch("coinpaprika")
        logger.info("Synced Coinpaprika global market data")
        return 1
    except Exception:
        logger.warning("Coinpaprika global sync failed", exc_info=True)
        return 0


async def sync_all(providers_config=None) -> dict[str, int]:
    """Run all data syncs in parallel (respecting rate limits). Returns {source: record_count}."""
    cfg = providers_config

    soso_key = getattr(cfg, "sosovalue_api_key", "") if cfg else ""
    fred_key = getattr(cfg, "fred_api_key", "") if cfg else ""

    tasks = {
        # SoSoValue
        "sosovalue_btc_etf": sync_sosovalue_etf_history(soso_key),
        "sosovalue_eth_etf": sync_sosovalue_eth_etf(soso_key),
        "sosovalue_etf_current": sync_sosovalue_etf_current(soso_key),
        "sosovalue_news": sync_sosovalue_news(soso_key),
        # Market sentiment
        "fear_greed": sync_fear_greed_history(),
        # Binance derivatives (BTC + ETH)
        "binance_btc_deriv": sync_binance_derivatives("BTC"),
        "binance_eth_deriv": sync_binance_eth_derivatives(),
        "binance_funding": sync_binance_funding_history(),
        # DeFi
        "defillama_eth_tvl": sync_defillama_tvl(),
        "defillama_extra": sync_defillama_extra(),
        # Market data
        "coingecko_btc": sync_coingecko_market(),
        "coingecko_eth": sync_coingecko_eth(),
        # On-chain
        "blockchain_info": sync_blockchain_info(),
        "blockchain_extra": sync_blockchain_extra(),
        "blockchain_extended": sync_blockchain_extended(),
        "mempool_space": sync_mempool_space(),
        # Stablecoin total supply
        "stablecoin_total": sync_stablecoin_total_supply(),
        # Multi-chain TVL
        "defillama_chains": sync_defillama_chains_tvl(),
        # Derivatives volume
        "defillama_derivatives": sync_defillama_derivatives_volume(),
        # Extended funding rate (2yr)
        "binance_funding_full": sync_binance_funding_full(),
        # Global market
        "coinpaprika_global": sync_coinpaprika_global(),
    }

    # FRED macro series
    if fred_key:
        tasks["fred_DFF"] = sync_fred_history(fred_key, "DFF")
        tasks["fred_DTWEXBGS"] = sync_fred_history(fred_key, "DTWEXBGS")
        tasks["fred_multi"] = sync_fred_multi(fred_key)

    results = {}
    # Run in batches of 3 to avoid overwhelming APIs
    task_items = list(tasks.items())
    for i in range(0, len(task_items), 3):
        batch = task_items[i : i + 3]
        batch_results = await asyncio.gather(
            *[t for _, t in batch],
            return_exceptions=True,
        )
        for (name, _), result in zip(batch, batch_results, strict=False):
            if isinstance(result, Exception):
                logger.warning("Sync %s failed: %s", name, result)
                results[name] = 0
            else:
                results[name] = result if isinstance(result, int) else 1

    total = sum(v for v in results.values() if isinstance(v, int))
    logger.info("Data sync complete: %d total records across %d sources", total, len(results))
    return results
