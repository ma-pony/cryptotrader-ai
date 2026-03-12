"""Macro data collector — FRED, CoinGecko, Alternative.me.

Uses unified SQLite store for caching: check cache first, only call API if stale.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import httpx

from cryptotrader.data.store import cache_result, get_cached_or_none, get_latest
from cryptotrader.models import MacroData

logger = logging.getLogger(__name__)


async def _fetch_fred(series: str, api_key: str, date: str | None = None) -> float:
    source_key = f"fred_{series}"
    cached = get_cached_or_none(source_key, date)
    if cached is not None:
        return float(cached) if isinstance(cached, int | float) else 0.0

    # Backtest mode: no live API call for historical data
    if date is not None:
        return 0.0

    if not api_key:
        return 0.0
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"series_id": series, "api_key": api_key, "sort_order": "desc", "limit": 1, "file_type": "json"},
            )
            r.raise_for_status()
            obs = r.json().get("observations", [])
            if obs:
                val = float(obs[0]["value"])
                obs_date = obs[0].get("date")
                cache_result(source_key, val, date=obs_date)
                return val
    except Exception:
        logger.warning("FRED fetch failed for %s", series, exc_info=True)
    return 0.0


async def _fetch_fear_greed(date: str | None = None) -> tuple[int, list[int]]:
    """Fetch Fear & Greed index.  Returns (latest_value, last_7_values)."""
    cached = get_cached_or_none("fear_greed", date)
    if cached is not None:
        latest = int(cached) if isinstance(cached, int | float) else 50
        return latest, []

    # Backtest mode: no live API call for historical data
    if date is not None:
        return 50, []

    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.alternative.me/fng/?limit=7")
            r.raise_for_status()
            data = r.json().get("data", [])
            if data:
                val = int(data[0]["value"])
                # Use the timestamp from the API as the date
                ts = data[0].get("timestamp")
                fg_date = None
                if ts:
                    fg_date = datetime.fromtimestamp(int(ts), tz=UTC).strftime("%Y-%m-%d")
                cache_result("fear_greed", val, date=fg_date)
                history = [int(d["value"]) for d in data]
                return val, history
    except Exception:
        logger.warning("Fear & Greed fetch failed", exc_info=True)
    return 50, []


async def _fetch_btc_dominance(date: str | None = None) -> float:
    cached = get_cached_or_none("btc_dominance", date)
    if cached is not None:
        return float(cached) if isinstance(cached, int | float) else 0.0

    # Backtest mode: no live API call for historical data
    if date is not None:
        return 0.0

    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.coingecko.com/api/v3/global")
            r.raise_for_status()
            val = float(r.json()["data"]["market_cap_percentage"]["btc"])
            cache_result("btc_dominance", val)
            return val
    except Exception:
        logger.warning("CoinGecko dominance fetch failed", exc_info=True)
    return 0.0


class MacroCollector:
    def __init__(self, providers_config=None):
        self._cfg = providers_config

    @staticmethod
    def _load_store_supplements() -> dict[str, float]:
        """Load supplementary data from unified SQLite store (best-effort)."""
        result = {
            "vix": 0.0,
            "sp500": 0.0,
            "stablecoin": 0.0,
            "hashrate": 0.0,
            "yield_curve": 0.0,
            "m2_supply": 0.0,
            "cpi": 0.0,
        }
        try:
            for source, key in [
                ("fred_VIXCLS", "vix"),
                ("fred_SP500", "sp500"),
                ("stablecoin_total_supply", "stablecoin"),
                ("btc_hashrate", "hashrate"),
                ("fred_T10Y2Y", "yield_curve"),
                ("fred_WM2NS", "m2_supply"),
                ("fred_CPIAUCSL", "cpi"),
            ]:
                latest = get_latest(source, limit=1)
                if not latest:
                    continue
                val = latest[0][1]
                if isinstance(val, dict):
                    val = val.get("total_supply", 0) if "supply" in source else 0
                if isinstance(val, int | float):
                    result[key] = float(val)
        except Exception:
            logger.debug("Failed to load store supplements", exc_info=True)
        return result

    async def collect(self, date: str | None = None) -> MacroData:
        cfg = self._cfg
        fred_on = getattr(cfg, "fred_enabled", True) if cfg else True
        coingecko_on = getattr(cfg, "coingecko_enabled", True) if cfg else True
        sosovalue_on = getattr(cfg, "sosovalue_enabled", True) if cfg else True

        fred_key = cfg.fred_api_key if cfg else ""
        soso_key = getattr(cfg, "sosovalue_api_key", "") if cfg else ""

        async def _noop_float():
            return 0.0

        async def _noop_dict():
            return {}

        async def _noop_fear_greed():
            return 50, []

        fed_task = _fetch_fred("DFF", fred_key, date) if (fred_on and fred_key) else _noop_float()
        dxy_task = _fetch_fred("DTWEXBGS", fred_key, date) if (fred_on and fred_key) else _noop_float()
        fg_task = _fetch_fear_greed(date) if True else _noop_fear_greed()
        dom_task = _fetch_btc_dominance(date) if coingecko_on else _noop_float()

        if sosovalue_on and soso_key:
            from cryptotrader.data.providers.sosovalue import fetch_etf_metrics

            etf_task = fetch_etf_metrics(soso_key)
        else:
            etf_task = _noop_dict()

        fed_rate, dxy, fg_result, btc_dom, etf_data = await asyncio.gather(
            fed_task,
            dxy_task,
            fg_task,
            dom_task,
            etf_task,
        )

        # Unpack Fear & Greed result (latest value + 7-day history)
        if isinstance(fg_result, tuple):
            fear_greed, fear_greed_history = fg_result
        else:
            fear_greed, fear_greed_history = int(fg_result), []

        supplements = self._load_store_supplements()

        return MacroData(
            fed_rate=fed_rate,
            dxy=dxy,
            btc_dominance=btc_dom,
            fear_greed_index=fear_greed,
            fear_greed_history=fear_greed_history,
            etf_daily_net_inflow=etf_data.get("dailyNetInflow", 0.0),
            etf_total_net_assets=etf_data.get("totalNetAssets", 0.0),
            etf_cum_net_inflow=etf_data.get("cumNetInflow", 0.0),
            etf_top_flows=etf_data.get("topEtfFlows", []),
            vix=supplements["vix"],
            sp500=supplements["sp500"],
            stablecoin_total_supply=supplements["stablecoin"],
            btc_hashrate=supplements["hashrate"],
            yield_curve=supplements["yield_curve"],
            m2_supply=supplements["m2_supply"],
            cpi=supplements["cpi"],
        )
