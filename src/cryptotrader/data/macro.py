"""Macro data collector — FRED, CoinGecko, Alternative.me."""

from __future__ import annotations

import logging

import httpx

from cryptotrader.models import MacroData

logger = logging.getLogger(__name__)


async def _fetch_fred(series: str, api_key: str) -> float:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"series_id": series, "api_key": api_key, "sort_order": "desc", "limit": 1, "file_type": "json"},
            )
            r.raise_for_status()
            obs = r.json().get("observations", [])
            if obs:
                return float(obs[0]["value"])
    except Exception:
        logger.warning("FRED fetch failed for %s", series, exc_info=True)
    return 0.0


async def _fetch_fear_greed() -> int:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.alternative.me/fng/?limit=1")
            r.raise_for_status()
            data = r.json().get("data", [])
            if data:
                return int(data[0]["value"])
    except Exception:
        logger.warning("Fear & Greed fetch failed", exc_info=True)
    return 50


async def _fetch_btc_dominance() -> float:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.coingecko.com/api/v3/global")
            r.raise_for_status()
            return float(r.json()["data"]["market_cap_percentage"]["btc"])
    except Exception:
        logger.warning("CoinGecko dominance fetch failed", exc_info=True)
    return 0.0


class MacroCollector:
    def __init__(self, providers_config=None):
        self._cfg = providers_config

    @staticmethod
    def _load_store_supplements() -> dict[str, float]:
        """Load supplementary data from unified SQLite store (best-effort)."""
        result = {"vix": 0.0, "sp500": 0.0, "stablecoin": 0.0, "hashrate": 0.0}
        try:
            from cryptotrader.data.store import get_latest

            for source, key in [
                ("fred_VIXCLS", "vix"),
                ("fred_SP500", "sp500"),
                ("stablecoin_total_supply", "stablecoin"),
                ("btc_hashrate", "hashrate"),
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
            pass
        return result

    async def collect(self) -> MacroData:
        cfg = self._cfg
        fred_on = getattr(cfg, "fred_enabled", True) if cfg else True
        coingecko_on = getattr(cfg, "coingecko_enabled", True) if cfg else True
        sosovalue_on = getattr(cfg, "sosovalue_enabled", True) if cfg else True

        fred_key = cfg.fred_api_key if cfg else ""
        soso_key = getattr(cfg, "sosovalue_api_key", "") if cfg else ""

        import asyncio

        async def _noop_float():
            return 0.0

        async def _noop_dict():
            return {}

        fed_task = _fetch_fred("DFF", fred_key) if (fred_on and fred_key) else _noop_float()
        dxy_task = _fetch_fred("DTWEXBGS", fred_key) if (fred_on and fred_key) else _noop_float()
        fg_task = _fetch_fear_greed()
        dom_task = _fetch_btc_dominance() if coingecko_on else _noop_float()

        if sosovalue_on and soso_key:
            from cryptotrader.data.providers.sosovalue import fetch_etf_metrics

            etf_task = fetch_etf_metrics(soso_key)
        else:
            etf_task = _noop_dict()

        fed_rate, dxy, fear_greed, btc_dom, etf_data = await asyncio.gather(
            fed_task,
            dxy_task,
            fg_task,
            dom_task,
            etf_task,
        )

        supplements = self._load_store_supplements()

        return MacroData(
            fed_rate=fed_rate,
            dxy=dxy,
            btc_dominance=btc_dom,
            fear_greed_index=fear_greed,
            etf_daily_net_inflow=etf_data.get("dailyNetInflow", 0.0),
            etf_total_net_assets=etf_data.get("totalNetAssets", 0.0),
            etf_cum_net_inflow=etf_data.get("cumNetInflow", 0.0),
            vix=supplements["vix"],
            sp500=supplements["sp500"],
            stablecoin_total_supply=supplements["stablecoin"],
            btc_hashrate=supplements["hashrate"],
        )
