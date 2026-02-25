"""On-chain data collector — Binance free API + optional paid providers."""

from __future__ import annotations

import asyncio as _aio
import logging
import os

from cryptotrader.models import OnchainData

logger = logging.getLogger(__name__)


class OnchainCollector:

    def __init__(self, providers_config=None):
        self._cfg = providers_config

    async def collect(self, pair: str, funding_rate: float = 0.0) -> OnchainData:
        from cryptotrader.data.providers.binance import fetch_derivatives_binance
        from cryptotrader.data.providers.defillama import fetch_tvl
        from cryptotrader.data.providers.coinglass import fetch_derivatives as fetch_cg
        from cryptotrader.data.providers.cryptoquant import fetch_exchange_netflow
        from cryptotrader.data.providers.whale_alert import fetch_whale_transfers

        symbol = pair.split("/")[0]
        cfg = self._cfg

        # Check enabled flags (default True if no config)
        defillama_on = getattr(cfg, "defillama_enabled", True) if cfg else True
        coinglass_on = getattr(cfg, "coinglass_enabled", True) if cfg else True
        cryptoquant_on = getattr(cfg, "cryptoquant_enabled", True) if cfg else True
        whale_alert_on = getattr(cfg, "whale_alert_enabled", True) if cfg else True

        # Prefer config keys, fall back to env vars
        cg_key = (cfg.coinglass_api_key if cfg else None) or os.environ.get("COINGLASS_API_KEY")
        cq_key = (cfg.cryptoquant_api_key if cfg else None) or os.environ.get("CRYPTOQUANT_API_KEY")
        wa_key = (cfg.whale_alert_api_key if cfg else None) or os.environ.get("WHALE_ALERT_API_KEY")

        # Build tasks — only call enabled providers

        async def _noop_dict():
            return {}

        async def _noop_float():
            return 0.0

        async def _noop_list():
            return []

        tvl_task = fetch_tvl() if defillama_on else _noop_dict()
        deriv_task = fetch_derivatives_binance(symbol)  # Binance is always free
        cg_task = fetch_cg(cg_key, symbol) if coinglass_on else _noop_dict()
        netflow_task = fetch_exchange_netflow(cq_key) if cryptoquant_on else _noop_float()
        whales_task = fetch_whale_transfers(wa_key) if whale_alert_on else _noop_list()

        tvl_data, deriv, cg_data, netflow, whales = await _aio.gather(
            tvl_task, deriv_task, cg_task, netflow_task, whales_task,
            return_exceptions=True,
        )

        quality = {
            "defillama": defillama_on and not isinstance(tvl_data, Exception),
            "binance": not isinstance(deriv, Exception),
            "coinglass": coinglass_on and not isinstance(cg_data, Exception),
            "cryptoquant": cryptoquant_on and not isinstance(netflow, Exception),
            "whale_alert": whale_alert_on and not isinstance(whales, Exception),
        }

        if isinstance(tvl_data, Exception):
            logger.warning("TVL fetch error: %s", tvl_data)
            tvl_data = {"defi_tvl": 0.0, "defi_tvl_change_7d": 0.0}
        if isinstance(deriv, Exception):
            logger.warning("Binance derivatives error: %s", deriv)
            deriv = {}
        if isinstance(cg_data, Exception):
            logger.warning("CoinGlass fetch error: %s", cg_data)
            cg_data = {}
        if isinstance(netflow, Exception):
            logger.warning("CryptoQuant fetch error: %s", netflow)
            netflow = 0.0
        if isinstance(whales, Exception):
            logger.warning("Whale Alert fetch error: %s", whales)
            whales = []

        # Prefer CoinGlass OI if available, fallback to Binance
        oi = cg_data.get("open_interest", 0.0) or deriv.get("open_interest", 0.0)

        # Merge liquidation data
        liq = cg_data.get("liquidations_24h", {})
        liq.update({
            "long_short_ratio": deriv.get("long_short_ratio", 1.0),
            "top_trader_ratio": deriv.get("top_trader_ratio", 1.0),
            "taker_buy_sell_ratio": deriv.get("taker_buy_sell_ratio", 1.0),
        })

        return OnchainData(
            exchange_netflow=netflow if isinstance(netflow, float) else 0.0,
            whale_transfers=whales if isinstance(whales, list) else [],
            open_interest=oi,
            liquidations_24h=liq,
            defi_tvl=tvl_data.get("defi_tvl", 0.0),
            defi_tvl_change_7d=tvl_data.get("defi_tvl_change_7d", 0.0),
            data_quality=quality,
        )
