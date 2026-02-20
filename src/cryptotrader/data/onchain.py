"""On-chain data collector — wires up all 4 providers with graceful degradation."""

from __future__ import annotations

import asyncio
import logging
import os

from cryptotrader.models import OnchainData

logger = logging.getLogger(__name__)


class OnchainCollector:

    async def collect(self, pair: str, funding_rate: float = 0.0) -> OnchainData:
        from cryptotrader.data.providers.defillama import fetch_tvl
        from cryptotrader.data.providers.coinglass import fetch_derivatives
        from cryptotrader.data.providers.cryptoquant import fetch_exchange_netflow
        from cryptotrader.data.providers.whale_alert import fetch_whale_transfers

        symbol = pair.split("/")[0]
        cg_key = os.environ.get("COINGLASS_API_KEY")
        cq_key = os.environ.get("CRYPTOQUANT_API_KEY")
        wa_key = os.environ.get("WHALE_ALERT_API_KEY")

        tvl_data, deriv, netflow, whales = await asyncio.gather(
            fetch_tvl(),
            fetch_derivatives(cg_key, symbol),
            fetch_exchange_netflow(cq_key),
            fetch_whale_transfers(wa_key),
            return_exceptions=True,
        )

        if isinstance(tvl_data, Exception):
            logger.warning("TVL fetch error: %s", tvl_data)
            tvl_data = {"defi_tvl": 0.0, "defi_tvl_change_7d": 0.0}
        if isinstance(deriv, Exception):
            logger.warning("Derivatives fetch error: %s", deriv)
            deriv = {"open_interest": 0.0, "liquidations_24h": {}}
        if isinstance(netflow, Exception):
            logger.warning("Netflow fetch error: %s", netflow)
            netflow = 0.0
        if isinstance(whales, Exception):
            logger.warning("Whale fetch error: %s", whales)
            whales = []

        return OnchainData(
            exchange_netflow=netflow if isinstance(netflow, float) else 0.0,
            whale_transfers=whales if isinstance(whales, list) else [],
            open_interest=deriv.get("open_interest", 0.0),
            liquidations_24h=deriv.get("liquidations_24h", {}),
            defi_tvl=tvl_data.get("defi_tvl", 0.0),
            defi_tvl_change_7d=tvl_data.get("defi_tvl_change_7d", 0.0),
        )
