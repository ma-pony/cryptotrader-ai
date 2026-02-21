"""On-chain data collector — Binance free API + graceful degradation."""

from __future__ import annotations

import asyncio
import logging

from cryptotrader.models import OnchainData

logger = logging.getLogger(__name__)


class OnchainCollector:

    async def collect(self, pair: str, funding_rate: float = 0.0) -> OnchainData:
        from cryptotrader.data.providers.binance import fetch_derivatives_binance
        from cryptotrader.data.providers.defillama import fetch_tvl

        symbol = pair.split("/")[0]

        tvl_data, deriv = await asyncio.gather(
            fetch_tvl(),
            fetch_derivatives_binance(symbol),
            return_exceptions=True,
        )

        if isinstance(tvl_data, Exception):
            logger.warning("TVL fetch error: %s", tvl_data)
            tvl_data = {"defi_tvl": 0.0, "defi_tvl_change_7d": 0.0}
        if isinstance(deriv, Exception):
            logger.warning("Binance derivatives error: %s", deriv)
            deriv = {}

        return OnchainData(
            exchange_netflow=0.0,  # No free source for this
            whale_transfers=[],
            open_interest=deriv.get("open_interest", 0.0),
            liquidations_24h={
                "long_short_ratio": deriv.get("long_short_ratio", 1.0),
                "top_trader_ratio": deriv.get("top_trader_ratio", 1.0),
                "taker_buy_sell_ratio": deriv.get("taker_buy_sell_ratio", 1.0),
            },
            defi_tvl=tvl_data.get("defi_tvl", 0.0),
            defi_tvl_change_7d=tvl_data.get("defi_tvl_change_7d", 0.0),
        )
