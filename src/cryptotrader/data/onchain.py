"""On-chain data collector placeholder."""

from __future__ import annotations

from cryptotrader.models import OnchainData


class OnchainCollector:

    async def collect(self, pair: str, funding_rate: float = 0.0) -> OnchainData:
        """Phase 1: returns OnchainData with funding_rate passed from market collector."""
        return OnchainData(exchange_netflow=funding_rate)
