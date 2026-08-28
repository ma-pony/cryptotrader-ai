"""Installed market evidence sources."""

from cryptotrader.market_sources.default import DefaultMarketDataSource
from cryptotrader.market_sources.protocol import MarketDataSource
from cryptotrader.market_sources.registry import MarketSourceRegistry

__all__ = ["DefaultMarketDataSource", "MarketDataSource", "MarketSourceRegistry"]
