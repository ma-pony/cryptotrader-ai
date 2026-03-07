"""Enhanced data sources for agents."""

from cryptotrader.config import ProvidersConfig
from cryptotrader.data.binance_sentiment import BinanceSentiment


class EnhancedDataProvider:
    """Provides enhanced market data from OKX and Binance."""

    CHAIN_MAP = {"BTC": "1", "ETH": "1", "SOL": "CT_501", "BNB": "56"}

    def __init__(self, config: ProvidersConfig | None = None):
        """Initialize with optional config."""
        self.config = config or ProvidersConfig()
        self.binance = BinanceSentiment()
        self._okx = None

        # Only initialize OKX if enabled and credentials present
        if self.config.okx_enabled and self.config.has_okx_credentials():
            from cryptotrader.data.okx_market import OKXMarket

            self._okx = OKXMarket(
                api_key=self.config.okx_api_key,
                secret_key=self.config.okx_secret_key,
                passphrase=self.config.okx_passphrase,
            )

    async def get_price_data(self, symbol: str, contract: str | None = None) -> dict:
        """Get enhanced price data from OKX."""
        if not self._okx:
            return {"okx_price": None, "okx_candles_24h": 0, "okx_available": False}

        base = symbol.split("/")[0] if "/" in symbol else symbol.replace("USDT", "")
        chain_id = self.CHAIN_MAP.get(base, "1")

        if not contract:
            contract = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

        price = await self._okx.get_price(chain_id, contract)
        candles = await self._okx.get_candles(chain_id, contract, bar="1H", limit=24)

        return {"okx_price": price, "okx_candles_24h": len(candles), "okx_available": price is not None}

    async def get_sentiment_data(self, symbol: str) -> dict:
        """Get sentiment and smart money data from Binance."""
        base = symbol.split("/")[0] if "/" in symbol else symbol.replace("USDT", "")
        chain_id = self.CHAIN_MAP.get(base, "56")

        social = await self.binance.get_social_hype(chain_id, limit=5)
        signals = await self.binance.get_smart_money_signals(chain_id, page=1)

        return {
            "social_hype_tokens": len(social),
            "smart_money_signals": len(signals),
            "sentiment_available": len(social) > 0,
        }
