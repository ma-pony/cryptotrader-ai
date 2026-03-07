"""OKX DEX Market Data API."""

import base64
import hashlib
import hmac
from datetime import UTC, datetime

import httpx


class OKXMarket:
    """OKX DEX market data client."""

    BASE_URL = "https://web3.okx.com"

    def __init__(self, api_key: str, secret_key: str, passphrase: str):
        """Initialize OKX client.

        Args:
            api_key: OKX API key
            secret_key: OKX secret key
            passphrase: OKX passphrase
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase

        if not all([self.api_key, self.secret_key, self.passphrase]):
            raise ValueError(
                "OKX credentials not configured. Set PROVIDER_OKX_API_KEY, PROVIDER_OKX_SECRET_KEY, "
                "and PROVIDER_OKX_PASSPHRASE environment variables or pass them to constructor."
            )

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate HMAC-SHA256 signature."""
        message = timestamp + method + path + body
        return base64.b64encode(hmac.new(self.secret_key.encode(), message.encode(), hashlib.sha256).digest()).decode()

    async def get_price(self, chain_id: str, contract_address: str) -> float | None:
        """Get token price in USD."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        path = "/api/v6/dex/market/price"
        body = f'[{{"chainIndex":"{chain_id}","tokenContractAddress":"{contract_address}"}}]'

        sign = self._sign(timestamp, "POST", path, body)

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    f"{self.BASE_URL}{path}",
                    content=body,
                    headers={
                        "OK-ACCESS-KEY": self.api_key,
                        "OK-ACCESS-SIGN": sign,
                        "OK-ACCESS-PASSPHRASE": self.passphrase,
                        "OK-ACCESS-TIMESTAMP": timestamp,
                        "Content-Type": "application/json",
                    },
                )
                data = resp.json()
                if data.get("code") == "0" and data.get("data"):
                    return float(data["data"][0].get("price", 0))
            except Exception:
                pass
        return None

    async def get_candles(self, chain_id: str, contract_address: str, bar: str = "1H", limit: int = 24) -> list[dict]:
        """Get K-line candles."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        params = f"chainIndex={chain_id}&tokenContractAddress={contract_address}&bar={bar}&limit={limit}"
        path = f"/api/v6/dex/market/candles?{params}"

        sign = self._sign(timestamp, "GET", path)

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    f"{self.BASE_URL}{path}",
                    headers={
                        "OK-ACCESS-KEY": self.api_key,
                        "OK-ACCESS-SIGN": sign,
                        "OK-ACCESS-PASSPHRASE": self.passphrase,
                        "OK-ACCESS-TIMESTAMP": timestamp,
                    },
                )
                data = resp.json()
                if data.get("code") == "0":
                    return data.get("data", [])
            except Exception:
                pass
        return []
