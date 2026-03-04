"""OKX DEX Market Data API."""
import os
import hmac
import hashlib
import base64
from datetime import datetime, timezone
from typing import Optional, List
import httpx


class OKXMarket:
    """OKX DEX market data client."""

    BASE_URL = "https://web3.okx.com"

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None,
                 passphrase: Optional[str] = None):
        """Initialize OKX client.

        Args:
            api_key: OKX API key (defaults to OKX_API_KEY env var)
            secret_key: OKX secret key (defaults to OKX_SECRET_KEY env var)
            passphrase: OKX passphrase (defaults to OKX_PASSPHRASE env var)
        """
        self.api_key = api_key or os.getenv("OKX_API_KEY")
        self.secret_key = secret_key or os.getenv("OKX_SECRET_KEY")
        self.passphrase = passphrase or os.getenv("OKX_PASSPHRASE")

        if not all([self.api_key, self.secret_key, self.passphrase]):
            raise ValueError(
                "OKX credentials not configured. Set OKX_API_KEY, OKX_SECRET_KEY, "
                "and OKX_PASSPHRASE environment variables or pass them to constructor."
            )

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate HMAC-SHA256 signature."""
        message = timestamp + method + path + body
        return base64.b64encode(
            hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        ).decode()

    async def get_price(self, chain_id: str, contract_address: str) -> Optional[float]:
        """Get token price in USD."""
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
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
                        "Content-Type": "application/json"
                    }
                )
                data = resp.json()
                if data.get("code") == "0" and data.get("data"):
                    return float(data["data"][0].get("price", 0))
            except Exception:
                pass
        return None

    async def get_candles(self, chain_id: str, contract_address: str,
                         bar: str = "1H", limit: int = 24) -> List[dict]:
        """Get K-line candles."""
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
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
                        "OK-ACCESS-TIMESTAMP": timestamp
                    }
                )
                data = resp.json()
                if data.get("code") == "0":
                    return data.get("data", [])
            except Exception:
                pass
        return []
