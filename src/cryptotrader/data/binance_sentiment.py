"""Binance Market Sentiment & Smart Money API."""

import httpx


class BinanceSentiment:
    """Binance sentiment and smart money data client."""

    BASE_URL = "https://web3.binance.com/bapi/defi/v1/public/wallet-direct"

    async def get_social_hype(self, chain_id: str = "56", limit: int = 10) -> list[dict]:
        """Get social hype leaderboard."""
        params = {
            "chainId": chain_id,
            "sentiment": "All",
            "socialLanguage": "ALL",
            "targetLanguage": "en",
            "timeRange": 1,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    f"{self.BASE_URL}/buw/wallet/market/token/pulse/social/hype/rank/leaderboard",
                    params=params,
                    headers={"Accept-Encoding": "identity"},
                )
                data = resp.json()
                if data.get("code") == "000000":
                    return data.get("data", {}).get("leaderBoardList", [])[:limit]
            except Exception:
                pass
        return []

    async def get_smart_money_signals(self, chain_id: str = "CT_501", page: int = 1) -> list[dict]:
        """Get smart money trading signals."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    f"{self.BASE_URL}/buw/wallet/web/signal/smart-money",
                    json={"smartSignalType": "", "page": page, "pageSize": 20, "chainId": chain_id},
                    headers={"Content-Type": "application/json", "Accept-Encoding": "identity"},
                )
                data = resp.json()
                if data.get("code") == "000000":
                    return data.get("data", [])
            except Exception:
                pass
        return []
