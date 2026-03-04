"""Binance Token Security Audit API."""
import httpx
from typing import Optional
from uuid import uuid4


class BinanceAudit:
    """Binance token security audit client."""

    BASE_URL = "https://web3.binance.com/bapi/defi/v1/public/wallet-direct"

    CHAIN_MAP = {
        "BTC": "1",
        "ETH": "1",
        "BSC": "56",
        "BASE": "8453",
        "SOL": "CT_501"
    }

    async def audit_token(self, symbol: str, contract_address: str, chain: str = "BSC") -> dict:
        """Audit token security.

        Returns:
            {
                "risk_level": "LOW" | "MEDIUM" | "HIGH",
                "is_honeypot": bool,
                "is_scam": bool,
                "buy_tax": float,
                "sell_tax": float,
                "issues": [str]
            }
        """
        chain_id = self.CHAIN_MAP.get(chain, "56")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    f"{self.BASE_URL}/security/token/audit",
                    json={
                        "binanceChainId": chain_id,
                        "contractAddress": contract_address,
                        "requestId": str(uuid4())
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Accept-Encoding": "identity"
                    }
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("code") != "000000":
                    return {"risk_level": "UNKNOWN", "issues": ["API error"]}

                result = data.get("data", {})
                return self._parse_audit_result(result)

            except Exception as e:
                return {"risk_level": "UNKNOWN", "issues": [f"Request failed: {e}"]}

    def _parse_audit_result(self, result: dict) -> dict:
        """Parse audit result."""
        issues = []
        risk_level = "LOW"

        # Check honeypot
        is_honeypot = result.get("isHoneypot", False)
        if is_honeypot:
            issues.append("Honeypot detected")
            risk_level = "HIGH"

        # Check scam
        is_scam = result.get("isScam", False)
        if is_scam:
            issues.append("Scam token")
            risk_level = "HIGH"

        # Check taxes
        buy_tax = float(result.get("buyTax") or 0)
        sell_tax = float(result.get("sellTax") or 0)

        if buy_tax > 10 or sell_tax > 10:
            issues.append(f"High tax: buy={buy_tax}%, sell={sell_tax}%")
            risk_level = "HIGH"

        # Check ownership
        if result.get("hasOwnership", False):
            issues.append("Owner can modify contract")
            if risk_level == "LOW":
                risk_level = "MEDIUM"

        return {
            "risk_level": risk_level,
            "is_honeypot": is_honeypot,
            "is_scam": is_scam,
            "buy_tax": buy_tax,
            "sell_tax": sell_tax,
            "issues": issues
        }
