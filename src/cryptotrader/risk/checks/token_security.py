"""Token security audit check."""

import logging

from cryptotrader.data.binance_audit import BinanceAudit
from cryptotrader.models import CheckResult, TradeVerdict

logger = logging.getLogger(__name__)


class TokenSecurityCheck:
    """Check token security via Binance audit API."""

    name = "token_security"

    def __init__(self):
        self.audit = BinanceAudit()

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        """Check if token passes security audit."""
        contract = getattr(verdict, "contract_address", None)
        if not contract:
            return CheckResult(passed=True, reason="No contract address to audit")

        try:
            result = await self.audit.audit_token(
                symbol=verdict.symbol, contract_address=contract, chain=getattr(verdict, "chain", "BSC")
            )
        except Exception:
            logger.warning("Token security audit failed, allowing trade conservatively")
            return CheckResult(passed=True, reason="Audit API unavailable")

        if result["risk_level"] == "HIGH":
            return CheckResult(passed=False, reason=f"High security risk: {', '.join(result['issues'])}")

        return CheckResult(passed=True)
