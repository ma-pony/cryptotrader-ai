"""Token security audit check."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cryptotrader.data.binance_audit import BinanceAudit
from cryptotrader.risk.models import RiskCheckResult

if TYPE_CHECKING:
    from cryptotrader.risk.models import RiskRequest

logger = logging.getLogger(__name__)


class TokenSecurityCheck:
    """Check token security via Binance audit API."""

    name = "token_security"

    def __init__(self, *, tax_threshold: float):
        self.audit = BinanceAudit(tax_threshold=tax_threshold)

    async def evaluate(self, request: RiskRequest, portfolio: dict) -> RiskCheckResult:
        """Check if token passes security audit."""
        contract = portfolio.get("contract_address")
        if not contract:
            return RiskCheckResult(passed=True, reason="No contract address to audit")

        try:
            result = await self.audit.audit_token(
                symbol=portfolio.get("symbol", request.context.pair.base),
                contract_address=contract,
                chain=portfolio.get("chain", "BSC"),
            )
        except Exception:
            logger.warning("Token security audit failed, allowing trade conservatively")
            return RiskCheckResult(passed=True, reason="Audit API unavailable")

        if result["risk_level"] == "HIGH":
            return RiskCheckResult(passed=False, reason=f"High security risk: {', '.join(result['issues'])}")

        return RiskCheckResult(passed=True)
