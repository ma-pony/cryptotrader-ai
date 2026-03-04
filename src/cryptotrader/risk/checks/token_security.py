"""Token security audit check."""
from cryptotrader.data.binance_audit import BinanceAudit
from cryptotrader.models import CheckResult, TradeVerdict


class TokenSecurityCheck:
    """Check token security via Binance audit API."""

    name = "token_security"

    def __init__(self):
        self.audit = BinanceAudit()

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        """Check if token passes security audit."""
        # Extract contract address from verdict (if available)
        contract = getattr(verdict, 'contract_address', None)
        if not contract:
            return CheckResult(passed=True, reason="No contract address to audit")

        result = await self.audit.audit_token(
            symbol=verdict.symbol,
            contract_address=contract,
            chain=getattr(verdict, 'chain', 'BSC')
        )

        if result['risk_level'] == 'HIGH':
            return CheckResult(
                passed=False,
                reason=f"High security risk: {', '.join(result['issues'])}"
            )

        return CheckResult(passed=True)
