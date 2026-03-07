"""Correlation risk check — rejects adding highly correlated positions."""

from __future__ import annotations

from cryptotrader.models import CheckResult, TradeVerdict

# Known high-correlation pairs (>0.85 historical correlation)
_CORRELATED_GROUPS: list[set[str]] = [
    {"BTC", "WBTC", "BTCB"},
    {"ETH", "WETH", "STETH", "CBETH"},
    {"SOL", "MSOL", "JITOSOL"},
    {"USDT", "USDC", "DAI", "BUSD"},
    {"DOGE", "SHIB", "FLOKI"},
]


def _find_group(symbol: str) -> set[str] | None:
    upper = symbol.upper()
    for group in _CORRELATED_GROUPS:
        if upper in group:
            return group
    return None


class CorrelationCheck:
    name = "correlation"
    max_correlated_positions: int = 2

    async def evaluate(self, verdict: TradeVerdict, portfolio: dict) -> CheckResult:
        if verdict.action == "hold":
            return CheckResult(passed=True)

        pair = getattr(verdict, "pair", "") or ""
        symbol = pair.split("/")[0] if "/" in pair else pair
        group = _find_group(symbol)
        if not group:
            return CheckResult(passed=True)

        # Count existing positions in the same correlation group
        positions = portfolio.get("positions", {})
        correlated_count = 0
        for pos_pair, pos_data in positions.items():
            pos_symbol = pos_pair.split("/")[0] if "/" in pos_pair else pos_pair
            amount = pos_data if isinstance(pos_data, int | float) else pos_data.get("amount", 0)
            if pos_symbol.upper() in group and amount != 0:
                correlated_count += 1

        if correlated_count >= self.max_correlated_positions:
            return CheckResult(
                passed=False,
                reason=f"Already {correlated_count} correlated positions in group {group}",
            )
        return CheckResult(passed=True)
