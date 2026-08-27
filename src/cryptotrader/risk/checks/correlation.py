"""Correlation risk check — rejects adding highly correlated positions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.risk.models import RiskCheckResult

if TYPE_CHECKING:
    from cryptotrader.config import PositionConfig
    from cryptotrader.risk.models import RiskRequest

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

    def __init__(self, config: PositionConfig) -> None:
        self._max_correlated = config.max_correlated_positions

    async def evaluate(self, request: RiskRequest, portfolio: dict) -> RiskCheckResult:
        if request.target.side == "flat":
            return RiskCheckResult(passed=True)

        from cryptotrader.pair import Pair

        pair = request.context.pair.canonical()
        symbol = request.context.pair.base
        group = _find_group(symbol)
        if not group:
            return RiskCheckResult(passed=True)

        # Count existing positions in the same correlation group
        positions = portfolio.get("positions", {})
        correlated_count = 0
        for pos_pair, pos_data in positions.items():
            if pos_pair == pair:
                continue
            try:
                pos_symbol = Pair.parse(pos_pair).base
            except (ValueError, NotImplementedError):
                pos_symbol = pos_pair
            amount = pos_data if isinstance(pos_data, int | float) else pos_data.get("amount", 0)
            if pos_symbol.upper() in group and amount != 0:
                correlated_count += 1

        if correlated_count >= self._max_correlated:
            return RiskCheckResult(
                passed=False,
                reason=f"Already {correlated_count} correlated positions in group {group}",
            )
        return RiskCheckResult(passed=True)
