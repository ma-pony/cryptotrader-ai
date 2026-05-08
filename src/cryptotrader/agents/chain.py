"""On-chain / derivatives analysis agent — uses tool-calling to actively query data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.agents.base import ToolAgent
from cryptotrader.agents.data_tools import CHAIN_TOOLS

if TYPE_CHECKING:
    from cryptotrader.agents.prompt_builder import PromptBuilder


def _format_liquidations(liq: dict) -> list[str]:
    """Return prompt lines for liquidation data."""
    parts = []
    longs = liq.get("long", 0)
    shorts = liq.get("short", 0)
    if longs > 0 or shorts > 0:
        parts.append(f"Liquidations 24h: longs=${longs:,.0f}, shorts=${shorts:,.0f}")
    ratio_parts = []
    ls_ratio = liq.get("long_short_ratio", 0.0)
    tt_ratio = liq.get("top_trader_ratio", 0.0)
    tb_ratio = liq.get("taker_buy_sell_ratio", 0.0)
    if ls_ratio > 0:
        ls_label = "more longs (crowded long)" if ls_ratio > 1.0 else "more shorts (crowded short)"
        ratio_parts.append(f"long/short ratio={ls_ratio:.2f} ({ls_label})")
    if tt_ratio > 0:
        tt_label = "top traders net long" if tt_ratio > 1.0 else "top traders net short"
        ratio_parts.append(f"top trader ratio={tt_ratio:.2f} ({tt_label})")
    if tb_ratio > 0:
        tb_label = "aggressive buying" if tb_ratio > 1.0 else "aggressive selling"
        ratio_parts.append(f"taker buy/sell ratio={tb_ratio:.2f} ({tb_label})")
    if ratio_parts:
        parts.append("Derivatives ratios: " + ", ".join(ratio_parts))
    return parts


def _format_btc_network_health(oc: object) -> list[str]:
    """Return prompt lines for BTC network health metrics, or [] if all are zero."""
    lines = []
    active_addr = getattr(oc, "btc_active_addresses", 0.0)
    tx_count = getattr(oc, "btc_tx_count", 0.0)
    avg_fee = getattr(oc, "btc_avg_fee_usd", 0.0)
    difficulty = getattr(oc, "btc_difficulty", 0.0)
    if active_addr > 0:
        lines.append(f"  Active addresses: {active_addr:,.0f}")
    if tx_count > 0:
        lines.append(f"  Daily transactions: {tx_count:,.0f}")
    if avg_fee > 0:
        lines.append(f"  Avg fee: ${avg_fee:.2f}")
    if difficulty > 0:
        lines.append(f"  Difficulty: {difficulty / 1e12:.1f}T")
    return lines


class ChainAgent(ToolAgent):
    def __init__(self, *, prompt_builder: PromptBuilder, model: str = "", backtest_mode: bool = False) -> None:
        from cryptotrader.agents.skills.tool import load_skill_tool

        super().__init__(
            agent_id="chain",
            prompt_builder=prompt_builder,
            tools=[*CHAIN_TOOLS, load_skill_tool],
            model=model,
            backtest_mode=backtest_mode,
        )

    _KNOWN_EXCHANGES = frozenset(
        {"binance", "coinbase", "kraken", "okx", "bybit", "bitfinex", "huobi", "kucoin", "gate", "gemini"}
    )

    def _format_whale_transfers(self, transfers: list[dict]) -> str:
        lines = ["Whale transfers (24h):"]
        for t in transfers[:5]:
            usd = t.get("amount_usd", 0)
            frm = t.get("from", "unknown")
            to = t.get("to", "unknown")
            frm_lower = frm.lower()
            to_lower = to.lower()
            frm_is_ex = any(ex in frm_lower for ex in self._KNOWN_EXCHANGES)
            to_is_ex = any(ex in to_lower for ex in self._KNOWN_EXCHANGES)
            if frm_is_ex and not to_is_ex:
                direction = "OUTFLOW — accumulation"
            elif not frm_is_ex and to_is_ex:
                direction = "INFLOW — potential selling"
            else:
                direction = "exchange-to-exchange" if frm_is_ex else "unknown direction"
            lines.append(f"  - ${usd / 1e6:.1f}M from {frm} → {to} ({direction})")
        return "\n".join(lines)
