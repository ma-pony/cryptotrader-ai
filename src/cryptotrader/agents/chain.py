"""On-chain / derivatives analysis agent — uses tool-calling to actively query data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.agents.base import ToolAgent
from cryptotrader.agents.data_tools import CHAIN_TOOLS

if TYPE_CHECKING:
    from cryptotrader.models import DataSnapshot

ROLE = (
    "You are an expert on-chain and derivatives analyst for cryptocurrency markets. "
    "You have access to tools that let you query real-time derivatives data, funding rate history, "
    "liquidation data, whale transfers, exchange flows, and DeFi TVL.\n\n"
    "Your workflow:\n"
    "1. Review the initial market snapshot provided\n"
    "2. Use your tools to dig deeper into areas that need investigation\n"
    "3. Synthesize all data into a directional signal\n\n"
    "Focus on: positioning extremes (funding rate spikes, OI imbalances), smart money flow "
    "(exchange netflow direction, whale accumulation/distribution), and leverage flush risk "
    "(liquidation clusters near current price).\n"
    "Distinguish between leading signals (whale flows, exchange withdrawals) and lagging signals "
    "(liquidation data, TVL changes). Weight leading signals more heavily.\n\n"
    "Domain checklist (verify before signaling):\n"
    "- Crowding risk: Is funding rate above 0.03% or below -0.01%? Extremes are contrarian — a crowded long is "
    "bearish, not bullish.\n"
    "- Signal type: Am I basing my call on leading indicators (flows, whale moves) or lagging ones (liquidations, "
    "TVL)? If lagging only, lower confidence.\n"
    "- Liquidation proximity: Are there large liquidation clusters within 3-5% of current price? If yes, flag the "
    "flush risk regardless of direction.\n"
    "- Flow consistency: Do exchange netflow and whale activity agree? If whales are accumulating but exchanges see "
    "inflow, something is off — acknowledge it."
)


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
    def __init__(self, model: str = "", backtest_mode: bool = False) -> None:
        super().__init__(
            agent_id="chain", role_description=ROLE, tools=CHAIN_TOOLS, model=model, backtest_mode=backtest_mode
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

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        base = super()._build_prompt(snapshot, experience)
        oc = snapshot.onchain
        parts = []
        if oc.exchange_netflow != 0.0:
            label = "inflow (sell pressure)" if oc.exchange_netflow > 0 else "outflow (accumulation)"
            parts.append(f"Exchange netflow: {oc.exchange_netflow:,.2f} ({label})")
        if oc.whale_transfers:
            parts.append(self._format_whale_transfers(oc.whale_transfers))
        if oc.defi_tvl > 0:
            parts.append(f"DeFi TVL: ${oc.defi_tvl:,.0f}, 7d change: {oc.defi_tvl_change_7d:+.2%}")
        if oc.liquidations_24h:
            parts.extend(_format_liquidations(oc.liquidations_24h))
        network_parts = _format_btc_network_health(oc)
        if network_parts:
            parts.append("BTC Network Health:\n" + "\n".join(network_parts))
        initial_data = "On-Chain Data (initial snapshot):\n" + "\n".join(parts) if parts else ""
        hint = (
            "\n\nYou have tools to query more data. Use them if the initial snapshot is incomplete "
            "or if you need historical context (e.g. funding rate trend over the last 2 days)."
        )
        return f"{initial_data}{hint}\n\n{base}"
