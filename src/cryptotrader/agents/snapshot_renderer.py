"""Crypto-domain snapshot renderer — render_crypto_snapshot().

Verbatim migration of BaseAgent._build_prompt() logic into a standalone
module (spec 017b FR-Y11/Y12/Y13). Keeps all domain semantics:
  - funding rate ELEVATED / NEGATIVE annotation
  - futures volume SPIKE / LOW annotation
  - open interest
  - news headlines sanitized via sanitize_input()
  - data quality warnings (on-chain / news / macro)
  - experience field capped via sanitize_input(max_chars=4000)
  - TechAgent indicators dict rendered when present in snapshot
"""

from __future__ import annotations

from cryptotrader.agents.base import FUNDING_RATE_HIGH, FUNDING_RATE_LOW
from cryptotrader.security import sanitize_input


def render_crypto_snapshot(snapshot: dict, experience: str = "") -> str:
    """将 snapshot dict 渲染为 markdown 字符串，供 PromptBuilder 组装 HumanMessage。

    Args:
        snapshot: 含 pair / timestamp / market.* / news.* / onchain.* / macro.* 等字段的 dict；
                  TechAgent 会在调用前注入 'indicators' 字段。
        experience: 可选的历史经验文本；非空时附加到末尾（经 sanitize_input 截断至 4000 字符）。

    Returns:
        格式化后的 markdown 文本字符串。

    安全保证：
        - 所有外部内容（news headlines / experience）经 sanitize_input() 处理。
        - 内部 prompt（system_prompt / output_schema）不经过 sanitize（trusted）。
        - 遵循 spec 015 防注入 invariant。
    """
    # ── Technical Indicators (TechAgent only) ────────────────────────────────
    indicators = snapshot.get("indicators")
    indicator_parts: list[str] = []
    if indicators:
        indicator_parts.append(f"Technical Indicators:\n{indicators}")

    # ── Core market fields ───────────────────────────────────────────────────
    pair = snapshot.get("pair", "<missing>")
    timestamp = snapshot.get("timestamp", "<missing>")
    ticker = snapshot.get(
        "ticker",
        snapshot.get("market", {}).get("ticker", "<missing>")
        if isinstance(snapshot.get("market"), dict)
        else "<missing>",
    )
    volatility = snapshot.get(
        "volatility",
        snapshot.get("market", {}).get("volatility", 0.0) if isinstance(snapshot.get("market"), dict) else 0.0,
    )
    fr = snapshot.get(
        "funding_rate",
        snapshot.get("market", {}).get("funding_rate", 0.0) if isinstance(snapshot.get("market"), dict) else 0.0,
    )

    # liquidations_24h — may be nested under onchain or at top level
    liq = snapshot.get("liquidations_24h", {})
    if not liq:
        onchain_raw = snapshot.get("onchain", {})
        if isinstance(onchain_raw, dict):
            liq = onchain_raw.get("liquidations_24h", {})
    if not isinstance(liq, dict):
        liq = {}

    vol_ratio = liq.get("volume_ratio", 0)
    fut_vol = liq.get("futures_volume", 0)

    parts: list[str] = [
        f"Pair: {pair}",
        f"Timestamp: {timestamp}",
        f"Ticker: {ticker}",
        f"Volatility: {float(volatility):.4f}",
        f"Funding rate: {float(fr):.6f}"
        + (
            " (ELEVATED — crowded long)"
            if float(fr) > FUNDING_RATE_HIGH
            else " (NEGATIVE — crowded short)"
            if float(fr) < FUNDING_RATE_LOW
            else ""
        ),
    ]

    if fut_vol > 0:
        parts.append(
            f"Futures volume: {fut_vol:,.0f} BTC, vs 20d avg: {vol_ratio:.2f}x"
            + (" (SPIKE)" if vol_ratio > 1.5 else " (LOW)" if vol_ratio < 0.7 else "")
        )

    # Open interest
    open_interest = snapshot.get("open_interest", 0)
    if not open_interest:
        onchain_raw = snapshot.get("onchain", {})
        if isinstance(onchain_raw, dict):
            open_interest = onchain_raw.get("open_interest", 0)
    if open_interest and float(open_interest) > 0:
        parts.append(f"Open interest: {float(open_interest):,.0f}")

    # ── On-chain context (spec 021 F1) ───────────────────────────────────────
    # Render full onchain payload so chain_agent has signals beyond just
    # open_interest + exchange_netflow. Previously these Binance-public
    # derivative ratios were fetched every cycle but never made it into
    # the prompt, leaving chain_agent permanently at sufficiency=low.
    onchain_raw = snapshot.get("onchain", {})
    if isinstance(onchain_raw, dict):
        onchain_lines: list[str] = []
        liq_dict = onchain_raw.get("liquidations_24h", {}) or {}
        ls = float(liq_dict.get("long_short_ratio", 0) or 0)
        top_t = float(liq_dict.get("top_trader_ratio", 0) or 0)
        taker = float(liq_dict.get("taker_buy_sell_ratio", 0) or 0)
        if ls > 0:
            tag = " (crowded long)" if ls > 1.5 else " (crowded short)" if ls < 0.7 else ""
            onchain_lines.append(f"Long/short account ratio: {ls:.2f}{tag}")
        if top_t > 0:
            tag = (
                " (top traders net long)"
                if top_t > 1.5
                else " (top traders net short)"
                if top_t < 0.7
                else ""
            )
            onchain_lines.append(f"Top trader L/S ratio: {top_t:.2f}{tag}")
        if taker > 0:
            tag = (
                " (aggressive buying)"
                if taker > 1.1
                else " (aggressive selling)"
                if taker < 0.9
                else ""
            )
            onchain_lines.append(f"Taker buy/sell ratio: {taker:.2f}{tag}")

        liq_long = float(liq_dict.get("long_24h", 0) or 0)
        liq_short = float(liq_dict.get("short_24h", 0) or 0)
        if liq_long + liq_short > 0:
            bigger = "long" if liq_long > liq_short else "short"
            onchain_lines.append(
                f"24h liquidations: long ${liq_long/1e6:.1f}M, short ${liq_short/1e6:.1f}M ({bigger}-heavy)"
            )

        whales = onchain_raw.get("whale_transfers", []) or []
        if whales:
            onchain_lines.append(f"Whale transfers (≥$500k, 24h): {len(whales)} events")

        defi_tvl = float(onchain_raw.get("defi_tvl", 0) or 0)
        defi_chg = float(onchain_raw.get("defi_tvl_change_7d", 0) or 0)
        if defi_tvl > 0:
            chg_tag = f" ({defi_chg:+.1%} 7d)" if abs(defi_chg) > 0.001 else ""
            onchain_lines.append(f"DeFi TVL: ${defi_tvl/1e9:.2f}B{chg_tag}")

        btc_addr = float(onchain_raw.get("btc_active_addresses", 0) or 0)
        btc_fee = float(onchain_raw.get("btc_avg_fee_usd", 0) or 0)
        bits = []
        if btc_addr > 0:
            bits.append(f"active addrs {btc_addr:,.0f}")
        # Reject implausible fee values — the local store field is named
        # `btc_avg_fee_usd` but historical data has values in satoshis (200k+),
        # not USD. A real BTC avg fee in USD is ~$0.50-$100. Anything outside
        # that range is a unit mismatch and would mislead the LLM ("avg fee
        # $213k!?"), so suppress rather than render bad data.
        if 0 < btc_fee < 1000:
            bits.append(f"avg fee ${btc_fee:.2f}")
        if bits:
            onchain_lines.append("BTC network: " + ", ".join(bits))

        if onchain_lines:
            parts.append("On-chain context:\n  " + "\n  ".join(onchain_lines))

    # News headlines — apply sanitize_input() to each external headline
    # (spec 015 req 7.5: prompt injection defence).
    headlines = snapshot.get("headlines", [])
    if not headlines:
        news_raw = snapshot.get("news", {})
        if isinstance(news_raw, dict):
            headlines = news_raw.get("headlines", [])
    if headlines:
        safe_headlines = [sanitize_input(h) for h in headlines]
        parts.append("News headlines:\n" + "\n".join(f"  - {h}" for h in safe_headlines if h))

    # ── Data quality warnings ────────────────────────────────────────────────
    # Tell agents when sources are empty so they don't infer from missing data.
    warnings: list[str] = []

    onchain_raw = snapshot.get("onchain", {})
    oi_val = snapshot.get("open_interest", 0)
    netflow_val = snapshot.get("exchange_netflow", 0)
    if isinstance(onchain_raw, dict):
        oi_val = onchain_raw.get("open_interest", oi_val)
        netflow_val = onchain_raw.get("exchange_netflow", netflow_val)
    if float(oi_val) == 0 and float(netflow_val) == 0:
        warnings.append("On-chain data unavailable (no API keys configured). Do NOT infer from missing data.")

    if not headlines:
        warnings.append("News sentiment unavailable. Do NOT assume neutral sentiment from missing data.")

    macro_raw = snapshot.get("macro", {})
    if not isinstance(macro_raw, dict):
        macro_raw = {}

    def _macro(name: str, default: float = 0.0) -> float:
        """Read a macro field, preferring top-level snapshot key, falling back to macro_raw."""
        v = snapshot.get(name)
        if v is None:
            v = macro_raw.get(name, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    fed_rate_val = _macro("fed_rate")
    dxy_val = _macro("dxy")
    btc_dom = _macro("btc_dominance")
    fg_idx = int(_macro("fear_greed_index", 50))
    etf_in = _macro("etf_daily_net_inflow")
    etf_aum = _macro("etf_total_net_assets")
    vix = _macro("vix")
    sp500 = _macro("sp500")
    yc = _macro("yield_curve")
    m2 = _macro("m2_supply")
    cpi = _macro("cpi")
    hashrate = _macro("btc_hashrate")

    # Only render the macro block when at least one field is populated, so the
    # prompt stays clean if macro pipeline is fully offline.
    macro_lines: list[str] = []
    if fed_rate_val > 0 or dxy_val > 0:
        macro_lines.append(f"Fed funds rate: {fed_rate_val:.2f}%" + (f" | DXY: {dxy_val:.2f}" if dxy_val > 0 else ""))
    if vix > 0 or sp500 > 0:
        bits = []
        if vix > 0:
            tag = " (HIGH FEAR)" if vix > 25 else " (LOW FEAR)" if vix < 15 else ""
            bits.append(f"VIX: {vix:.2f}{tag}")
        if sp500 > 0:
            bits.append(f"S&P500: {sp500:,.2f}")
        macro_lines.append(" | ".join(bits))
    if yc != 0 or m2 > 0 or cpi > 0:
        bits = []
        if yc != 0:
            tag = " (INVERTED)" if yc < 0 else ""
            bits.append(f"10y-2y yield curve: {yc:.2f}%{tag}")
        if m2 > 0:
            bits.append(f"M2: ${m2:,.1f}B")
        if cpi > 0:
            bits.append(f"CPI: {cpi:.2f}")
        macro_lines.append(" | ".join(bits))
    if btc_dom > 0 or fg_idx != 50:
        fg_tag = (
            " (EXTREME FEAR)"
            if fg_idx <= 24
            else " (FEAR)"
            if fg_idx <= 44
            else " (NEUTRAL)"
            if fg_idx <= 54
            else " (GREED)"
            if fg_idx <= 74
            else " (EXTREME GREED)"
        )
        bits = []
        if btc_dom > 0:
            bits.append(f"BTC dominance: {btc_dom:.2f}%")
        bits.append(f"Fear & Greed: {fg_idx}{fg_tag}")
        macro_lines.append(" | ".join(bits))
    if etf_in != 0 or etf_aum > 0:
        bits = []
        if etf_in != 0:
            sign = "+" if etf_in > 0 else ""
            bits.append(f"ETF 24h net flow: {sign}${etf_in / 1e6:,.1f}M")
        if etf_aum > 0:
            bits.append(f"ETF AUM: ${etf_aum / 1e9:,.1f}B")
        macro_lines.append(" | ".join(bits))
    if hashrate > 0:
        macro_lines.append(f"BTC hashrate: {hashrate / 1e9:.1f} GH/s")

    if macro_lines:
        parts.append("Macro context:\n  " + "\n  ".join(macro_lines))

    if fed_rate_val == 0 and dxy_val == 0 and vix == 0:
        warnings.append("Macro data unavailable (FRED/DXY/VIX). Do NOT infer from zero values — they are missing.")

    if warnings:
        parts.append("⚠ DATA QUALITY WARNINGS:\n" + "\n".join(f"  - {w}" for w in warnings))

    # ── Experience ───────────────────────────────────────────────────────────
    if experience:
        parts.append(sanitize_input(experience, max_chars=4000))

    result_parts = indicator_parts + parts
    return "\n".join(result_parts)
