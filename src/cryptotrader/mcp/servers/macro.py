"""Macro MCP tools with explicit provider credentials supplied by the runtime."""

from __future__ import annotations

from cryptotrader.mcp.compat import FastMCP
from cryptotrader.mcp.utils import truncate_response

mcp = FastMCP("cryptotrader-macro")
_FG_LABELS = {(0, 25): "Extreme Fear", (25, 50): "Fear", (50, 75): "Greed", (75, 101): "Extreme Greed"}


def _classify_fg(value: int) -> str:
    return next((label for (lo, hi), label in _FG_LABELS.items() if lo <= value < hi), "Unknown")


@mcp.tool()
async def macro_fear_greed() -> dict:
    from cryptotrader.data.macro import _fetch_fear_greed

    value, history = await _fetch_fear_greed()
    return {"value": value, "classification": _classify_fg(value), "history_7d": history}


@mcp.tool()
async def macro_btc_dominance() -> dict:
    from cryptotrader.data.macro import _fetch_btc_dominance

    return {"btc_dominance": await _fetch_btc_dominance()}


@mcp.tool()
async def macro_fred_series(series_id: str, provider_key: str = "") -> dict:
    """Fetch FRED only when an explicit credential has been resolved by a caller."""
    if not provider_key:
        return {"value": 0.0, "series_id": series_id, "data_available": False}
    from cryptotrader.data.macro import _fetch_fred

    return {"value": await _fetch_fred(series_id, provider_key), "series_id": series_id, "data_available": True}


@mcp.tool()
async def macro_etf_flow(etf_type: str, provider_key: str = "") -> dict:
    if not provider_key:
        return {"net_flow": 0.0, "data_available": False}
    from cryptotrader.data.providers.sosovalue import fetch_etf_metrics

    return truncate_response({**await fetch_etf_metrics(provider_key, etf_type), "data_available": True})
