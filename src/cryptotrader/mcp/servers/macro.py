"""Macro MCP Server — Fear & Greed, BTC dominance, FRED, ETF flow tools."""

from __future__ import annotations

from cryptotrader.mcp.compat import FastMCP
from cryptotrader.mcp.utils import truncate_response

mcp = FastMCP("cryptotrader-macro")

_FG_LABELS = {(0, 25): "Extreme Fear", (25, 50): "Fear", (50, 75): "Greed", (75, 101): "Extreme Greed"}


def _classify_fg(value: int) -> str:
    for (lo, hi), label in _FG_LABELS.items():
        if lo <= value < hi:
            return label
    return "Unknown"


@mcp.tool()
async def macro_fear_greed() -> dict:
    """查询 Fear & Greed 指数。"""
    from cryptotrader.data.macro import _fetch_fear_greed

    value, history = await _fetch_fear_greed()
    return {"value": value, "classification": _classify_fg(value), "history_7d": history}


@mcp.tool()
async def macro_btc_dominance() -> dict:
    """查询 BTC 市值占比。"""
    from cryptotrader.data.macro import _fetch_btc_dominance

    dominance = await _fetch_btc_dominance()
    return {"btc_dominance": dominance}


@mcp.tool()
async def macro_fred_series(series_id: str = "DFF") -> dict:
    """查询 FRED 宏观经济数据序列。"""
    from cryptotrader.config import load_config
    from cryptotrader.data.macro import _fetch_fred

    cfg = load_config()
    api_key = cfg.providers.fred_api_key
    if not api_key:
        return {"value": 0.0, "series_id": series_id, "data_available": False}
    value = await _fetch_fred(series_id, api_key)
    return {"value": value, "series_id": series_id, "data_available": True}


@mcp.tool()
async def macro_etf_flow(etf_type: str = "us-btc-spot") -> dict:
    """查询 ETF 资金流数据。"""
    from cryptotrader.config import load_config
    from cryptotrader.data.providers.sosovalue import fetch_etf_metrics

    cfg = load_config()
    api_key = cfg.providers.sosovalue_api_key
    if not api_key:
        return {"net_flow": 0.0, "data_available": False}
    data = await fetch_etf_metrics(api_key, etf_type)
    return truncate_response({**data, "data_available": True})


if __name__ == "__main__":
    mcp.run(transport="stdio")
