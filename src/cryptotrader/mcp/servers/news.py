"""News MCP Server — RSS aggregation and SoSoValue news tools."""

from __future__ import annotations

import asyncio

from cryptotrader.mcp.compat import FastMCP
from cryptotrader.mcp.utils import truncate_response

mcp = FastMCP("cryptotrader-news")


@mcp.tool()
async def news_rss(max_per_source: int = 5) -> dict:
    """查询 RSS 新闻聚合（CoinDesk / CoinTelegraph / Decrypt）。"""
    from cryptotrader.data.providers.rss_news import fetch_crypto_news

    articles = await asyncio.to_thread(fetch_crypto_news, max_per_source)
    return truncate_response({"articles": articles, "count": len(articles)})


@mcp.tool()
async def news_sosovalue(page: int = 1) -> dict:
    """查询 SoSoValue 特色新闻。"""
    from cryptotrader.config import load_config
    from cryptotrader.data.providers.sosovalue import fetch_news

    cfg = load_config()
    api_key = cfg.providers.sosovalue_api_key
    if not api_key:
        return {"articles": [], "count": 0, "data_available": False}
    articles = await fetch_news(api_key, page_size=20)
    return truncate_response({"articles": articles, "count": len(articles), "data_available": True})


if __name__ == "__main__":
    mcp.run(transport="stdio")
