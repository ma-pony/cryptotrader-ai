"""News MCP tools; credentialed calls receive an explicit provider key."""

from __future__ import annotations

import asyncio

from cryptotrader.mcp.compat import FastMCP
from cryptotrader.mcp.utils import truncate_response

mcp = FastMCP("cryptotrader-news")


@mcp.tool()
async def news_rss(max_per_source: int = 5) -> dict:
    from cryptotrader.data.providers.rss_news import fetch_crypto_news

    articles = await asyncio.to_thread(fetch_crypto_news, max_per_source)
    return truncate_response({"articles": articles, "count": len(articles)})


@mcp.tool()
async def news_sosovalue(page: int = 1, provider_key: str = "") -> dict:
    if not provider_key:
        return {"articles": [], "count": 0, "data_available": False}
    from cryptotrader.data.providers.sosovalue import fetch_news

    articles = await fetch_news(provider_key, page_size=20)
    return truncate_response({"articles": articles, "count": len(articles), "data_available": True})
