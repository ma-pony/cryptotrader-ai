"""On-chain MCP tools with explicit provider credentials at the call boundary."""

from __future__ import annotations

from cryptotrader.mcp.compat import FastMCP
from cryptotrader.mcp.utils import truncate_response

mcp = FastMCP("cryptotrader-onchain")


@mcp.tool()
async def onchain_defi_tvl(chain: str = "Ethereum") -> dict:
    """Return DeFi TVL from the public DefiLlama provider."""
    from cryptotrader.data.providers.defillama import fetch_tvl

    return truncate_response(await fetch_tvl(chain))


@mcp.tool()
async def onchain_derivatives(symbol: str = "BTC", provider_key: str = "") -> dict:
    """Return CoinGlass derivatives data when the caller supplies its key."""
    if not provider_key:
        return {"open_interest": 0.0, "liquidations_24h": {}, "data_available": False}
    from cryptotrader.data.providers.coinglass import fetch_derivatives

    return truncate_response({**await fetch_derivatives(provider_key, symbol), "data_available": True})


@mcp.tool()
async def onchain_exchange_netflow(provider_key: str = "") -> dict:
    """Return CryptoQuant exchange netflow with an explicit provider key."""
    if not provider_key:
        return {"exchange_netflow": 0.0, "data_available": False}
    from cryptotrader.data.providers.cryptoquant import fetch_exchange_netflow

    return {"exchange_netflow": await fetch_exchange_netflow(provider_key), "data_available": True}


@mcp.tool()
async def onchain_whale_transfers(provider_key: str = "") -> dict:
    """Return WhaleAlert transfers with an explicit provider key."""
    if not provider_key:
        return {"transfers": [], "count": 0, "data_available": False}
    from cryptotrader.data.providers.whale_alert import fetch_whale_transfers

    transfers = await fetch_whale_transfers(provider_key)
    return truncate_response({"transfers": transfers, "count": len(transfers), "data_available": True})
