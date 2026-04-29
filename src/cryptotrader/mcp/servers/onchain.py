"""Onchain MCP Server — DeFi TVL, derivatives, exchange netflow, whale transfers."""

from __future__ import annotations

from cryptotrader.mcp.compat import FastMCP
from cryptotrader.mcp.utils import truncate_response

mcp = FastMCP("cryptotrader-onchain")


@mcp.tool()
async def onchain_defi_tvl(chain: str = "Ethereum") -> dict:
    """查询 DeFi TVL（DefiLlama）。"""
    from cryptotrader.data.providers.defillama import fetch_tvl

    return truncate_response(await fetch_tvl(chain))


@mcp.tool()
async def onchain_derivatives(symbol: str = "BTC") -> dict:
    """查询衍生品持仓量和清算量（CoinGlass）。"""
    from cryptotrader.config import load_config
    from cryptotrader.data.providers.coinglass import fetch_derivatives

    cfg = load_config()
    api_key = cfg.providers.coinglass_api_key
    if not api_key:
        return {"open_interest": 0.0, "liquidations_24h": {}, "data_available": False}
    data = await fetch_derivatives(api_key, symbol)
    return truncate_response({**data, "data_available": True})


@mcp.tool()
async def onchain_exchange_netflow() -> dict:
    """查询交易所净流量（CryptoQuant）。"""
    from cryptotrader.config import load_config
    from cryptotrader.data.providers.cryptoquant import fetch_exchange_netflow

    cfg = load_config()
    api_key = cfg.providers.cryptoquant_api_key
    if not api_key:
        return {"exchange_netflow": 0.0, "data_available": False}
    value = await fetch_exchange_netflow(api_key)
    return {"exchange_netflow": value, "data_available": True}


@mcp.tool()
async def onchain_whale_transfers() -> dict:
    """查询大额转账监控（WhaleAlert）。"""
    from cryptotrader.config import load_config
    from cryptotrader.data.providers.whale_alert import fetch_whale_transfers

    cfg = load_config()
    api_key = cfg.providers.whale_alert_api_key
    if not api_key:
        return {"transfers": [], "count": 0, "data_available": False}
    transfers = await fetch_whale_transfers(api_key)
    return truncate_response({"transfers": transfers, "count": len(transfers), "data_available": True})


if __name__ == "__main__":
    mcp.run(transport="stdio")
