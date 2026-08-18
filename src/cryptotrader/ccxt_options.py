"""Exchange-specific ccxt market loading options."""

from __future__ import annotations

_FETCH_MARKETS_BY_EXCHANGE = {
    "binance": ["spot", "linear"],
    "bybit": ["spot", "linear"],
    "coinbase": ["spot"],
    "okx": ["spot", "swap"],
}


def fetch_market_types(exchange_id: str) -> list[str]:
    """Return market type names accepted by the selected ccxt exchange."""
    return _FETCH_MARKETS_BY_EXCHANGE.get(exchange_id, ["spot"]).copy()
