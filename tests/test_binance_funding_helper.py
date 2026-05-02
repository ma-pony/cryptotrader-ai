"""Tests for the shared ccxt-based funding helpers in data/providers/binance.py.

Replaces 4 pre-existing direct ``httpx`` calls to ``fapi/v1/fundingRate``
with a single ``fetch_funding_history_ccxt`` helper. These tests pin down:

- Symbol coercion (bare base / Binance native / spot canonical / perp canonical)
- Pagination behaviour (multi-page fetch, since-cursor advance, limit clamp)
- Drop-in shape compatibility with the legacy Binance JSON
  (``{fundingTime, fundingRate}`` dict per record)
- Lifecycle: ccxt client is closed after the call
- Error swallow on ccxt failure
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.data.providers.binance import (
    _perp_symbol_for,
    fetch_funding_history_ccxt,
    fetch_funding_rate_binance,
)

# ── _perp_symbol_for ──


@pytest.mark.parametrize(
    ("input_pair", "expected"),
    [
        ("BTC", "BTC/USDT:USDT"),  # bare base
        ("BTCUSDT", "BTC/USDT:USDT"),  # Binance native
        ("BTC/USDT", "BTC/USDT:USDT"),  # spot canonical
        ("BTC/USDT:USDT", "BTC/USDT:USDT"),  # already perp
        ("ETH/USDC", "ETH/USDC:USDC"),  # USDC variant
        ("ETHUSDC", "ETH/USDC:USDC"),  # USDC native
    ],
)
def test_perp_symbol_for_handles_all_input_shapes(input_pair: str, expected: str) -> None:
    assert _perp_symbol_for(input_pair) == expected


# ── fetch_funding_history_ccxt: shape + pagination ──


async def test_fetch_funding_history_returns_legacy_shape() -> None:
    """Each record must have ``fundingTime`` ms-int and ``fundingRate`` float."""
    fake = MagicMock()
    fake.fetch_funding_rate_history = AsyncMock(
        return_value=[
            {"timestamp": 1700000000000, "fundingRate": 0.0001, "symbol": "BTC/USDT:USDT"},
            {"timestamp": 1700028800000, "fundingRate": 0.0003, "symbol": "BTC/USDT:USDT"},
        ]
    )
    fake.close = AsyncMock()

    # ``page_size=10`` so the 2-row batch is treated as a short page and the
    # loop terminates after one fetch (otherwise the mock would keep
    # returning 2 rows == page_size and never short-circuit).
    with patch("cryptotrader.data.providers.binance._open_market_client", AsyncMock(return_value=fake)):
        out = await fetch_funding_history_ccxt("BTC/USDT", page_size=10, pause_seconds=0)

    assert out == [
        {"fundingTime": 1700000000000, "fundingRate": 0.0001},
        {"fundingTime": 1700028800000, "fundingRate": 0.0003},
    ]
    fake.close.assert_awaited_once()


async def test_fetch_funding_history_paginates_until_short_batch() -> None:
    """Helper keeps calling until ccxt returns fewer than ``page_size`` rows."""
    fake = MagicMock()
    fake.fetch_funding_rate_history = AsyncMock(
        side_effect=[
            [{"timestamp": 100, "fundingRate": 0.001}, {"timestamp": 200, "fundingRate": 0.002}],
            [{"timestamp": 300, "fundingRate": 0.003}],  # short → terminate
        ]
    )
    fake.close = AsyncMock()

    with patch("cryptotrader.data.providers.binance._open_market_client", AsyncMock(return_value=fake)):
        out = await fetch_funding_history_ccxt("BTC/USDT", page_size=2, pause_seconds=0)

    # All three records collected across 2 pages.
    assert [r["fundingTime"] for r in out] == [100, 200, 300]
    assert fake.fetch_funding_rate_history.await_count == 2
    # Cursor for the 2nd call should be last_ts + 1 = 201.
    second_call = fake.fetch_funding_rate_history.await_args_list[1]
    assert second_call.kwargs["since"] == 201


async def test_fetch_funding_history_respects_explicit_limit() -> None:
    """``limit`` truncates the result even mid-page."""
    fake = MagicMock()
    fake.fetch_funding_rate_history = AsyncMock(
        return_value=[{"timestamp": i, "fundingRate": 0.0001} for i in range(1, 6)]
    )
    fake.close = AsyncMock()

    with patch("cryptotrader.data.providers.binance._open_market_client", AsyncMock(return_value=fake)):
        out = await fetch_funding_history_ccxt("BTC/USDT", limit=3, page_size=10, pause_seconds=0)

    assert len(out) == 3
    assert [r["fundingTime"] for r in out] == [1, 2, 3]


async def test_fetch_funding_history_skips_entries_with_missing_fields() -> None:
    """Rows without timestamp or fundingRate are silently dropped, not crashed on."""
    fake = MagicMock()
    fake.fetch_funding_rate_history = AsyncMock(
        return_value=[
            {"timestamp": 100, "fundingRate": 0.001},  # ok
            {"timestamp": 200},  # missing rate
            {"fundingRate": 0.002},  # missing ts
            {"timestamp": 300, "fundingRate": 0.003},  # ok
        ]
    )
    fake.close = AsyncMock()

    with patch("cryptotrader.data.providers.binance._open_market_client", AsyncMock(return_value=fake)):
        out = await fetch_funding_history_ccxt("BTC/USDT", page_size=10, pause_seconds=0)

    assert [r["fundingTime"] for r in out] == [100, 300]


async def test_fetch_funding_history_closes_client_even_on_ccxt_error() -> None:
    fake = MagicMock()
    fake.fetch_funding_rate_history = AsyncMock(side_effect=RuntimeError("network"))
    fake.close = AsyncMock()

    with (
        patch("cryptotrader.data.providers.binance._open_market_client", AsyncMock(return_value=fake)),
        pytest.raises(RuntimeError, match="network"),
    ):
        await fetch_funding_history_ccxt("BTC/USDT", pause_seconds=0)

    # Connector still released.
    fake.close.assert_awaited_once()


# ── fetch_funding_rate_binance (current rate): ccxt path ──


async def test_fetch_funding_rate_binance_uses_ccxt_fetch_funding_rate() -> None:
    """The "latest" funding helper now calls ccxt instead of premiumIndex REST."""
    fake = MagicMock()
    fake.fetch_funding_rate = AsyncMock(
        return_value={"fundingRate": 0.00045, "fundingTimestamp": 1700000000000, "symbol": "BTC/USDT:USDT"}
    )
    fake.close = AsyncMock()

    with patch("cryptotrader.data.providers.binance._open_market_client", AsyncMock(return_value=fake)):
        out = await fetch_funding_rate_binance("BTC")

    assert out["funding_rate"] == 0.00045
    assert out["next_funding_time"] == 1700000000000
    fake.fetch_funding_rate.assert_awaited_once_with("BTC/USDT:USDT")
    fake.close.assert_awaited_once()


async def test_fetch_funding_rate_binance_swallows_errors() -> None:
    """ccxt raising returns the default ``{0.0, 0}`` shape, not an exception."""
    fake = MagicMock()
    fake.fetch_funding_rate = AsyncMock(side_effect=RuntimeError("blocked"))
    fake.close = AsyncMock()

    with patch("cryptotrader.data.providers.binance._open_market_client", AsyncMock(return_value=fake)):
        out = await fetch_funding_rate_binance("BTC")

    assert out == {"funding_rate": 0.0, "next_funding_time": 0}
