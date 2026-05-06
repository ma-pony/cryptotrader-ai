"""LiveExchange position-key contract per spec 013 Phase 3b.

After spec 013, ccxt's unified symbol IS the project canonical pair (spot
``BTC/USDT``, linear perp ``BTC/USDT:USDT``, inverse perp ``BTC/USD:BTC``).
LiveExchange.get_positions returns ccxt symbols verbatim — no
canonicalization translation. The earlier ``_canonical_pair`` band-aid
(commit d05a0bf) is retired by this phase.

T039 will delete this file once the migration is fully verified.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.execution.exchange import LiveExchange


def _make_exchange(*, leverage: int = 1, margin_mode: str = "isolated") -> LiveExchange:
    with patch("ccxt.async_support.okx") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.load_markets = AsyncMock()
        mock_cls.return_value = mock_inst
        return LiveExchange(
            "okx",
            "k",
            "s",
            sandbox=True,
            passphrase="p",
            leverage=leverage,
            margin_mode=margin_mode,
        )


@pytest.mark.asyncio
async def test_get_positions_returns_ccxt_perp_symbol_verbatim():
    """OKX perp ``BTC/USDT:USDT`` is the canonical key — no translation."""
    ex = _make_exchange()
    ex._exchange.fetch_positions = AsyncMock(
        return_value=[
            {
                "symbol": "BTC/USDT:USDT",
                "contracts": 0.02,
                "side": "long",
                "entryPrice": 84708.8,
                "unrealizedPnl": -1.68,
                "liquidationPrice": None,
            }
        ]
    )
    ex._markets_loaded = True

    positions = await ex.get_positions()
    assert "BTC/USDT:USDT" in positions, "ccxt unified symbol is the canonical key"
    assert "BTC/USDT" not in positions, "spot form must not appear for a perp position"
    assert positions["BTC/USDT:USDT"]["amount"] == 0.02
    assert positions["BTC/USDT:USDT"]["avg_price"] == 84708.8


@pytest.mark.asyncio
async def test_get_positions_keeps_spot_symbol_unchanged():
    ex = _make_exchange()
    ex._exchange.fetch_positions = AsyncMock(
        return_value=[{"symbol": "ETH/USDT", "contracts": 1.5, "side": "long", "entryPrice": 2300.0}]
    )
    ex._markets_loaded = True

    positions = await ex.get_positions()
    assert positions["ETH/USDT"]["amount"] == 1.5


@pytest.mark.asyncio
async def test_get_positions_returns_inverse_perp_symbol_verbatim():
    """Inverse perp ``BTC/USD:BTC`` survives unchanged."""
    ex = _make_exchange()
    ex._exchange.fetch_positions = AsyncMock(
        return_value=[{"symbol": "BTC/USD:BTC", "contracts": 100.0, "side": "long", "entryPrice": 84000.0}]
    )
    ex._markets_loaded = True

    positions = await ex.get_positions()
    assert "BTC/USD:BTC" in positions
    assert positions["BTC/USD:BTC"]["amount"] == 100.0


@pytest.mark.asyncio
async def test_get_positions_skips_zero_contract_entries():
    ex = _make_exchange()
    ex._exchange.fetch_positions = AsyncMock(
        return_value=[
            {"symbol": "BTC/USDT:USDT", "contracts": 0.0, "side": "long"},
            {"symbol": "ETH/USDT:USDT", "contracts": 0.5, "side": "long", "entryPrice": 2300.0},
        ]
    )
    ex._markets_loaded = True

    positions = await ex.get_positions()
    assert "BTC/USDT:USDT" not in positions
    assert "ETH/USDT:USDT" in positions


# ── _ensure_leverage: configurable perp leverage application ──


@pytest.mark.asyncio
async def test_ensure_leverage_noop_when_leverage_is_one():
    """leverage=1 must not call set_leverage (matches OKX default; saves API call)."""
    ex = _make_exchange(leverage=1)
    ex._exchange.set_leverage = AsyncMock()

    await ex._ensure_leverage("BTC/USDT:USDT")
    ex._exchange.set_leverage.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_leverage_skips_spot_pairs():
    """Spot symbols don't have leverage; never invoke set_leverage."""
    ex = _make_exchange(leverage=2)
    ex._exchange.set_leverage = AsyncMock()

    await ex._ensure_leverage("BTC/USDT")  # spot
    ex._exchange.set_leverage.assert_not_called()
    # cached so future calls also skip
    assert "BTC/USDT" in ex._leveraged_symbols


@pytest.mark.asyncio
async def test_ensure_leverage_sets_both_pos_sides_for_perp():
    """In long_short_mode OKX requires posSide for both long and short."""
    ex = _make_exchange(leverage=2, margin_mode="isolated")
    ex._exchange.set_leverage = AsyncMock(return_value={})

    await ex._ensure_leverage("BTC/USDT:USDT")
    assert ex._exchange.set_leverage.call_count == 2
    # Inspect the kwargs/positional args for both calls.
    sides = []
    for call in ex._exchange.set_leverage.call_args_list:
        leverage, symbol, params = call.args
        assert leverage == 2
        assert symbol == "BTC/USDT:USDT"
        assert params["mgnMode"] == "isolated"
        sides.append(params["posSide"])
    assert set(sides) == {"long", "short"}


@pytest.mark.asyncio
async def test_ensure_leverage_idempotent_on_repeat_calls():
    """Second call for the same symbol is a no-op (already in cache)."""
    ex = _make_exchange(leverage=2)
    ex._exchange.set_leverage = AsyncMock(return_value={})

    await ex._ensure_leverage("BTC/USDT:USDT")
    await ex._ensure_leverage("BTC/USDT:USDT")
    assert ex._exchange.set_leverage.call_count == 2  # only the first call's two posSides


@pytest.mark.asyncio
async def test_ensure_leverage_swallows_errors_and_retries_until_limit():
    """Failure (e.g. already-open position) must not crash place_order. We
    retry on the next order so a config change can take effect once the
    locking position closes — but cap retries to avoid log spam.
    """
    ex = _make_exchange(leverage=3)
    ex._exchange.set_leverage = AsyncMock(side_effect=RuntimeError("position already open"))

    # First call: 2 set_leverage attempts, both fail; not cached yet (retryable)
    await ex._ensure_leverage("BTC/USDT:USDT")
    assert "BTC/USDT:USDT" not in ex._leveraged_symbols
    assert ex._leverage_attempts["BTC/USDT:USDT"] == 1
    assert ex._exchange.set_leverage.call_count == 2

    # Second call: another 2 attempts; still retryable
    await ex._ensure_leverage("BTC/USDT:USDT")
    assert "BTC/USDT:USDT" not in ex._leveraged_symbols
    assert ex._leverage_attempts["BTC/USDT:USDT"] == 2
    assert ex._exchange.set_leverage.call_count == 4

    # Third call: hits limit (3) → cached, no further retries
    await ex._ensure_leverage("BTC/USDT:USDT")
    assert "BTC/USDT:USDT" in ex._leveraged_symbols
    assert ex._exchange.set_leverage.call_count == 6

    # Fourth call: cached → no-op
    await ex._ensure_leverage("BTC/USDT:USDT")
    assert ex._exchange.set_leverage.call_count == 6


@pytest.mark.asyncio
async def test_ensure_leverage_retries_succeed_after_intermittent_failure():
    """Once the locking position closes (or the venue recovers), the next
    retry should apply the configured leverage."""
    ex = _make_exchange(leverage=3)
    # First call fails, second succeeds.
    call_count = {"n": 0}

    async def flaky(_lev, _sym, _params):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise RuntimeError("position already open")
        return {}

    ex._exchange.set_leverage = AsyncMock(side_effect=flaky)

    await ex._ensure_leverage("BTC/USDT:USDT")
    assert "BTC/USDT:USDT" not in ex._leveraged_symbols  # both posSides failed

    await ex._ensure_leverage("BTC/USDT:USDT")
    assert "BTC/USDT:USDT" in ex._leveraged_symbols  # both posSides ok this time
