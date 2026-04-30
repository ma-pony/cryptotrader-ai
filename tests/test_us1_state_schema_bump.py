"""US1 (spec 013) Phase 3c — state schema v2 (Pair) + checkpoint compat shim.

Verifies:
- ``build_initial_state(pair=...)`` accepts both ``Pair`` and ``str``, stores
  a ``Pair`` instance internally
- Schema version marker (``_state_schema_version``) is set
- ``get_pair(state)`` returns the ``Pair``; legacy str triggers compat shim
  with WARN (FR-204)
- Round-trip: scheduler-style construction then node-style read produces
  the same canonical str
"""

from __future__ import annotations

import logging

import pytest

from cryptotrader.config import load_config
from cryptotrader.pair import Pair
from cryptotrader.state import STATE_SCHEMA_VERSION, build_initial_state, get_pair


class TestBuildInitialStateAcceptsBoth:
    def test_str_input_stored_as_pair(self) -> None:
        config = load_config()
        state = build_initial_state("BTC/USDT", config=config)
        assert isinstance(state["metadata"]["pair"], Pair)
        assert state["metadata"]["pair"].canonical() == "BTC/USDT"

    def test_pair_input_stored_as_is(self) -> None:
        config = load_config()
        p = Pair.parse("ETH/USDT:USDT")
        state = build_initial_state(p, config=config)
        assert state["metadata"]["pair"] is p

    def test_swap_str_input_preserves_market_type(self) -> None:
        config = load_config()
        state = build_initial_state("BTC/USDT:USDT", config=config)
        assert state["metadata"]["pair"].market_type == "swap"
        assert state["metadata"]["pair"].settle == "USDT"

    def test_schema_version_marker_set(self) -> None:
        config = load_config()
        state = build_initial_state("BTC/USDT", config=config)
        assert state["metadata"]["_state_schema_version"] == STATE_SCHEMA_VERSION
        assert STATE_SCHEMA_VERSION == 2


@pytest.fixture
def reset_legacy_warn_cache():
    """Auto-clean ``_legacy_str_warned`` so each test starts from a known state.

    spec 014 followup: previously the cache leaked across tests, making the
    "warns only once" property order-dependent and silently broken.
    """
    from cryptotrader.state import _legacy_str_warned

    snapshot = set(_legacy_str_warned)
    yield _legacy_str_warned
    _legacy_str_warned.clear()
    _legacy_str_warned.update(snapshot)


class TestGetPairCompatShim:
    def test_pair_input_passthrough(self) -> None:
        p = Pair.parse("BTC/USDT:USDT")
        state = {"metadata": {"pair": p}}
        assert get_pair(state) is p  # type: ignore[arg-type]

    def test_legacy_str_input_coerced_with_warn(self, caplog, reset_legacy_warn_cache) -> None:
        reset_legacy_warn_cache.discard("XRP/USDT")
        state = {"metadata": {"pair": "XRP/USDT"}}
        with caplog.at_level(logging.WARNING, logger="cryptotrader.state"):
            result = get_pair(state)  # type: ignore[arg-type]
        assert isinstance(result, Pair)
        assert result.canonical() == "XRP/USDT"
        assert any("legacy v1 schema" in r.message for r in caplog.records), (
            "expected WARN on str-pair compat shim per FR-204"
        )

    def test_legacy_str_warns_only_once_per_pair(self, caplog, reset_legacy_warn_cache) -> None:
        reset_legacy_warn_cache.discard("DOGE/USDT")
        state = {"metadata": {"pair": "DOGE/USDT"}}
        with caplog.at_level(logging.WARNING, logger="cryptotrader.state"):
            get_pair(state)  # type: ignore[arg-type]
            get_pair(state)  # type: ignore[arg-type]
            get_pair(state)  # type: ignore[arg-type]
        warns = [r for r in caplog.records if "legacy v1 schema" in r.message and "DOGE/USDT" in r.message]
        assert len(warns) == 1, f"expected 1 WARN, got {len(warns)}"

    def test_invalid_type_raises(self) -> None:
        state = {"metadata": {"pair": 42}}
        with pytest.raises(TypeError, match="Pair or str"):
            get_pair(state)  # type: ignore[arg-type]


class TestSchedulerStyleRoundTrip:
    def test_canonical_str_through_pipeline(self) -> None:
        """Phase 2 scheduler passes canonical str → build_initial_state coerces
        → consumer's get_pair().canonical() returns the same str."""
        config = load_config()
        for canonical in ["BTC/USDT", "ETH/USDT:USDT", "BTC/USD:BTC"]:
            state = build_initial_state(canonical, config=config)
            assert get_pair(state).canonical() == canonical
