"""US4 (spec 013, T034) — API responses expose canonical pair identity.

Validates the response shape for both /api/portfolio/snapshot and the
canonical multi-venue cycle endpoints. Uses unit-level assertions so the
test runs without a live DB.
"""

# ruff: noqa: F401, F811 - 导入 fixture 供本模块的 pytest 参数解析。

from __future__ import annotations

from tests.test_runtime_config_api import api_harness


class TestPortfolioPositionOutShape:
    def test_position_out_includes_pair_display_and_market_type(self) -> None:
        from api.routes.portfolio_v2 import PositionOut

        out = PositionOut(
            pair="BTC/USDT:USDT",
            pair_display="BTC/USDT (perp)",
            market_type="swap",
            side="long",
            size=0.02,
            avg_price=84500.0,
        )
        dump = out.model_dump()
        assert dump["pair"] == "BTC/USDT:USDT"
        assert dump["pair_display"] == "BTC/USDT (perp)"
        assert dump["market_type"] == "swap"

    def test_serialize_positions_derives_display_from_pair(self) -> None:
        from api.routes.portfolio_v2 import _serialize_positions

        raw = {
            "BTC/USDT": {"amount": 0.5, "avg_price": 80000.0},
            "ETH/USDT:USDT": {"amount": 1.0, "avg_price": 2300.0},
            "BTC/USD:BTC": {"amount": 100.0, "avg_price": 84000.0},
        }
        out = _serialize_positions(raw)
        by_pair = {p.pair: p for p in out}
        assert by_pair["BTC/USDT"].pair_display == "BTC/USDT"
        assert by_pair["BTC/USDT"].market_type == "spot"
        assert by_pair["ETH/USDT:USDT"].pair_display == "ETH/USDT (perp)"
        assert by_pair["ETH/USDT:USDT"].market_type == "swap"
        assert by_pair["BTC/USD:BTC"].pair_display == "BTC/USD (perp)"

    def test_serialize_positions_prefers_db_market_type(self) -> None:
        """Phase 5 stored market_type — when present, use it (over derivation).
        This guards against future Pair.parse semantics drift."""
        from api.routes.portfolio_v2 import _serialize_positions

        # Even if pair string is malformed, DB-stored market_type wins
        raw = {
            "BTC/USDT": {"amount": 0.5, "avg_price": 80000.0, "market_type": "swap"},
        }
        out = _serialize_positions(raw)
        assert out[0].market_type == "swap"

    def test_serialize_positions_falls_back_to_spot_on_bad_pair(self) -> None:
        from api.routes.portfolio_v2 import _serialize_positions

        raw = {"BAD-PAIR": {"amount": 1.0, "avg_price": 100.0}}
        out = _serialize_positions(raw)
        assert out[0].pair_display == "BAD-PAIR"
        assert out[0].market_type == "spot"


class TestMultiVenueCyclePair:
    async def test_cycle_book_includes_pair_market_type_and_connection_hierarchy(self, api_harness) -> None:
        from tests.test_multi_venue_journal import _record

        await api_harness.runtime.cycle.journal.save(_record())
        response = await api_harness.client.get("/api/cycles/cycle-1")

        assert response.status_code == 200
        book = response.json()["books"][0]
        assert book["pair"] == "BTC/USDT:USDT"
        assert book["market_type"] == "swap"
        assert [connection["connection_id"] for connection in book["connections"]] == [
            "sim-first",
            "sim-second",
        ]
