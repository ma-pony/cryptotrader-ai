"""US4 (spec 013, T034) — API responses include pair_display + market_type.

Validates the response shape for both /api/portfolio/snapshot and the
decisions endpoints. Uses unit-level assertions on the helper functions
plus minimal Pydantic round-trips so the test runs without a live DB.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from cryptotrader._compat import UTC


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


class TestDecisionsPairMeta:
    @pytest.mark.parametrize(
        ("pair", "expected_display", "expected_mt"),
        [
            ("BTC/USDT", "BTC/USDT", "spot"),
            ("BTC/USDT:USDT", "BTC/USDT (perp)", "swap"),
            ("BTC/USD:BTC", "BTC/USD (perp)", "swap"),
        ],
    )
    def test_pair_meta_helper(self, pair: str, expected_display: str, expected_mt: str) -> None:
        from api.routes.decisions import _pair_meta

        assert _pair_meta(pair) == (expected_display, expected_mt)

    def test_pair_meta_falls_back_for_malformed(self) -> None:
        from api.routes.decisions import _pair_meta

        assert _pair_meta("not-a-pair") == ("not-a-pair", "spot")

    def test_decision_list_item_includes_pair_meta(self) -> None:
        from api.routes.decisions import _commit_to_list_item

        commit = MagicMock()
        commit.hash = "abc123"
        commit.timestamp = datetime.now(UTC)
        commit.pair = "BTC/USDT:USDT"
        commit.snapshot_summary = {"price": 84500.0}
        commit.verdict = None
        commit.fill_price = None
        commit.order = None
        commit.trace_id = None
        commit.pnl = None
        commit.risk_gate = None
        commit.debate_skip_reason = ""
        commit.debate_rounds = 0

        item = _commit_to_list_item(commit)
        assert item.pair == "BTC/USDT:USDT"
        assert item.pair_display == "BTC/USDT (perp)"
        assert item.market_type == "swap"
