"""US3 (spec 013) — DB market_type column on portfolios + decision_commits.

Validates:
- ``Portfolio.market_type`` is derived from pair via ``Pair.parse(pair).market_type``
- ``DecisionCommitRow.market_type`` is populated by the journal serializer
- Legacy rows (no column) get the default 'spot' via the migration ALTER TABLE
- Bad / unparseable pair strings fall back to 'spot' (matches column DEFAULT)
"""

from __future__ import annotations

import pytest

from cryptotrader.journal.store import _market_type_for as _market_type_for_journal
from cryptotrader.portfolio.manager import _market_type_for as _market_type_for_pm


class TestMarketTypeDerivation:
    @pytest.mark.parametrize(
        ("pair", "expected"),
        [
            ("BTC/USDT", "spot"),
            ("ETH/USDT", "spot"),
            ("BTC/USDT:USDT", "swap"),
            ("ETH/USDT:USDT", "swap"),
            ("BTC/USD:BTC", "swap"),
        ],
    )
    def test_canonical_pair_to_market_type(self, pair: str, expected: str) -> None:
        assert _market_type_for_pm(pair) == expected
        assert _market_type_for_journal(pair) == expected

    @pytest.mark.parametrize("bad", ["BTCUSDT", "", "/", "BTC/", "/USDT"])
    def test_bad_pair_falls_back_to_spot(self, bad: str) -> None:
        assert _market_type_for_pm(bad) == "spot"
        assert _market_type_for_journal(bad) == "spot"


class TestPortfolioModelHasMarketType:
    def test_portfolio_orm_has_market_type_column(self) -> None:
        from cryptotrader.portfolio.manager import _pm_models

        _, Portfolio, _, _ = _pm_models()  # noqa: N806 — ORM class
        cols = {c.name for c in Portfolio.__table__.columns}
        assert "market_type" in cols

    def test_portfolio_market_type_default_is_spot(self) -> None:
        from cryptotrader.portfolio.manager import _pm_models

        _, Portfolio, _, _ = _pm_models()  # noqa: N806 — ORM class
        col = Portfolio.__table__.columns["market_type"]
        assert col.default is not None
        assert col.default.arg == "spot"
        assert col.nullable is False


class TestDecisionCommitRowHasMarketType:
    def test_decision_commit_orm_has_market_type_column(self) -> None:
        from cryptotrader.journal.store import _sa_models

        _, DecisionCommitRow = _sa_models()  # noqa: N806 — ORM class
        cols = {c.name for c in DecisionCommitRow.__table__.columns}
        assert "market_type" in cols

    def test_decision_commit_market_type_in_observability_columns(self) -> None:
        """The migration list must include market_type so existing DBs get backfilled."""
        from cryptotrader.journal.store import _OBSERVABILITY_COLUMNS

        col_names = [c[0] for c in _OBSERVABILITY_COLUMNS]
        assert "market_type" in col_names
        # Default 'spot' so old rows survive the NOT NULL constraint
        spec = next(c for c in _OBSERVABILITY_COLUMNS if c[0] == "market_type")
        assert "spot" in spec[2]
        assert "NOT NULL" in spec[2]


class TestJournalDcToRowDictWritesMarketType:
    def test_serialize_perp_pair_records_swap(self) -> None:
        from datetime import datetime

        from cryptotrader._compat import UTC
        from cryptotrader.journal.store import JournalStore
        from cryptotrader.models import DecisionCommit

        store = JournalStore(database_url=None)
        dc = DecisionCommit(
            hash="abc123",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT:USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
        )
        row = store._dc_to_row_dict(dc)
        assert row["market_type"] == "swap"
        assert row["pair"] == "BTC/USDT:USDT"

    def test_serialize_spot_pair_records_spot(self) -> None:
        from datetime import datetime

        from cryptotrader._compat import UTC
        from cryptotrader.journal.store import JournalStore
        from cryptotrader.models import DecisionCommit

        store = JournalStore(database_url=None)
        dc = DecisionCommit(
            hash="def456",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="ETH/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
        )
        row = store._dc_to_row_dict(dc)
        assert row["market_type"] == "spot"
