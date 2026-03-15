"""DB migration integration test for dashboard-observability task 11.4.

Verifies that:
1. Old rows (NULL new columns) can be read back via _row_to_dc() and return
   safe defaults for all 5 new fields — unit test with mock row objects.
2. The in-memory fallback store (used in tests and as DB fallback) correctly
   handles old-style commits that were created without the new fields.
3. New records written and read back contain correct values for all 5 fields.

Note on SQLite vs PostgreSQL:
- Production uses PostgreSQL; _ensure_tables() issues ALTER TABLE ADD COLUMN
  IF NOT EXISTS for pre-existing tables. This requires a running PostgreSQL
  instance which is not available in unit/integration test environments.
- SQLite cannot compile the JSONB column type used by _sa_models(). The DB
  layer is therefore only exercised via the in-memory fallback path in tests.
- The _row_to_dc() null-safety is tested via mock row objects that simulate
  the state of a PostgreSQL row where new columns are NULL (old record).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cryptotrader.journal.store import JournalStore
from cryptotrader.models import (
    ConsensusMetrics,
    DecisionCommit,
    GateResult,
    NodeTraceEntry,
    TradeVerdict,
)

# ---------------------------------------------------------------------------
# Task 11.4.2 + 11.4.3: _row_to_dc() null-safety unit test with mock rows
# ---------------------------------------------------------------------------


class _OldStyleRow:
    """Simulate a PostgreSQL row from before the observability migration.

    All five new columns are NULL (as they would be for rows inserted before
    the ALTER TABLE migration ran).
    """

    hash = "legacy_row_1"
    parent_hash = None
    timestamp = datetime.now(UTC)
    pair = "BTC/USDT"
    snapshot_summary = {}
    analyses = {}
    debate_rounds = 0
    challenges = []
    divergence = 0.0
    verdict = None
    risk_gate = None
    order_data = None
    fill_price = None
    slippage = None
    portfolio_after = {}
    pnl = None
    retrospective = None
    trace_id = None
    # New columns — NULL for old rows
    consensus_metrics = None
    verdict_source = None
    experience_memory = None
    node_trace = None
    debate_skip_reason = None


class TestRowToDcNullSafety:
    """_row_to_dc() must handle NULL / missing columns without raising exceptions.

    These tests simulate old PostgreSQL rows (before the observability migration)
    where all 5 new columns have NULL values.
    """

    def _make_store(self) -> JournalStore:
        return JournalStore(None)

    def _make_old_row(self, **overrides) -> _OldStyleRow:
        """Return an old-style row, optionally overriding fields."""
        row = _OldStyleRow()
        for k, v in overrides.items():
            setattr(row, k, v)
        return row

    def test_row_to_dc_handles_null_consensus_metrics(self) -> None:
        """_row_to_dc() returns consensus_metrics=None when column is NULL."""
        store = self._make_store()
        dc = store._row_to_dc(self._make_old_row())
        assert dc.consensus_metrics is None

    def test_row_to_dc_handles_null_verdict_source(self) -> None:
        """_row_to_dc() falls back to 'ai' when verdict_source column is NULL."""
        store = self._make_store()
        dc = store._row_to_dc(self._make_old_row(verdict_source=None))
        assert dc.verdict_source == "ai"

    def test_row_to_dc_handles_null_experience_memory(self) -> None:
        """_row_to_dc() returns {} when experience_memory column is NULL."""
        store = self._make_store()
        dc = store._row_to_dc(self._make_old_row(experience_memory=None))
        assert dc.experience_memory == {}

    def test_row_to_dc_handles_null_node_trace(self) -> None:
        """_row_to_dc() returns [] when node_trace column is NULL."""
        store = self._make_store()
        dc = store._row_to_dc(self._make_old_row(node_trace=None))
        assert dc.node_trace == []

    def test_row_to_dc_handles_null_debate_skip_reason(self) -> None:
        """_row_to_dc() returns '' when debate_skip_reason column is NULL."""
        store = self._make_store()
        dc = store._row_to_dc(self._make_old_row(debate_skip_reason=None))
        assert dc.debate_skip_reason == ""

    def test_row_to_dc_handles_all_null_new_fields_without_raising(self) -> None:
        """_row_to_dc() must not raise when all 5 new columns are NULL."""
        store = self._make_store()
        row = self._make_old_row()
        dc = store._row_to_dc(row)
        # Basic fields still correct
        assert dc.hash == "legacy_row_1"
        assert dc.pair == "BTC/USDT"
        # All new fields use safe defaults
        assert dc.consensus_metrics is None
        assert dc.verdict_source == "ai"
        assert dc.experience_memory == {}
        assert dc.node_trace == []
        assert dc.debate_skip_reason == ""

    def test_row_to_dc_handles_missing_attribute_via_getattr(self) -> None:
        """_row_to_dc() uses getattr() with None default so completely missing
        attributes (column not in SELECT) are handled safely."""

        class _BareRow:
            """Row object that does NOT have the new attributes at all."""

            hash = "bare_hash"
            parent_hash = None
            timestamp = datetime.now(UTC)
            pair = "ETH/USDT"
            snapshot_summary = None
            analyses = None
            debate_rounds = 0
            challenges = None
            divergence = None
            verdict = None
            risk_gate = None
            order_data = None
            fill_price = None
            slippage = None
            portfolio_after = None
            pnl = None
            retrospective = None
            # New columns NOT present at all (simulates truly missing attribute)

        store = self._make_store()
        dc = store._row_to_dc(_BareRow())
        # Should return safe defaults for missing attributes
        assert dc.consensus_metrics is None
        assert dc.verdict_source == "ai"
        assert dc.experience_memory == {}
        assert dc.node_trace == []
        assert dc.debate_skip_reason == ""

    def test_row_to_dc_correctly_decodes_non_null_consensus_metrics(self) -> None:
        """_row_to_dc() correctly decodes a non-NULL consensus_metrics dict."""
        store = self._make_store()
        cm_dict = {
            "strength": 0.75,
            "mean_score": 0.65,
            "dispersion": 0.10,
            "skip_threshold": 0.50,
            "confusion_threshold": 0.05,
        }
        row = self._make_old_row(consensus_metrics=cm_dict, verdict_source="weighted")
        dc = store._row_to_dc(row)
        assert dc.consensus_metrics is not None
        assert dc.consensus_metrics.strength == pytest.approx(0.75)
        assert dc.consensus_metrics.mean_score == pytest.approx(0.65)
        assert dc.verdict_source == "weighted"

    def test_row_to_dc_correctly_decodes_node_trace_list(self) -> None:
        """_row_to_dc() correctly decodes a non-NULL node_trace JSON list."""
        store = self._make_store()
        trace_data = [
            {"node": "data_collect", "duration_ms": 210, "summary": "fetched"},
            {"node": "debate_gate", "duration_ms": 15, "summary": "skipped"},
        ]
        row = self._make_old_row(node_trace=trace_data)
        dc = store._row_to_dc(row)
        assert len(dc.node_trace) == 2
        assert dc.node_trace[0].node == "data_collect"
        assert dc.node_trace[0].duration_ms == 210
        assert dc.node_trace[1].node == "debate_gate"


# ---------------------------------------------------------------------------
# Task 11.4.3: In-memory store full roundtrip with new fields
# ---------------------------------------------------------------------------


class TestInMemoryStoreNewFieldsRoundtrip:
    """In-memory JournalStore must persist and retrieve all 5 new fields correctly.

    This tests the _serialize() / _deserialize() path used by the in-memory
    fallback, ensuring it is consistent with the DB (_dc_to_row_dict / _row_to_dc)
    path behavior.
    """

    @pytest.fixture
    def store(self) -> JournalStore:
        return JournalStore(None)

    @pytest.mark.asyncio
    async def test_full_observability_roundtrip_in_memory(self, store: JournalStore) -> None:
        """All 5 new fields survive a full in-memory store commit/show roundtrip."""
        cm = ConsensusMetrics(
            strength=0.65,
            mean_score=0.55,
            dispersion=0.14,
            skip_threshold=0.50,
            confusion_threshold=0.05,
        )
        commit = DecisionCommit(
            hash="mem_full_01",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={"price": 60000.0},
            analyses={},
            debate_rounds=1,
            verdict=TradeVerdict(action="long", confidence=0.80, position_scale=0.70),
            risk_gate=GateResult(passed=True),
            consensus_metrics=cm,
            verdict_source="ai",
            experience_memory={"success_patterns": ["trend_follow"], "forbidden_zones": []},
            node_trace=[
                NodeTraceEntry(node="debate_gate", duration_ms=35, summary="consensus skip"),
                NodeTraceEntry(node="make_verdict", duration_ms=1200, summary="ai verdict"),
            ],
            debate_skip_reason="consensus",
        )

        await store.commit(commit)
        dc = await store.show("mem_full_01")

        assert dc is not None
        assert dc.consensus_metrics is not None
        assert dc.consensus_metrics.strength == pytest.approx(0.65)
        assert dc.consensus_metrics.mean_score == pytest.approx(0.55)
        assert dc.consensus_metrics.skip_threshold == pytest.approx(0.50)
        assert dc.verdict_source == "ai"
        assert dc.experience_memory == {"success_patterns": ["trend_follow"], "forbidden_zones": []}
        assert len(dc.node_trace) == 2
        assert dc.node_trace[0].node == "debate_gate"
        assert dc.node_trace[0].duration_ms == 35
        assert dc.node_trace[1].node == "make_verdict"
        assert dc.debate_skip_reason == "consensus"

    @pytest.mark.asyncio
    async def test_old_style_commit_returns_safe_defaults(self, store: JournalStore) -> None:
        """A commit constructed without new fields returns safe defaults after roundtrip."""
        old_commit = DecisionCommit(
            hash="old_style_mem",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="ETH/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=1,
        )
        await store.commit(old_commit)
        dc = await store.show("old_style_mem")

        assert dc is not None
        assert dc.consensus_metrics is None
        assert dc.verdict_source == "ai"
        assert dc.experience_memory == {}
        assert dc.node_trace == []
        assert dc.debate_skip_reason == ""

    @pytest.mark.asyncio
    async def test_weighted_verdict_source_persists_in_memory(self, store: JournalStore) -> None:
        """verdict_source='weighted' is correctly persisted in memory store."""
        commit = DecisionCommit(
            hash="mem_weighted",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            verdict_source="weighted",
        )
        await store.commit(commit)
        dc = await store.show("mem_weighted")

        assert dc is not None
        assert dc.verdict_source == "weighted"

    @pytest.mark.asyncio
    async def test_hold_all_mock_verdict_source_persists_in_memory(self, store: JournalStore) -> None:
        """verdict_source='hold_all_mock' is correctly persisted in memory store."""
        commit = DecisionCommit(
            hash="mem_mock",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="SOL/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            verdict_source="hold_all_mock",
        )
        await store.commit(commit)
        dc = await store.show("mem_mock")

        assert dc is not None
        assert dc.verdict_source == "hold_all_mock"

    @pytest.mark.asyncio
    async def test_log_includes_new_observability_fields(self, store: JournalStore) -> None:
        """JournalStore.log() returns commits with all 5 new observability fields."""
        commit = DecisionCommit(
            hash="log_obs_mem",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            consensus_metrics=ConsensusMetrics(
                strength=0.9, mean_score=0.8, dispersion=0.05, skip_threshold=0.5, confusion_threshold=0.05
            ),
            verdict_source="weighted",
            debate_skip_reason="consensus",
        )
        await store.commit(commit)

        logs = await store.log(limit=5)
        assert len(logs) == 1
        assert logs[0].verdict_source == "weighted"
        assert logs[0].debate_skip_reason == "consensus"
        assert logs[0].consensus_metrics is not None
        assert logs[0].consensus_metrics.strength == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_null_consensus_metrics_persisted_correctly(self, store: JournalStore) -> None:
        """consensus_metrics=None is stored and read back as None (not empty dict)."""
        commit = DecisionCommit(
            hash="null_cm_mem",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            consensus_metrics=None,
        )
        await store.commit(commit)
        dc = await store.show("null_cm_mem")

        assert dc is not None
        assert dc.consensus_metrics is None

    @pytest.mark.asyncio
    async def test_empty_experience_memory_persists_as_empty_dict(self, store: JournalStore) -> None:
        """experience_memory={} is stored and read back as {} (not None)."""
        commit = DecisionCommit(
            hash="empty_mem_01",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            experience_memory={},
        )
        await store.commit(commit)
        dc = await store.show("empty_mem_01")

        assert dc is not None
        assert dc.experience_memory == {}

    @pytest.mark.asyncio
    async def test_empty_node_trace_persists_as_empty_list(self, store: JournalStore) -> None:
        """node_trace=[] is stored and read back as [] (not None)."""
        commit = DecisionCommit(
            hash="empty_trace_01",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            node_trace=[],
        )
        await store.commit(commit)
        dc = await store.show("empty_trace_01")

        assert dc is not None
        assert dc.node_trace == []
