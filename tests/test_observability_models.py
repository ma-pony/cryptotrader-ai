"""TDD tests for dashboard-observability tasks 1.1, 1.2, 1.3.

Tests are written first (RED phase). They verify:
- Task 1.1: ConsensusMetrics and NodeTraceEntry value objects
- Task 1.2: Five new observability fields on DecisionCommit
- Task 1.3: DB storage layer serialization / deserialization for new fields
"""

from __future__ import annotations

import json
from dataclasses import asdict
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

# ── Task 1.1: ConsensusMetrics value object ──────────────────────────────────


class TestConsensusMetrics:
    """ConsensusMetrics is a dataclass holding debate gate snapshot data."""

    def test_required_fields_exist(self) -> None:
        cm = ConsensusMetrics(
            strength=0.72,
            mean_score=0.6,
            dispersion=0.12,
            skip_threshold=0.5,
            confusion_threshold=0.05,
        )
        assert cm.strength == pytest.approx(0.72)
        assert cm.mean_score == pytest.approx(0.6)
        assert cm.dispersion == pytest.approx(0.12)
        assert cm.skip_threshold == pytest.approx(0.5)
        assert cm.confusion_threshold == pytest.approx(0.05)

    def test_is_dataclass_serializable_with_asdict(self) -> None:
        cm = ConsensusMetrics(
            strength=0.5,
            mean_score=0.4,
            dispersion=0.1,
            skip_threshold=0.5,
            confusion_threshold=0.05,
        )
        d = asdict(cm)
        assert set(d.keys()) == {"strength", "mean_score", "dispersion", "skip_threshold", "confusion_threshold"}

    def test_roundtrip_json(self) -> None:
        cm = ConsensusMetrics(
            strength=0.8,
            mean_score=0.7,
            dispersion=0.05,
            skip_threshold=0.5,
            confusion_threshold=0.05,
        )
        serialized = json.dumps(asdict(cm))
        restored = json.loads(serialized)
        cm2 = ConsensusMetrics(**restored)
        assert cm2.strength == pytest.approx(cm.strength)
        assert cm2.mean_score == pytest.approx(cm.mean_score)


# ── Task 1.1: NodeTraceEntry value object ─────────────────────────────────────


class TestNodeTraceEntry:
    """NodeTraceEntry records a single node execution in the pipeline."""

    def test_required_fields_exist(self) -> None:
        entry = NodeTraceEntry(node="debate_gate", duration_ms=120, summary="skipped due to consensus")
        assert entry.node == "debate_gate"
        assert entry.duration_ms == 120
        assert entry.summary == "skipped due to consensus"

    def test_is_dataclass_serializable_with_asdict(self) -> None:
        entry = NodeTraceEntry(node="make_verdict", duration_ms=450, summary="AI verdict: long")
        d = asdict(entry)
        assert set(d.keys()) == {"node", "duration_ms", "summary"}

    def test_roundtrip_json(self) -> None:
        entry = NodeTraceEntry(node="risk_gate", duration_ms=30, summary="passed all checks")
        serialized = json.dumps(asdict(entry))
        restored = json.loads(serialized)
        entry2 = NodeTraceEntry(**restored)
        assert entry2.node == entry.node
        assert entry2.duration_ms == entry.duration_ms
        assert entry2.summary == entry.summary

    def test_summary_can_be_empty_string(self) -> None:
        entry = NodeTraceEntry(node="data_collect", duration_ms=200, summary="")
        assert entry.summary == ""


# ── Task 1.2: DecisionCommit new observability fields ────────────────────────


class TestDecisionCommitNewFields:
    """The five new observability fields must exist with correct defaults."""

    def _make_commit(self, **kwargs) -> DecisionCommit:
        defaults = {
            "hash": "abc12345",
            "parent_hash": None,
            "timestamp": datetime.now(UTC),
            "pair": "BTC/USDT",
            "snapshot_summary": {},
            "analyses": {},
            "debate_rounds": 0,
        }
        defaults.update(kwargs)
        return DecisionCommit(**defaults)

    def test_consensus_metrics_defaults_to_none(self) -> None:
        dc = self._make_commit()
        assert dc.consensus_metrics is None

    def test_verdict_source_defaults_to_ai(self) -> None:
        dc = self._make_commit()
        assert dc.verdict_source == "ai"

    def test_experience_memory_defaults_to_empty_dict(self) -> None:
        dc = self._make_commit()
        assert dc.experience_memory == {}

    def test_node_trace_defaults_to_empty_list(self) -> None:
        dc = self._make_commit()
        assert dc.node_trace == []

    def test_debate_skip_reason_defaults_to_empty_string(self) -> None:
        dc = self._make_commit()
        assert dc.debate_skip_reason == ""

    def test_consensus_metrics_field_accepts_consensus_metrics_instance(self) -> None:
        cm = ConsensusMetrics(
            strength=0.7,
            mean_score=0.6,
            dispersion=0.1,
            skip_threshold=0.5,
            confusion_threshold=0.05,
        )
        dc = self._make_commit(consensus_metrics=cm)
        assert dc.consensus_metrics is cm
        assert dc.consensus_metrics.strength == pytest.approx(0.7)

    def test_verdict_source_accepts_weighted(self) -> None:
        dc = self._make_commit(verdict_source="weighted")
        assert dc.verdict_source == "weighted"

    def test_verdict_source_accepts_hold_all_mock(self) -> None:
        dc = self._make_commit(verdict_source="hold_all_mock")
        assert dc.verdict_source == "hold_all_mock"

    def test_experience_memory_stores_dict(self) -> None:
        mem = {"success_patterns": ["pattern1"], "forbidden_zones": []}
        dc = self._make_commit(experience_memory=mem)
        assert dc.experience_memory == mem

    def test_node_trace_stores_list_of_entries(self) -> None:
        entries = [
            NodeTraceEntry(node="data_collect", duration_ms=200, summary="ok"),
            NodeTraceEntry(node="debate_gate", duration_ms=50, summary="skipped"),
        ]
        dc = self._make_commit(node_trace=entries)
        assert len(dc.node_trace) == 2
        assert dc.node_trace[0].node == "data_collect"

    def test_debate_skip_reason_stores_string(self) -> None:
        dc = self._make_commit(debate_skip_reason="consensus")
        assert dc.debate_skip_reason == "consensus"

    def test_old_code_path_no_new_args_still_works(self) -> None:
        """Constructing DecisionCommit without any new fields must succeed."""
        dc = DecisionCommit(
            hash="legacy01",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="ETH/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=1,
        )
        assert dc.verdict_source == "ai"
        assert dc.consensus_metrics is None
        assert dc.node_trace == []

    def test_asdict_includes_new_fields(self) -> None:
        dc = self._make_commit()
        d = asdict(dc)
        assert "consensus_metrics" in d
        assert "verdict_source" in d
        assert "experience_memory" in d
        assert "node_trace" in d
        assert "debate_skip_reason" in d


# ── Task 1.3: DB storage serialization / deserialization ─────────────────────


class TestJournalStoreNewFields:
    """The in-memory JournalStore must round-trip all five new fields."""

    @pytest.fixture
    def store(self) -> JournalStore:
        return JournalStore(None)

    @pytest.fixture
    def full_commit(self) -> DecisionCommit:
        """DecisionCommit with all five new observability fields populated."""
        return DecisionCommit(
            hash="obs_test1",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={"price": 50000.0},
            analyses={},
            debate_rounds=0,
            verdict=TradeVerdict(action="long", confidence=0.75, position_scale=0.8),
            risk_gate=GateResult(passed=True),
            consensus_metrics=ConsensusMetrics(
                strength=0.72,
                mean_score=0.60,
                dispersion=0.12,
                skip_threshold=0.50,
                confusion_threshold=0.05,
            ),
            verdict_source="ai",
            experience_memory={"success_patterns": ["pattern_a"], "forbidden_zones": []},
            node_trace=[
                NodeTraceEntry(node="data_collect", duration_ms=210, summary="fetched ok"),
                NodeTraceEntry(node="debate_gate", duration_ms=15, summary="skipped"),
            ],
            debate_skip_reason="consensus",
        )

    @pytest.mark.asyncio
    async def test_consensus_metrics_roundtrip(self, store: JournalStore, full_commit: DecisionCommit) -> None:
        await store.commit(full_commit)
        dc = await store.show("obs_test1")
        assert dc is not None
        assert dc.consensus_metrics is not None
        assert dc.consensus_metrics.strength == pytest.approx(0.72)
        assert dc.consensus_metrics.mean_score == pytest.approx(0.60)
        assert dc.consensus_metrics.dispersion == pytest.approx(0.12)
        assert dc.consensus_metrics.skip_threshold == pytest.approx(0.50)
        assert dc.consensus_metrics.confusion_threshold == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_verdict_source_roundtrip(self, store: JournalStore, full_commit: DecisionCommit) -> None:
        await store.commit(full_commit)
        dc = await store.show("obs_test1")
        assert dc is not None
        assert dc.verdict_source == "ai"

    @pytest.mark.asyncio
    async def test_experience_memory_roundtrip(self, store: JournalStore, full_commit: DecisionCommit) -> None:
        await store.commit(full_commit)
        dc = await store.show("obs_test1")
        assert dc is not None
        assert dc.experience_memory == {"success_patterns": ["pattern_a"], "forbidden_zones": []}

    @pytest.mark.asyncio
    async def test_node_trace_roundtrip(self, store: JournalStore, full_commit: DecisionCommit) -> None:
        await store.commit(full_commit)
        dc = await store.show("obs_test1")
        assert dc is not None
        assert len(dc.node_trace) == 2
        assert dc.node_trace[0].node == "data_collect"
        assert dc.node_trace[0].duration_ms == 210
        assert dc.node_trace[1].node == "debate_gate"

    @pytest.mark.asyncio
    async def test_debate_skip_reason_roundtrip(self, store: JournalStore, full_commit: DecisionCommit) -> None:
        await store.commit(full_commit)
        dc = await store.show("obs_test1")
        assert dc is not None
        assert dc.debate_skip_reason == "consensus"

    @pytest.mark.asyncio
    async def test_old_record_none_fields_return_defaults(self, store: JournalStore) -> None:
        """A commit without new fields must return safe defaults after roundtrip."""
        old_commit = DecisionCommit(
            hash="legacy_01",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="ETH/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=1,
        )
        await store.commit(old_commit)
        dc = await store.show("legacy_01")
        assert dc is not None
        assert dc.consensus_metrics is None
        assert dc.verdict_source == "ai"
        assert dc.experience_memory == {}
        assert dc.node_trace == []
        assert dc.debate_skip_reason == ""

    @pytest.mark.asyncio
    async def test_weighted_verdict_source_roundtrip(self, store: JournalStore) -> None:
        commit = DecisionCommit(
            hash="weighted_01",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            verdict_source="weighted",
        )
        await store.commit(commit)
        dc = await store.show("weighted_01")
        assert dc is not None
        assert dc.verdict_source == "weighted"

    @pytest.mark.asyncio
    async def test_hold_all_mock_verdict_source_roundtrip(self, store: JournalStore) -> None:
        commit = DecisionCommit(
            hash="mock_01",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            verdict_source="hold_all_mock",
        )
        await store.commit(commit)
        dc = await store.show("mock_01")
        assert dc is not None
        assert dc.verdict_source == "hold_all_mock"

    @pytest.mark.asyncio
    async def test_null_consensus_metrics_roundtrip(self, store: JournalStore) -> None:
        """consensus_metrics=None must survive a roundtrip without errors."""
        commit = DecisionCommit(
            hash="null_cm_01",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            consensus_metrics=None,
        )
        await store.commit(commit)
        dc = await store.show("null_cm_01")
        assert dc is not None
        assert dc.consensus_metrics is None

    @pytest.mark.asyncio
    async def test_log_returns_commits_with_new_fields(self, store: JournalStore, full_commit: DecisionCommit) -> None:
        """log() must also include new observability fields."""
        await store.commit(full_commit)
        logs = await store.log(limit=5)
        assert len(logs) == 1
        assert logs[0].debate_skip_reason == "consensus"
        assert logs[0].verdict_source == "ai"


class TestJournalStoreDcToRowDict:
    """Unit tests for _dc_to_row_dict / _row_to_dc (in-memory path is tested via commit/show)."""

    def _make_store(self) -> JournalStore:
        return JournalStore(None)

    def test_dc_to_row_dict_includes_consensus_metrics(self) -> None:
        store = self._make_store()
        cm = ConsensusMetrics(
            strength=0.6, mean_score=0.5, dispersion=0.1, skip_threshold=0.5, confusion_threshold=0.05
        )
        dc = DecisionCommit(
            hash="row_dict_01",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            consensus_metrics=cm,
        )
        row = store._dc_to_row_dict(dc)
        assert "consensus_metrics" in row
        assert row["consensus_metrics"] is not None
        # Should be a dict (JSON-serializable)
        assert isinstance(row["consensus_metrics"], dict)
        assert row["consensus_metrics"]["strength"] == pytest.approx(0.6)

    def test_dc_to_row_dict_includes_verdict_source(self) -> None:
        store = self._make_store()
        dc = DecisionCommit(
            hash="row_dict_02",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            verdict_source="weighted",
        )
        row = store._dc_to_row_dict(dc)
        assert row["verdict_source"] == "weighted"

    def test_dc_to_row_dict_includes_node_trace(self) -> None:
        store = self._make_store()
        entries = [NodeTraceEntry(node="risk_gate", duration_ms=25, summary="passed")]
        dc = DecisionCommit(
            hash="row_dict_03",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            node_trace=entries,
        )
        row = store._dc_to_row_dict(dc)
        assert "node_trace" in row
        assert isinstance(row["node_trace"], list)
        assert row["node_trace"][0]["node"] == "risk_gate"

    def test_dc_to_row_dict_includes_debate_skip_reason(self) -> None:
        store = self._make_store()
        dc = DecisionCommit(
            hash="row_dict_04",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
            debate_skip_reason="confusion",
        )
        row = store._dc_to_row_dict(dc)
        assert row["debate_skip_reason"] == "confusion"

    def test_dc_to_row_dict_null_consensus_metrics(self) -> None:
        store = self._make_store()
        dc = DecisionCommit(
            hash="row_dict_05",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
        )
        row = store._dc_to_row_dict(dc)
        assert row["consensus_metrics"] is None
