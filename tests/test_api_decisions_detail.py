"""Tests for GET /api/decisions/{commit_hash} — FR-804.

Returns the full DecisionCommit shape (agent_analyses / debate_rounds /
risk_gate / execution / node_timeline per data-model §2).
404 when commit_hash unknown.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cryptotrader._compat import UTC


@pytest.fixture
def client() -> TestClient:
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


def _mock_config() -> MagicMock:
    cfg = MagicMock()
    cfg.infrastructure.database_url = None
    return cfg


def _full_commit(commit_hash: str = "a1b2c3d"):
    """Build a richly populated DecisionCommit covering all detail-view fields."""
    from cryptotrader.models import (
        AgentAnalysis,
        ConsensusMetrics,
        DecisionCommit,
        GateResult,
        NodeTraceEntry,
        Order,
        OrderStatus,
        TradeVerdict,
    )

    analyses = {
        "NewsAgent": AgentAnalysis(
            agent_id="news",
            pair="BTC/USDT",
            direction="bullish",
            confidence=0.7,
            reasoning="positive news flow",
        ),
        "MacroAgent": AgentAnalysis(
            agent_id="macro",
            pair="BTC/USDT",
            direction="neutral",
            confidence=0.5,
            reasoning="neutral macro",
        ),
        "SentimentAgent": AgentAnalysis(
            agent_id="sentiment",
            pair="BTC/USDT",
            direction="bullish",
            confidence=0.6,
            reasoning="positive twitter",
        ),
        "TechnicalAgent": AgentAnalysis(
            agent_id="technical",
            pair="BTC/USDT",
            direction="bullish",
            confidence=0.8,
            reasoning="MA cross",
        ),
    }

    return DecisionCommit(
        hash=commit_hash,
        parent_hash=None,
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        snapshot_summary={"price": 65000.0},
        analyses=analyses,
        debate_rounds=2,
        challenges=[
            {"round": 1, "bull": "bull says X", "bear": "bear says Y"},
            {"round": 2, "bull": "bull rebut", "bear": "bear rebut"},
        ],
        verdict=TradeVerdict(
            action="long",
            confidence=0.72,
            position_scale=0.5,
            reasoning="bullish consensus",
            thesis="momentum + macro",
            invalidation="below 64000",
            verdict_source="ai",
        ),
        risk_gate=GateResult(passed=True),
        order=Order(
            pair="BTC/USDT",
            side="buy",
            amount=0.05,
            price=65000.0,
            status=OrderStatus.FILLED,
        ),
        fill_price=65010.0,
        slippage=0.0001,
        trace_id="trace-xyz",
        consensus_metrics=ConsensusMetrics(
            strength=0.55,
            mean_score=0.5,
            dispersion=0.1,
            skip_threshold=0.5,
            confusion_threshold=0.05,
        ),
        verdict_source="ai",
        node_trace=[
            NodeTraceEntry(node="tag_regime_node", duration_ms=120, summary="ok"),
            NodeTraceEntry(node="agents_parallel", duration_ms=850, summary="4 agents"),
            NodeTraceEntry(node="debate_gate", duration_ms=20, summary="proceed"),
            NodeTraceEntry(node="verdict", duration_ms=300, summary="long 0.5"),
            NodeTraceEntry(node="risk_gate", duration_ms=15, summary="passed"),
            NodeTraceEntry(node="execute", duration_ms=210, summary="filled"),
        ],
    )


class TestDecisionsDetailShape:
    def test_returns_200_with_full_commit_shape(self, client: TestClient) -> None:
        commit = _full_commit("a1b2c3d")
        mock_store = MagicMock()
        mock_store.show = AsyncMock(return_value=commit)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            resp = client.get("/api/decisions/a1b2c3d")

        assert resp.status_code == 200
        body = resp.json()
        for key in (
            "commit_hash",
            "ts",
            "pair",
            "price",
            "agent_analyses",
            "debate_rounds",
            "verdict",
            "risk_gate",
            "execution",
            "node_timeline",
            "trace_id",
        ):
            assert key in body, f"missing detail key: {key}"

    def test_agent_analyses_array_with_required_fields(self, client: TestClient) -> None:
        commit = _full_commit()
        mock_store = MagicMock()
        mock_store.show = AsyncMock(return_value=commit)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            body = client.get("/api/decisions/a1b2c3d").json()

        analyses = body["agent_analyses"]
        assert len(analyses) == 4
        names = {a["name"] for a in analyses}
        assert names == {"NewsAgent", "MacroAgent", "SentimentAgent", "TechnicalAgent"}
        for a in analyses:
            for key in ("name", "score", "confidence", "reasoning", "is_mock"):
                assert key in a

    def test_debate_rounds_serialized_when_present(self, client: TestClient) -> None:
        commit = _full_commit()
        mock_store = MagicMock()
        mock_store.show = AsyncMock(return_value=commit)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            body = client.get("/api/decisions/a1b2c3d").json()

        rounds = body["debate_rounds"]
        assert isinstance(rounds, list)
        assert len(rounds) == 2
        for r in rounds:
            for key in ("round", "bull_message", "bear_message"):
                assert key in r

    def test_risk_gate_includes_check_list(self, client: TestClient) -> None:
        commit = _full_commit()
        mock_store = MagicMock()
        mock_store.show = AsyncMock(return_value=commit)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            body = client.get("/api/decisions/a1b2c3d").json()

        gate = body["risk_gate"]
        assert gate["passed"] is True
        assert isinstance(gate["checks"], list)

    def test_node_timeline_in_chronological_shape(self, client: TestClient) -> None:
        commit = _full_commit()
        mock_store = MagicMock()
        mock_store.show = AsyncMock(return_value=commit)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            body = client.get("/api/decisions/a1b2c3d").json()

        timeline = body["node_timeline"]
        assert len(timeline) == 6
        for entry in timeline:
            for key in ("node", "start_ms", "duration_ms"):
                assert key in entry
        # start_ms must be non-decreasing (cumulative from durations)
        starts = [e["start_ms"] for e in timeline]
        assert starts == sorted(starts)

    def test_execution_present_when_filled(self, client: TestClient) -> None:
        commit = _full_commit()
        mock_store = MagicMock()
        mock_store.show = AsyncMock(return_value=commit)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            body = client.get("/api/decisions/a1b2c3d").json()

        ex = body["execution"]
        assert ex is not None
        for key in ("order_id", "status", "fill_price", "fill_size"):
            assert key in ex


class TestDecisionsDetailErrors:
    def test_404_when_commit_unknown(self, client: TestClient) -> None:
        mock_store = MagicMock()
        mock_store.show = AsyncMock(return_value=None)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            resp = client.get("/api/decisions/deadbeef")

        assert resp.status_code == 404
        assert "detail" in resp.json()
