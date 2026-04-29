"""Tests for journal-related helpers exposed by api/nodes layer."""

from __future__ import annotations

from cryptotrader.models import NodeTraceEntry
from cryptotrader.nodes.journal import _aggregate_latency, _to_agent_analyses


class TestAggregateLatency:
    def test_empty_trace_yields_zero_total(self) -> None:
        out = _aggregate_latency([])
        assert out["total"] == 0.0
        assert out["data"] == 0.0
        assert out["agents"] == 0.0

    def test_dict_entries_collapse_into_buckets(self) -> None:
        trace = [
            {"node": "collect_data", "duration_ms": 100},
            {"node": "tech_agent", "duration_ms": 200},
            {"node": "chain_agent", "duration_ms": 180},
            {"node": "debate_round", "duration_ms": 50},
            {"node": "verdict", "duration_ms": 120},
            {"node": "risk_gate", "duration_ms": 10},
            {"node": "execute_trade", "duration_ms": 80},
        ]
        out = _aggregate_latency(trace)
        assert out["data"] == 100
        assert out["agents"] == 380
        assert out["debate"] == 50
        assert out["verdict"] == 120
        assert out["risk"] == 10
        assert out["execute"] == 80
        assert out["total"] == 740.0

    def test_unknown_nodes_bucket_into_other(self) -> None:
        trace = [
            {"node": "custom_foo_node", "duration_ms": 15},
            {"node": "bar_unknown", "duration_ms": 25},
        ]
        out = _aggregate_latency(trace)
        assert out["other"] == 40
        assert out["total"] == 40.0

    def test_node_trace_entry_objects_supported(self) -> None:
        trace = [
            NodeTraceEntry(node="tech_agent", duration_ms=150, summary=""),
            NodeTraceEntry(node="news_agent", duration_ms=90, summary=""),
        ]
        out = _aggregate_latency(trace)
        assert out["agents"] == 240

    def test_missing_duration_defaults_zero(self) -> None:
        trace = [{"node": "tech_agent"}, {"node": "risk_gate", "duration_ms": None}]
        out = _aggregate_latency(trace)
        assert out["agents"] == 0
        assert out["risk"] == 0


class TestToAgentAnalysesPreservesIsMock:
    """Regression: nodes/journal._to_agent_analyses dropped is_mock when
    re-hydrating raw analysis dicts back into AgentAnalysis dataclasses,
    making every fallback decision look real in the journal (observed in
    the 22:38 cycle on 2026-04-29)."""

    def test_is_mock_true_round_trips(self) -> None:
        raw = {
            "chain_agent": {
                "direction": "neutral",
                "confidence": 0.1,
                "reasoning": "LLM unavailable - mock analysis",
                "is_mock": True,
                "data_sufficiency": "low",
            }
        }
        out = _to_agent_analyses(raw, pair="BTC/USDT")
        assert out["chain_agent"].is_mock is True
        assert out["chain_agent"].reasoning == "LLM unavailable - mock analysis"

    def test_is_mock_false_round_trips(self) -> None:
        raw = {
            "tech_agent": {
                "direction": "bearish",
                "confidence": 0.6,
                "reasoning": "RSI < 30",
                "is_mock": False,
                "data_sufficiency": "high",
            }
        }
        out = _to_agent_analyses(raw, pair="BTC/USDT")
        assert out["tech_agent"].is_mock is False

    def test_missing_is_mock_defaults_false(self) -> None:
        raw = {"x": {"direction": "bullish", "confidence": 0.7, "reasoning": "ok"}}
        out = _to_agent_analyses(raw, pair="ETH/USDT")
        assert out["x"].is_mock is False
