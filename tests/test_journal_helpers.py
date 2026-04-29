"""Tests for journal-related helpers exposed by api/nodes layer."""

from __future__ import annotations

from cryptotrader.models import NodeTraceEntry
from cryptotrader.nodes.journal import _aggregate_latency


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
