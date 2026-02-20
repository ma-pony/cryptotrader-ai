"""Smoke tests for LangGraph construction."""

from cryptotrader.graph import build_trading_graph, ArenaState, merge_dicts


def test_merge_dicts():
    a = {"x": 1, "nested": {"a": 1}}
    b = {"y": 2, "nested": {"b": 2}}
    result = merge_dicts(a, b)
    assert result["x"] == 1
    assert result["y"] == 2
    assert result["nested"]["a"] == 1
    assert result["nested"]["b"] == 2


def test_build_graph_compiles():
    graph = build_trading_graph()
    assert graph is not None


def test_graph_has_nodes():
    graph = build_trading_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "collect_data", "inject_experience",
        "tech_agent", "chain_agent", "news_agent", "macro_agent",
        "cross_challenge", "check_convergence",
        "verdict", "risk_gate", "execute", "record_rejection",
    }
    assert expected.issubset(node_names)
