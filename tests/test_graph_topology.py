"""Integration tests for LangGraph topology: node names and conditional edge routing.

Requirement 4.3 -- verify that build_trading_graph(), build_lite_graph(), and
build_debate_graph() expose the expected nodes and conditional edge mappings
without executing any LLM calls or external I/O.
"""

from __future__ import annotations

import pytest

from cryptotrader.graph import (
    build_debate_graph,
    build_lite_graph,
    build_trading_graph,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_names(graph) -> set[str]:
    """Return all node names (excluding __start__ / __end__ sentinels)."""
    return {n for n in graph.get_graph().nodes if not n.startswith("__")}


def _conditional_edges(graph) -> dict[tuple[str, str], str]:
    """Return mapping of (source, target) -> condition_label for conditional edges."""
    return {(e.source, e.target): e.data for e in graph.get_graph().edges if e.conditional}


def _unconditional_edges(graph) -> set[tuple[str, str]]:
    """Return set of (source, target) for non-conditional edges (excluding sentinels)."""
    return {
        (e.source, e.target)
        for e in graph.get_graph().edges
        if not e.conditional and not e.source.startswith("__") and not e.target.startswith("__")
    }


# ---------------------------------------------------------------------------
# build_trading_graph()
# ---------------------------------------------------------------------------


class TestBuildTradingGraph:
    """Topology tests for the full trading pipeline graph."""

    @pytest.fixture(scope="class")
    def graph(self):
        return build_trading_graph()

    def test_compiles(self, graph):
        assert graph is not None

    def test_required_nodes_present(self, graph):
        nodes = _node_names(graph)
        required = {
            "collect_data",
            "update_pnl",
            "stop_loss_check",
            "inject_experience",
            "tech_agent",
            "chain_agent",
            "news_agent",
            "macro_agent",
            "debate_gate",
            "debate_round_1",
            "debate_round_2",
            "enrich_context",
            "verdict",
            "risk_gate",
            "execute",
            "record_trade",
            "record_rejection",
        }
        assert required.issubset(nodes), f"Missing nodes: {required - nodes}"

    def test_debate_gate_skip_routes_to_enrich_context(self, graph):
        """debate_gate_router returning 'skip' must connect to enrich_context."""
        cond = _conditional_edges(graph)
        assert ("debate_gate", "enrich_context") in cond
        assert cond[("debate_gate", "enrich_context")] == "skip"

    def test_debate_gate_debate_routes_to_debate_round_1(self, graph):
        """debate_gate_router returning 'debate' must connect to debate_round_1."""
        cond = _conditional_edges(graph)
        assert ("debate_gate", "debate_round_1") in cond
        assert cond[("debate_gate", "debate_round_1")] == "debate"

    def test_risk_gate_approved_routes_to_execute(self, graph):
        """risk_router returning 'approved' must connect to execute."""
        cond = _conditional_edges(graph)
        assert ("risk_gate", "execute") in cond
        assert cond[("risk_gate", "execute")] == "approved"

    def test_risk_gate_rejected_routes_to_record_rejection(self, graph):
        """risk_router returning 'rejected' must connect to record_rejection."""
        cond = _conditional_edges(graph)
        assert ("risk_gate", "record_rejection") in cond
        assert cond[("risk_gate", "record_rejection")] == "rejected"

    def test_stop_loss_check_continue_routes_to_inject_experience(self, graph):
        cond = _conditional_edges(graph)
        assert ("stop_loss_check", "inject_experience") in cond
        assert cond[("stop_loss_check", "inject_experience")] == "continue"

    def test_stop_loss_check_exit_routes_to_risk_gate(self, graph):
        cond = _conditional_edges(graph)
        assert ("stop_loss_check", "risk_gate") in cond
        assert cond[("stop_loss_check", "risk_gate")] == "exit_position"

    def test_agents_fan_in_to_debate_gate(self, graph):
        """All four analysis agents must feed into debate_gate."""
        uncond = _unconditional_edges(graph)
        for agent in ("tech_agent", "chain_agent", "news_agent", "macro_agent"):
            assert (agent, "debate_gate") in uncond, f"{agent} -> debate_gate edge missing"

    def test_debate_rounds_are_sequential(self, graph):
        uncond = _unconditional_edges(graph)
        assert ("debate_round_1", "debate_round_2") in uncond
        assert ("debate_round_2", "enrich_context") in uncond

    def test_enrich_context_leads_to_verdict(self, graph):
        uncond = _unconditional_edges(graph)
        assert ("enrich_context", "verdict") in uncond

    def test_verdict_leads_to_risk_gate(self, graph):
        uncond = _unconditional_edges(graph)
        assert ("verdict", "risk_gate") in uncond

    def test_execute_leads_to_record_trade(self, graph):
        uncond = _unconditional_edges(graph)
        assert ("execute", "record_trade") in uncond


# ---------------------------------------------------------------------------
# build_lite_graph()
# ---------------------------------------------------------------------------


class TestBuildLiteGraph:
    """Topology tests for the lightweight graph (no debate, no risk gate)."""

    @pytest.fixture(scope="class")
    def graph(self):
        return build_lite_graph()

    def test_compiles(self, graph):
        assert graph is not None

    def test_required_nodes_present(self, graph):
        nodes = _node_names(graph)
        required = {
            "collect_data",
            "update_pnl",
            "inject_experience",
            "tech_agent",
            "chain_agent",
            "news_agent",
            "macro_agent",
            "enrich_context",
            "verdict",
        }
        assert required.issubset(nodes), f"Missing nodes: {required - nodes}"

    def test_no_debate_gate_node(self, graph):
        """Lite graph must NOT include the debate_gate node."""
        assert "debate_gate" not in _node_names(graph)

    def test_no_risk_gate_node(self, graph):
        """Lite graph must NOT include the risk_gate node."""
        assert "risk_gate" not in _node_names(graph)

    def test_no_conditional_edges(self, graph):
        """Lite graph has no conditional edges."""
        assert _conditional_edges(graph) == {}

    def test_agents_fan_in_to_enrich_context(self, graph):
        """In lite graph, agents feed directly into enrich_context (no debate)."""
        uncond = _unconditional_edges(graph)
        for agent in ("tech_agent", "chain_agent", "news_agent", "macro_agent"):
            assert (agent, "enrich_context") in uncond, f"{agent} -> enrich_context edge missing"

    def test_enrich_context_leads_to_verdict(self, graph):
        uncond = _unconditional_edges(graph)
        assert ("enrich_context", "verdict") in uncond

    def test_collect_data_leads_to_update_pnl(self, graph):
        uncond = _unconditional_edges(graph)
        assert ("collect_data", "update_pnl") in uncond

    def test_inject_experience_fans_out_to_agents(self, graph):
        uncond = _unconditional_edges(graph)
        for agent in ("tech_agent", "chain_agent", "news_agent", "macro_agent"):
            assert ("inject_experience", agent) in uncond


# ---------------------------------------------------------------------------
# build_debate_graph()
# ---------------------------------------------------------------------------


class TestBuildDebateGraph:
    """Topology tests for the bull/bear adversarial debate graph."""

    @pytest.fixture(scope="class")
    def graph(self):
        return build_debate_graph()

    def test_compiles(self, graph):
        assert graph is not None

    def test_required_nodes_present(self, graph):
        nodes = _node_names(graph)
        required = {
            "collect_data",
            "update_pnl",
            "inject_experience",
            "tech_agent",
            "chain_agent",
            "news_agent",
            "macro_agent",
            "debate",
            "enrich_context",
            "verdict",
        }
        assert required.issubset(nodes), f"Missing nodes: {required - nodes}"

    def test_uses_single_debate_node_not_rounds(self, graph):
        """Debate graph uses a single 'debate' node, not debate_round_1/2."""
        nodes = _node_names(graph)
        assert "debate" in nodes
        assert "debate_round_1" not in nodes
        assert "debate_round_2" not in nodes

    def test_no_debate_gate_node(self, graph):
        """Debate graph uses the older bull/bear debate, not the gate-based flow."""
        assert "debate_gate" not in _node_names(graph)

    def test_no_risk_gate_node(self, graph):
        """Debate graph has no risk_gate."""
        assert "risk_gate" not in _node_names(graph)

    def test_no_conditional_edges(self, graph):
        """Debate graph has no conditional edges."""
        assert _conditional_edges(graph) == {}

    def test_agents_fan_in_to_debate(self, graph):
        """All four agents must connect into the debate node."""
        uncond = _unconditional_edges(graph)
        for agent in ("tech_agent", "chain_agent", "news_agent", "macro_agent"):
            assert (agent, "debate") in uncond, f"{agent} -> debate edge missing"

    def test_debate_leads_to_enrich_context(self, graph):
        uncond = _unconditional_edges(graph)
        assert ("debate", "enrich_context") in uncond

    def test_enrich_context_leads_to_verdict(self, graph):
        uncond = _unconditional_edges(graph)
        assert ("enrich_context", "verdict") in uncond


# ---------------------------------------------------------------------------
# debate_gate_router semantics (unit-level routing verification)
# ---------------------------------------------------------------------------


class TestDebateGateRouterSemantics:
    """Verify that debate_gate_router produces the values that the conditional
    edge mapping in build_trading_graph() expects."""

    def test_router_skip_matches_graph_edge_label(self):
        """debate_gate_router must return 'skip' when debate is skipped,
        matching the conditional edge key declared in build_trading_graph()."""
        from cryptotrader.nodes.debate import debate_gate_router

        state = {"data": {"debate_skipped": True}}
        result = debate_gate_router(state)
        assert result == "skip"

    def test_router_debate_matches_graph_edge_label(self):
        """debate_gate_router must return 'debate' when debate proceeds,
        matching the conditional edge key declared in build_trading_graph()."""
        from cryptotrader.nodes.debate import debate_gate_router

        state = {"data": {"debate_skipped": False}}
        result = debate_gate_router(state)
        assert result == "debate"

    def test_router_default_missing_key_is_debate(self):
        """When debate_skipped key is absent, router defaults to 'debate'."""
        from cryptotrader.nodes.debate import debate_gate_router

        state = {"data": {}}
        result = debate_gate_router(state)
        assert result == "debate"
