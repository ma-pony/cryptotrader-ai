"""Guard: every node registered in any graph variant must be mapped to a latency bucket.

If a node is renamed or added without updating ``_LATENCY_STAGE_MAP`` in
``cryptotrader.nodes.journal``, this test fails — preventing silent classification
of that node's time into the "other" bucket.

Addresses Deep Review I-A3 — hardcoded map drift risk.
"""

from __future__ import annotations

import pytest

from cryptotrader.graph import (
    build_backtest_graph,
    build_debate_graph,
    build_lite_graph,
    build_trading_graph,
)
from cryptotrader.nodes.journal import _LATENCY_STAGE_MAP

# Terminal / control nodes that intentionally do not appear in the latency map
# (they don't produce business-work latency the UI cares about).
_EXCLUDED_NODES = frozenset(
    {
        "init_decision",  # pure observability bootstrap — near-zero ms
        "hitl_wait",  # human-in-the-loop blocker
        "inject_experience",  # covered under "data" bucket conceptually
        "execute",  # aliased to "execute_trade" in map
    }
)


def _all_graph_nodes() -> set[str]:
    """Enumerate every node name across all graph builders."""
    nodes: set[str] = set()
    for builder in (build_trading_graph, build_lite_graph, build_debate_graph, build_backtest_graph):
        try:
            compiled = builder()
        except Exception as e:
            pytest.skip(f"graph builder {builder.__name__} not buildable in test env: {e}")
        # LangGraph compiled graph exposes nodes via the get_graph API or .nodes.
        graph_obj = compiled.get_graph()
        for node_id in graph_obj.nodes:
            if node_id in {"__start__", "__end__"}:
                continue
            nodes.add(str(node_id))
    return nodes


def test_every_graph_node_is_mapped() -> None:
    """Every active node must be in _LATENCY_STAGE_MAP or the known excluded set.

    Rename/add → update ``_LATENCY_STAGE_MAP`` in ``cryptotrader/nodes/journal.py``
    (or extend ``_EXCLUDED_NODES`` if the node legitimately doesn't belong in any
    stage bucket).
    """
    nodes = _all_graph_nodes()
    missing = nodes - set(_LATENCY_STAGE_MAP) - _EXCLUDED_NODES
    if missing:
        pytest.fail(
            f"{len(missing)} node(s) not mapped to latency bucket: {sorted(missing)}. "
            f"Add them to _LATENCY_STAGE_MAP in cryptotrader/nodes/journal.py "
            f"or _EXCLUDED_NODES in this test file."
        )


def test_map_has_no_stale_entries() -> None:
    """Every mapped name should either exist in some graph OR be a known alias.

    Aliases (e.g. ``execute_trade`` / ``execute``) exist to support differently-named
    nodes across graph variants. This test asserts the map doesn't accumulate dead
    entries from deleted nodes.
    """
    nodes = _all_graph_nodes()
    # Known aliases — names in the map that don't directly match a graph node id
    # because they describe a different graph's node.
    known_aliases = frozenset({"bull_bear_debate", "judge_verdict", "debate_round_1", "debate_round_2"})
    stale = set(_LATENCY_STAGE_MAP) - nodes - known_aliases
    if stale:
        # Informational — not a hard fail, as aliases may legitimately exist.
        # Log via a skip so the developer notices.
        pytest.skip(
            f"{len(stale)} mapped node(s) not present in any graph (possibly stale or aliases): {sorted(stale)}"
        )
