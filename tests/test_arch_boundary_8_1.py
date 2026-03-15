"""Tests for architecture task 8.1 -- same-layer dependency elimination.

Verifies:
1. read_portfolio_from_exchange is importable from portfolio.manager
2. nodes/verdict.py no longer imports from nodes/execution for read_portfolio_from_exchange
3. graph_supervisor.py has experimental status comment at the top
4. agents/langchain_agents.py has experimental status comment at the top
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_SRC = pathlib.Path(__file__).parent.parent / "src"


# ── Test 1: read_portfolio_from_exchange resides in portfolio.manager ──


def test_read_portfolio_from_exchange_importable_from_portfolio_manager():
    """read_portfolio_from_exchange must be importable from portfolio.manager."""
    from cryptotrader.portfolio import manager as pm_module

    assert hasattr(pm_module, "read_portfolio_from_exchange"), (
        "read_portfolio_from_exchange must be defined in portfolio/manager.py"
    )
    assert callable(pm_module.read_portfolio_from_exchange)


def test_read_portfolio_from_exchange_is_async():
    """read_portfolio_from_exchange must be an async function."""
    from cryptotrader.portfolio.manager import read_portfolio_from_exchange

    assert inspect.iscoroutinefunction(read_portfolio_from_exchange), (
        "read_portfolio_from_exchange must be an async function"
    )


# ── Test 2: nodes/verdict.py does NOT import read_portfolio_from_exchange from nodes/execution ──


def test_verdict_does_not_import_read_portfolio_from_execution():
    """nodes/verdict.py must not import read_portfolio_from_exchange from nodes.execution."""
    verdict_path = _SRC / "cryptotrader/nodes/verdict.py"
    tree = ast.parse(verdict_path.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "nodes.execution" in node.module:
            for alias in node.names:
                assert alias.name != "read_portfolio_from_exchange", (
                    "nodes/verdict.py must not import read_portfolio_from_exchange "
                    "from nodes.execution -- use portfolio.manager instead"
                )


def test_verdict_imports_read_portfolio_from_portfolio_manager():
    """nodes/verdict.py must import read_portfolio_from_exchange from portfolio.manager."""
    verdict_path = _SRC / "cryptotrader/nodes/verdict.py"
    source = verdict_path.read_text()

    # Accept both top-level and lazy (inside-function) imports
    assert "portfolio.manager" in source, "nodes/verdict.py must reference portfolio.manager"
    assert "read_portfolio_from_exchange" in source, (
        "nodes/verdict.py must reference read_portfolio_from_exchange from portfolio.manager"
    )


# ── Test 3: graph_supervisor.py has experimental status comment ──


def test_graph_supervisor_has_experimental_comment():
    """graph_supervisor.py must have a top-level experimental status comment."""
    path = _SRC / "cryptotrader/graph_supervisor.py"
    first_lines = "\n".join(path.read_text().splitlines()[:20]).lower()
    assert any(kw in first_lines for kw in ("experimental", "not enabled", "not used")), (
        "graph_supervisor.py must have an experimental/not-in-main-path status comment at the top of the file"
    )


# ── Test 4: agents/langchain_agents.py has experimental status comment ──


def test_langchain_agents_has_experimental_comment():
    """agents/langchain_agents.py must have a top-level experimental status comment."""
    path = _SRC / "cryptotrader/agents/langchain_agents.py"
    first_lines = "\n".join(path.read_text().splitlines()[:20]).lower()
    assert any(kw in first_lines for kw in ("experimental", "not enabled", "not used")), (
        "agents/langchain_agents.py must have an experimental/not-in-main-path status comment at the top of the file"
    )


# ── Test 5: Functional tests ──


@pytest.mark.asyncio
async def test_read_portfolio_from_exchange_functional():
    """read_portfolio_from_exchange from portfolio.manager returns correct structure."""
    from cryptotrader.portfolio.manager import read_portfolio_from_exchange

    state = {
        "messages": [],
        "data": {"snapshot_summary": {"price": 50000.0}},
        "metadata": {"pair": "BTC/USDT", "engine": "paper"},
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }

    mock_exchange = MagicMock()
    mock_exchange.get_balance = AsyncMock(return_value={"USDT": 10000.0})
    mock_exchange.get_positions = AsyncMock(return_value={"BTC/USDT": {"amount": 0.1, "avg_price": 48000.0}})

    with patch(
        "cryptotrader.nodes.execution._get_exchange",
        new_callable=AsyncMock,
        return_value=(mock_exchange, None),
    ):
        result = await read_portfolio_from_exchange(state)

    assert result is not None
    assert "cash" in result
    assert "positions" in result
    assert "total_value" in result
    assert result["cash"] == 10000.0


@pytest.mark.asyncio
async def test_read_portfolio_from_exchange_returns_none_on_failure():
    """read_portfolio_from_exchange returns None when exchange call fails."""
    from cryptotrader.portfolio.manager import read_portfolio_from_exchange

    state = {
        "messages": [],
        "data": {"snapshot_summary": {"price": 50000.0}},
        "metadata": {"pair": "BTC/USDT", "engine": "paper"},
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }

    with patch(
        "cryptotrader.nodes.execution._get_exchange",
        side_effect=RuntimeError("exchange down"),
    ):
        result = await read_portfolio_from_exchange(state)

    assert result is None
