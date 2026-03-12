"""Tests for supervisor pattern modules: skills, tools, langchain_agents, graph_supervisor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Skills tests ──


def test_get_skill_descriptions():
    from cryptotrader.agents.skills import get_skill_descriptions

    desc = get_skill_descriptions()
    assert "funding_rate_analysis" in desc
    assert "btc_dominance_analysis" in desc
    assert "liquidation_cascade_analysis" in desc
    assert "fear_greed_interpretation" in desc


def test_load_skill_content_found():
    from cryptotrader.agents.skills import load_skill_content

    content = load_skill_content("funding_rate_analysis")
    assert "Funding Rate" in content
    assert ">0.1%" in content


def test_load_skill_content_not_found():
    from cryptotrader.agents.skills import load_skill_content

    result = load_skill_content("nonexistent")
    assert "not found" in result
    assert "funding_rate_analysis" in result


def test_skills_count():
    from cryptotrader.agents.skills import TRADING_SKILLS

    assert len(TRADING_SKILLS) == 4
    for skill in TRADING_SKILLS:
        assert "name" in skill
        assert "description" in skill
        assert "content" in skill
        assert len(skill["content"]) > 50


# ── Tools tests ──


def test_load_skill_tool():
    from cryptotrader.agents.tools import load_skill

    result = load_skill.invoke("funding_rate_analysis")
    assert "Funding Rate" in result


def test_load_skill_tool_not_found():
    from cryptotrader.agents.tools import load_skill

    result = load_skill.invoke("bad_name")
    assert "not found" in result


@patch("cryptotrader.learning.verbal.get_experience", new_callable=AsyncMock)
@patch("cryptotrader.learning.verbal.format_experience_text", new_callable=AsyncMock)
def test_load_past_experience_tool(mock_format, mock_get_exp):
    from cryptotrader.agents.tools import load_past_experience

    mock_get_exp.return_value = [MagicMock()]
    mock_format.return_value = "Past: BTC went up after similar pattern"
    result = load_past_experience.invoke("BTC bullish breakout")
    assert "BTC" in result


@patch("cryptotrader.learning.verbal.get_experience", new_callable=AsyncMock)
@patch("cryptotrader.learning.verbal.format_experience_text", new_callable=AsyncMock)
def test_load_past_experience_empty(mock_format, mock_get_exp):
    from cryptotrader.agents.tools import load_past_experience

    mock_get_exp.return_value = []
    mock_format.return_value = ""
    result = load_past_experience.invoke("unknown context")
    assert "No relevant past experience" in result


# ── Graph supervisor tests ──


@pytest.mark.asyncio
async def test_supervisor_analyze_valid_json():
    """supervisor_analyze parses valid JSON from supervisor output."""
    from cryptotrader.graph_supervisor import supervisor_analyze

    mock_snapshot = MagicMock()
    mock_snapshot.pair = "BTC/USDT"
    mock_snapshot.market.ticker = {"last": 50000, "baseVolume": 1000}
    mock_snapshot.market.volatility = 0.02
    mock_snapshot.market.funding_rate = 0.01
    mock_snapshot.market.orderbook_imbalance = 0.5
    mock_snapshot.onchain.open_interest = 1000000
    mock_snapshot.onchain.liquidations_24h = 50000
    mock_snapshot.macro.fear_greed_index = 65
    mock_snapshot.macro.btc_dominance = 55.0

    state = {
        "messages": [],
        "data": {"snapshot": mock_snapshot},
        "metadata": {"verdict_model": "gpt-4o-mini"},
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }

    mock_result = {
        "messages": [
            MagicMock(content='{"action": "long", "confidence": 0.8, "reasoning": "bullish", "position_scale": 0.7}')
        ]
    }

    with patch("cryptotrader.graph_supervisor.create_supervisor_agent") as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = mock_result
        mock_create.return_value = mock_agent

        result = await supervisor_analyze(state)

    verdict = result["data"]["verdict"]
    assert verdict["action"] == "long"
    assert verdict["confidence"] == 0.8
    assert verdict["position_scale"] == 0.7
    # Ensure defaults are filled in
    assert "divergence" in verdict
    assert "thesis" in verdict
    assert "invalidation" in verdict


@pytest.mark.asyncio
async def test_supervisor_analyze_invalid_json_fallback():
    """supervisor_analyze falls back on non-JSON output."""
    from cryptotrader.graph_supervisor import supervisor_analyze

    mock_snapshot = MagicMock()
    mock_snapshot.pair = "ETH/USDT"
    mock_snapshot.market.ticker = {"last": 3000, "baseVolume": 500}
    mock_snapshot.market.volatility = 0.03
    mock_snapshot.market.funding_rate = -0.01
    mock_snapshot.market.orderbook_imbalance = 0.3
    mock_snapshot.onchain.open_interest = 500000
    mock_snapshot.onchain.liquidations_24h = 20000
    mock_snapshot.macro.fear_greed_index = 30
    mock_snapshot.macro.btc_dominance = 58.0

    state = {
        "messages": [],
        "data": {"snapshot": mock_snapshot},
        "metadata": {},
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }

    mock_result = {"messages": [MagicMock(content="I think we should hold because of uncertainty.")]}

    with patch("cryptotrader.graph_supervisor.create_supervisor_agent") as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = mock_result
        mock_create.return_value = mock_agent

        result = await supervisor_analyze(state)

    verdict = result["data"]["verdict"]
    assert verdict["action"] == "hold"
    assert verdict["confidence"] == 0.5
    assert "uncertainty" in verdict["reasoning"]


def test_build_supervisor_graph_compiles():
    """Supervisor graph compiles without errors."""
    from cryptotrader.graph_supervisor import build_supervisor_graph

    graph = build_supervisor_graph()
    assert graph is not None


# ── CLI graph mode test ──


def test_cli_graph_option_exists():
    """CLI run command accepts --graph option."""
    from cli.main import app

    # Find the 'run' command and check it has a 'graph' parameter
    for command_info in app.registered_commands:
        if command_info.callback and command_info.callback.__name__ == "run":
            import inspect

            sig = inspect.signature(command_info.callback)
            assert "graph" in sig.parameters
            break
    else:
        pytest.fail("'run' command not found in CLI app")


# ── API graph_mode test ──


def test_analyze_request_has_graph_mode():
    """AnalyzeRequest model includes graph_mode field."""
    from api.routes.analyze import AnalyzeRequest

    req = AnalyzeRequest(pair="BTC/USDT", graph_mode="supervisor")
    assert req.graph_mode == "supervisor"

    # Default is 'full'
    req2 = AnalyzeRequest()
    assert req2.graph_mode == "full"
