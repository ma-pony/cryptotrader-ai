"""Tests for tool isolation — AgentConfig.tools filtering."""

from __future__ import annotations

import logging

from cryptotrader.agents.base import ToolAgent
from cryptotrader.config import AgentConfig, AgentsConfig


class TestToolIsolation:
    def test_single_tool_filter(self):
        agents = {
            "custom_agent": AgentConfig(
                agent_id="custom_agent",
                tools=["get_funding_rate_history"],
            ),
        }
        cfg = AgentsConfig(_agents=agents)
        agent = cfg.build("custom_agent")
        assert isinstance(agent, ToolAgent)
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "get_funding_rate_history"

    def test_empty_tools_gets_default_for_builtin(self):
        cfg = AgentsConfig()
        agent = cfg.build("chain_agent")
        assert isinstance(agent, ToolAgent)
        assert len(agent.tools) > 0

    def test_unknown_tool_warns(self, caplog):
        agents = {
            "custom_agent": AgentConfig(
                agent_id="custom_agent",
                tools=["nonexistent_tool"],
            ),
        }
        cfg = AgentsConfig(_agents=agents)
        with caplog.at_level(logging.WARNING):
            agent = cfg.build("custom_agent")
        assert isinstance(agent, ToolAgent)
        assert len(agent.tools) == 0
        assert "unknown tool" in caplog.text
