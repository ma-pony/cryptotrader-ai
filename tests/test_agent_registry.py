"""Tests for AgentsConfig.build() — Agent instance registry."""

from __future__ import annotations

import pytest

from cryptotrader.config import AgentConfig, AgentNotFoundError, AgentsConfig


class TestBuild:
    def test_build_tech_agent_returns_correct_type(self):
        cfg = AgentsConfig()
        agent = cfg.build("tech_agent")
        assert agent.__class__.__name__ == "TechAgent"

    def test_build_chain_agent_returns_tool_agent(self):
        from cryptotrader.agents.base import ToolAgent

        cfg = AgentsConfig()
        agent = cfg.build("chain_agent", backtest_mode=True)
        assert isinstance(agent, ToolAgent)
        assert agent.backtest_mode is True

    def test_build_unknown_agent_raises(self):
        cfg = AgentsConfig()
        with pytest.raises(AgentNotFoundError, match="unknown_agent"):
            cfg.build("unknown_agent")

    def test_build_custom_model_passed_through(self):
        agents = {
            "tech_agent": AgentConfig(agent_id="tech_agent", model="gpt-5"),
        }
        cfg = AgentsConfig(_agents=agents)
        agent = cfg.build("tech_agent")
        assert agent.model == "gpt-5"

    def test_build_disabled_agent_raises(self):
        agents = {
            "tech_agent": AgentConfig(agent_id="tech_agent", enabled=False),
        }
        cfg = AgentsConfig(_agents=agents)
        with pytest.raises(AgentNotFoundError):
            cfg.build("tech_agent")

    def test_build_custom_agent_with_prompt_template(self, tmp_path):
        from cryptotrader.agents.base import BaseAgent

        prompt_file = tmp_path / "whale_prompt.md"
        prompt_file.write_text("You are a whale watcher agent.")
        agents = {
            "whale_agent": AgentConfig(
                agent_id="whale_agent",
                prompt_template=str(prompt_file),
            ),
        }
        cfg = AgentsConfig(_agents=agents)
        agent = cfg.build("whale_agent")
        assert isinstance(agent, BaseAgent)
        assert "whale watcher" in agent.role_description

    def test_build_custom_agent_with_tools(self):
        from cryptotrader.agents.base import ToolAgent

        agents = {
            "whale_agent": AgentConfig(
                agent_id="whale_agent",
                tools=["get_whale_transfers"],
            ),
        }
        cfg = AgentsConfig(_agents=agents)
        agent = cfg.build("whale_agent")
        assert isinstance(agent, ToolAgent)
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "get_whale_transfers"

    def test_build_custom_agent_no_prompt_no_tools(self):
        from cryptotrader.agents.base import BaseAgent

        agents = {
            "custom_agent": AgentConfig(agent_id="custom_agent"),
        }
        cfg = AgentsConfig(_agents=agents)
        agent = cfg.build("custom_agent")
        assert isinstance(agent, BaseAgent)
        assert "custom_agent" in agent.role_description

    def test_list_active_excludes_disabled(self):
        agents = {
            "tech_agent": AgentConfig(agent_id="tech_agent", enabled=False),
            "news_agent": AgentConfig(agent_id="news_agent", enabled=True),
        }
        cfg = AgentsConfig(_agents=agents)
        active = cfg.list_active()
        ids = [a.agent_id for a in active]
        assert "tech_agent" not in ids
        assert "news_agent" in ids
