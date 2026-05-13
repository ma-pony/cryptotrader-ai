"""Tests for tool isolation — AgentConfig.tools filtering."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage, SystemMessage

from cryptotrader.agents.base import ToolAgent
from cryptotrader.config import AgentConfig, AgentsConfig


def _fake_pb() -> MagicMock:
    pb = MagicMock()
    pb.build.return_value = (SystemMessage(content="sys"), HumanMessage(content="usr"))
    return pb


def _real_pb(agent_id: str) -> MagicMock:
    from cryptotrader.agents.prompt_builder import (
        DefaultSkillProvider,
        PromptBuilder,
    )

    repo_root = Path(__file__).parent.parent
    return PromptBuilder(
        agent_id=agent_id,
        config_dir=repo_root / "config" / "agents",
        skill_provider=DefaultSkillProvider(skills_root=repo_root / "agent_skills"),
    )


class TestToolIsolation:
    def test_single_tool_filter(self):
        agents = {
            "custom_agent": AgentConfig(
                agent_id="custom_agent",
                tools=["get_funding_rate_history"],
            ),
        }
        cfg = AgentsConfig(_agents=agents)
        agent = cfg.build("custom_agent", prompt_builder=_fake_pb())
        assert isinstance(agent, ToolAgent)
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "get_funding_rate_history"

    def test_empty_tools_gets_default_for_builtin(self):
        cfg = AgentsConfig()
        agent = cfg.build("chain_agent", prompt_builder=_real_pb("chain"))
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
            agent = cfg.build("custom_agent", prompt_builder=_fake_pb())
        assert isinstance(agent, ToolAgent)
        assert len(agent.tools) == 0
        assert "unknown tool" in caplog.text
