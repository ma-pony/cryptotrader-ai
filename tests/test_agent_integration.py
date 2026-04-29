"""Integration tests — SC-002/SC-003/SC-004/SC-005/SC-010."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cryptotrader.config import AgentConfig, AgentsConfig


@pytest.fixture
def _reset_config_cache():
    import cryptotrader.config as cfg_mod

    original = cfg_mod._cached_config
    cfg_mod._cached_config = None
    yield
    cfg_mod._cached_config = original


class TestSC002ModelConfig:
    """SC-002: TOML config model end-to-end."""

    def test_custom_model_propagates_to_agent(self):
        agents = {
            "news_agent": AgentConfig(agent_id="news_agent", model="gpt-5"),
        }
        cfg = AgentsConfig(_agents=agents)
        agent = cfg.build("news_agent")
        assert agent.model == "gpt-5"

    def test_custom_timeout_on_agent_config(self):
        agents = {
            "news_agent": AgentConfig(agent_id="news_agent", timeout_seconds=30),
        }
        cfg = AgentsConfig(_agents=agents)
        agent_cfg = cfg.get("news_agent")
        assert agent_cfg is not None
        assert agent_cfg.timeout_seconds == 30


class TestSC003EnabledFalse:
    """SC-003: enabled=false excludes agent from pipeline."""

    def test_disabled_agent_not_in_list_active(self):
        agents = {
            "tech_agent": AgentConfig(agent_id="tech_agent", enabled=False),
        }
        cfg = AgentsConfig(_agents=agents)
        active_ids = [a.agent_id for a in cfg.list_active()]
        assert "tech_agent" not in active_ids

    def test_disabled_agent_build_raises(self):
        from cryptotrader.config import AgentNotFoundError

        agents = {
            "tech_agent": AgentConfig(agent_id="tech_agent", enabled=False),
        }
        cfg = AgentsConfig(_agents=agents)
        with pytest.raises(AgentNotFoundError):
            cfg.build("tech_agent")


class TestSC004SkillInjection:
    """SC-004: Skill file content injected into prompt."""

    def test_skill_content_in_role_description(self, tmp_path):
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()
        (skill_dir / "momentum_strategy.md").write_text("RSI < 35 and MACD golden cross")

        with patch(
            "cryptotrader.agents.skill_loader._DEFAULT_SEARCH_PATHS",
            [skill_dir],
        ):
            agents = {
                "tech_agent": AgentConfig(
                    agent_id="tech_agent",
                    skills=["momentum_strategy"],
                ),
            }
            cfg = AgentsConfig(_agents=agents)
            agent = cfg.build("tech_agent")

        assert "RSI < 35" in agent.role_description
        assert "STRATEGY SKILLS" in agent.role_description


class TestSC005CustomAgent:
    """SC-005: Custom agent registered and buildable."""

    def test_custom_agent_builds(self, tmp_path):
        prompt_file = tmp_path / "whale_prompt.md"
        prompt_file.write_text("You analyze whale behavior.")
        agents = {
            "whale_agent": AgentConfig(
                agent_id="whale_agent",
                prompt_template=str(prompt_file),
                tools=["get_whale_transfers"],
            ),
        }
        cfg = AgentsConfig(_agents=agents)
        agent = cfg.build("whale_agent")
        assert agent.agent_id == "whale_agent"
        assert "whale behavior" in agent.role_description
        assert len(agent.tools) == 1


class TestSC010PromptTemplateOverride:
    """SC-010: Custom prompt_template replaces built-in ROLE constant."""

    def test_custom_prompt_replaces_builtin_role(self, tmp_path):
        from cryptotrader.agents.tech import ROLE as BUILTIN_ROLE

        custom_content = "You are a custom technical analyst using quantum computing."
        prompt_file = tmp_path / "custom_tech.md"
        prompt_file.write_text(custom_content)

        agents = {
            "tech_agent": AgentConfig(
                agent_id="tech_agent",
                prompt_template=str(prompt_file),
            ),
        }
        cfg = AgentsConfig(_agents=agents)
        agent = cfg.build("tech_agent")

        from cryptotrader.agents.base import BaseAgent

        assert isinstance(agent, BaseAgent)
        assert custom_content in agent.role_description
        assert BUILTIN_ROLE not in agent.role_description
