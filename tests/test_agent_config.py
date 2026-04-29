"""Tests for AgentConfig / AgentsConfig dataclasses and TOML parsing."""

from __future__ import annotations

import pytest

from cryptotrader.config import (
    AgentConfig,
    AgentNotFoundError,
    AgentsConfig,
    AppConfig,
    ConfigurationError,
    _build_agents_config,
    _build_config,
    validate_config,
)


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig(agent_id="test")
        assert cfg.agent_id == "test"
        assert cfg.model == ""
        assert cfg.timeout_seconds == 0
        assert cfg.enabled is True
        assert cfg.prompt_template == ""
        assert cfg.tools == []
        assert cfg.skills == []
        assert cfg.regime_skills == {}


class TestAgentsConfig:
    def test_get_returns_configured_agent(self):
        ac = AgentConfig(agent_id="news_agent", model="gpt-5")
        cfg = AgentsConfig(_agents={"news_agent": ac})
        result = cfg.get("news_agent")
        assert result is not None
        assert result.model == "gpt-5"

    def test_get_returns_builtin_default_when_not_configured(self):
        cfg = AgentsConfig()
        result = cfg.get("tech_agent")
        assert result is not None
        assert result.agent_id == "tech_agent"
        assert result.model == ""

    def test_get_returns_none_for_unknown(self):
        cfg = AgentsConfig()
        assert cfg.get("unknown_agent") is None

    def test_list_active_returns_4_builtins_when_empty(self):
        cfg = AgentsConfig()
        active = cfg.list_active()
        ids = [a.agent_id for a in active]
        assert len(ids) == 4
        assert "tech_agent" in ids
        assert "chain_agent" in ids
        assert "news_agent" in ids
        assert "macro_agent" in ids

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
        # chain_agent and macro_agent are builtins not in _agents, so they appear
        assert "chain_agent" in ids
        assert "macro_agent" in ids

    def test_list_active_custom_agent(self):
        agents = {
            "whale_agent": AgentConfig(agent_id="whale_agent", enabled=True),
        }
        cfg = AgentsConfig(_agents=agents)
        active = cfg.list_active()
        ids = [a.agent_id for a in active]
        assert "whale_agent" in ids
        assert len(ids) == 5  # 4 builtins + 1 custom


class TestAgentNotFoundError:
    def test_message_and_fields(self):
        err = AgentNotFoundError("missing_agent", ["tech_agent", "news_agent"])
        assert err.agent_id == "missing_agent"
        assert err.registered == ["tech_agent", "news_agent"]
        assert "missing_agent" in str(err)


class TestBuildAgentsConfig:
    def test_parses_toml_agents_section(self):
        toml_data = {
            "agents": {
                "news_agent": {
                    "model": "gpt-5",
                    "timeout_seconds": 45,
                    "enabled": True,
                    "skills": ["momentum_strategy"],
                },
                "whale_agent": {
                    "model": "claude-3.5",
                    "prompt_template": "prompts/whale.md",
                    "tools": ["get_whale_transfers"],
                },
            }
        }
        cfg = _build_agents_config(toml_data)
        news = cfg.get("news_agent")
        assert news is not None
        assert news.model == "gpt-5"
        assert news.timeout_seconds == 45
        assert news.skills == ["momentum_strategy"]

        whale = cfg.get("whale_agent")
        assert whale is not None
        assert whale.model == "claude-3.5"
        assert whale.tools == ["get_whale_transfers"]

    def test_parses_regime_skills(self):
        toml_data = {
            "agents": {
                "macro_agent": {
                    "regime_skills": {
                        "trending_up": ["bull_momentum"],
                        "high_volatility": ["vol_protect"],
                    },
                }
            }
        }
        cfg = _build_agents_config(toml_data)
        macro = cfg.get("macro_agent")
        assert macro is not None
        assert macro.regime_skills == {
            "trending_up": ["bull_momentum"],
            "high_volatility": ["vol_protect"],
        }

    def test_empty_agents_section(self):
        cfg = _build_agents_config({})
        assert cfg.list_active() == AgentsConfig().list_active()

    def test_build_config_includes_agents(self):
        toml_data = {
            "models": {"fallback": "deepseek-chat"},
            "agents": {
                "news_agent": {"model": "gpt-5"},
            },
        }
        app_cfg = _build_config(toml_data)
        assert app_cfg.agents.get("news_agent") is not None
        assert app_cfg.agents.get("news_agent").model == "gpt-5"


class TestValidateConfigAgents:
    def _make_app_config(self, **agents_kwargs) -> AppConfig:
        agents_map = {}
        for agent_id, data in agents_kwargs.items():
            agents_map[agent_id] = AgentConfig(agent_id=agent_id, **data)
        return AppConfig(agents=AgentsConfig(_agents=agents_map))

    def test_all_builtins_disabled_raises(self):
        agents = {
            bid: AgentConfig(agent_id=bid, enabled=False)
            for bid in ("tech_agent", "chain_agent", "news_agent", "macro_agent")
        }
        cfg = AppConfig(agents=AgentsConfig(_agents=agents))
        with pytest.raises(ConfigurationError, match="agents"):
            validate_config(cfg)

    def test_prompt_template_missing_file_raises(self):
        cfg = self._make_app_config(
            whale_agent={"prompt_template": "/nonexistent/path/whale.md"},
        )
        with pytest.raises(ConfigurationError, match="prompt_template"):
            validate_config(cfg)

    def test_invalid_timeout_raises(self):
        cfg = self._make_app_config(
            tech_agent={"timeout_seconds": -5},
        )
        with pytest.raises(ConfigurationError, match="timeout_seconds"):
            validate_config(cfg)

    def test_valid_config_passes(self):
        cfg = AppConfig()
        validate_config(cfg)
