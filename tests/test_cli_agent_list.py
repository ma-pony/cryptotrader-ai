"""Tests for `arena agent list` CLI command."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from cli.main import app
from cryptotrader.config import AgentConfig, AgentsConfig, AppConfig

runner = CliRunner()


def _make_config(**agents_overrides) -> AppConfig:
    agents_map = {}
    for agent_id, data in agents_overrides.items():
        agents_map[agent_id] = AgentConfig(agent_id=agent_id, **data)
    return AppConfig(agents=AgentsConfig(_agents=agents_map))


class TestAgentList:
    def test_lists_builtin_agents(self):
        with patch("cryptotrader.config.load_config", return_value=AppConfig()):
            result = runner.invoke(app, ["agent", "list"])
        assert result.exit_code == 0
        assert "tech_agent" in result.output
        assert "chain_agent" in result.output
        assert "news_agent" in result.output
        assert "macro_agent" in result.output
        assert "builtin" in result.output

    def test_lists_custom_agent(self):
        cfg = _make_config(whale_agent={"model": "gpt-5"})
        with patch("cryptotrader.config.load_config", return_value=cfg):
            result = runner.invoke(app, ["agent", "list"])
        assert result.exit_code == 0
        assert "whale_agent" in result.output
        assert "custom" in result.output
        assert "gpt-5" in result.output

    def test_shows_default_model_placeholder(self):
        with patch("cryptotrader.config.load_config", return_value=AppConfig()):
            result = runner.invoke(app, ["agent", "list"])
        assert result.exit_code == 0
        assert "<default>" in result.output

    def test_no_runtime_error(self):
        with patch("cryptotrader.config.load_config", return_value=AppConfig()):
            result = runner.invoke(app, ["agent", "list"])
        assert result.exit_code == 0
        assert result.exception is None
