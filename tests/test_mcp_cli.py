"""Tests for arena mcp list / mcp call CLI commands."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


class TestMCPList:
    def test_mcp_list_disabled(self):
        result = runner.invoke(app, ["mcp", "list"])
        assert result.exit_code == 0
        assert "MCP is disabled" in result.output or "cryptotrader-binance" in result.output

    def test_mcp_list_shows_servers(self):
        result = runner.invoke(app, ["mcp", "list"])
        assert result.exit_code == 0
        assert "cryptotrader-binance" in result.output
        assert "cryptotrader-macro" in result.output

    def test_mcp_list_no_runtime_error(self):
        result = runner.invoke(app, ["mcp", "list"])
        assert result.exit_code == 0
        assert result.exception is None


class TestMCPCall:
    def test_call_disabled_exits_nonzero(self):
        result = runner.invoke(app, ["mcp", "call", "binance_derivatives"])
        assert result.exit_code == 1
        assert "disabled" in result.output.lower()

    def test_call_invalid_json_exits_nonzero(self):
        from cryptotrader.config import AppConfig
        from cryptotrader.mcp.config import MCPConfig

        cfg = AppConfig(mcp=MCPConfig(enabled=True))
        with patch("cryptotrader.config.load_config", return_value=cfg):
            result = runner.invoke(app, ["mcp", "call", "test", "--args", "{bad json"])
        assert result.exit_code == 1
