"""MCP CLI remains code-owned and independent from trading configuration."""

from typer.testing import CliRunner

from cli.main import app


def test_mcp_list_shows_code_owned_servers() -> None:
    result = CliRunner().invoke(app, ["mcp", "list"])

    assert result.exit_code == 0
    assert "MCP is disabled" in result.output
    assert "cryptotrader-binance" in result.output
    assert "cryptotrader-macro" in result.output


def test_mcp_call_is_not_registered_while_catalog_is_disabled() -> None:
    result = CliRunner().invoke(app, ["mcp", "--help"])

    assert result.exit_code == 0
    assert "call" not in result.output
