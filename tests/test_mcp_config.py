"""Tests for MCP configuration — MCPConfig, MCPServerConfig, TOML parsing."""

from __future__ import annotations

from cryptotrader.mcp.config import MCPConfig, MCPServerConfig


class TestMCPServerConfig:
    def test_defaults(self):
        sc = MCPServerConfig(name="test-server")
        assert sc.name == "test-server"
        assert sc.transport == "stdio"
        assert sc.host == "localhost"
        assert sc.port == 8080
        assert sc.enabled is True
        assert sc.tools == []

    def test_custom_fields(self):
        sc = MCPServerConfig(
            name="my-server",
            transport="http",
            host="127.0.0.1",
            port=9090,
            enabled=False,
            tools=["tool_a", "tool_b"],
        )
        assert sc.transport == "http"
        assert sc.enabled is False
        assert len(sc.tools) == 2


class TestMCPConfig:
    def test_defaults(self):
        cfg = MCPConfig()
        assert cfg.enabled is False
        assert cfg.transport == "stdio"
        assert cfg.fallback_on_error is True
        assert cfg.call_timeout_s == 5.0
        assert cfg.servers == []

    def test_with_servers(self):
        servers = [
            MCPServerConfig(name="s1", tools=["t1", "t2"]),
            MCPServerConfig(name="s2", enabled=False),
        ]
        cfg = MCPConfig(enabled=True, servers=servers)
        assert cfg.enabled is True
        assert len(cfg.servers) == 2
        assert cfg.servers[0].name == "s1"
        assert cfg.servers[1].enabled is False


class TestTOMLParsing:
    def test_load_config_default_mcp(self):
        from cryptotrader.config import load_config

        cfg = load_config()
        assert isinstance(cfg.mcp, MCPConfig)
        assert cfg.mcp.enabled is False

    def test_default_toml_has_four_servers(self):
        from cryptotrader.config import load_config

        cfg = load_config()
        assert len(cfg.mcp.servers) == 4
        names = {s.name for s in cfg.mcp.servers}
        assert "cryptotrader-binance" in names
        assert "cryptotrader-macro" in names
        assert "cryptotrader-onchain" in names
        assert "cryptotrader-news" in names

    def test_binance_server_has_three_tools(self):
        from cryptotrader.config import load_config

        cfg = load_config()
        binance_server = next(s for s in cfg.mcp.servers if s.name == "cryptotrader-binance")
        assert len(binance_server.tools) == 3
        assert "binance_derivatives" in binance_server.tools

    def test_build_mcp_config_no_section(self):
        from cryptotrader.config import _build_mcp_config

        cfg = _build_mcp_config({})
        assert cfg.enabled is False
        assert cfg.servers == []
