"""Tests for otel.py, mcp/registry.py — push coverage over 70%."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

# ── otel.py ──


class TestOtel:
    def test_noop_span(self):
        from cryptotrader.otel import _NoOpSpan

        span = _NoOpSpan()
        span.set_attribute("key", "val")
        span.record_exception(ValueError("x"))
        span.set_status("ok")

    def test_noop_tracer(self):
        from cryptotrader.otel import _NoOpTracer

        tracer = _NoOpTracer()
        with tracer.start_as_current_span("test") as span:
            span.set_attribute("k", "v")

    def test_get_tracer_returns_noop_by_default(self):
        from cryptotrader.otel import _NoOpTracer, get_tracer

        tracer = get_tracer()
        assert isinstance(tracer, _NoOpTracer)

    def test_is_active_false_by_default(self):
        from cryptotrader.otel import is_active

        assert is_active() is False

    def test_setup_otel_no_endpoint(self):
        from cryptotrader.otel import setup_otel

        with patch.dict("os.environ", {}, clear=False):
            if "OTLP_ENDPOINT" in __import__("os").environ:
                with patch.dict("os.environ", {"OTLP_ENDPOINT": ""}):
                    setup_otel()
            else:
                setup_otel()


# ── mcp/registry.py ──


class _FakeTool:
    def __init__(self, name: str):
        self.name = name


class TestMCPRegistry:
    def test_register_and_find_server(self):
        from cryptotrader.mcp.registry import MCPRegistry

        cfg = MagicMock()
        cfg.call_timeout_s = 30
        registry = MCPRegistry(cfg)
        mock_server = MagicMock()
        mock_server.list_tools.return_value = [_FakeTool("tool_a"), _FakeTool("tool_b")]
        registry.register_server("test", mock_server)
        assert registry.find_server("tool_a") is mock_server
        assert registry.find_server("nonexistent") is None

    def test_list_tools(self):
        from cryptotrader.mcp.registry import MCPRegistry

        cfg = MagicMock()
        cfg.call_timeout_s = 30
        registry = MCPRegistry(cfg)
        mock_server = MagicMock()
        mock_server.list_tools.return_value = [_FakeTool("my_tool")]
        registry.register_server("test", mock_server)
        tools = registry.list_tools()
        assert "my_tool" in tools

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        from cryptotrader.mcp.registry import MCPRegistry, MCPToolNotFoundError

        cfg = MagicMock()
        cfg.call_timeout_s = 30
        registry = MCPRegistry(cfg)
        with pytest.raises(MCPToolNotFoundError):
            await registry.call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        from cryptotrader.mcp.registry import MCPRegistry

        cfg = MagicMock()
        cfg.call_timeout_s = 30
        registry = MCPRegistry(cfg)
        mock_server = MagicMock()
        mock_server.list_tools.return_value = [_FakeTool("my_tool")]

        async def fake_call(tool_name, args):
            return {"result": "ok"}

        mock_server.call_tool = fake_call
        registry.register_server("test", mock_server)
        result = await registry.call_tool("my_tool", {})
        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_call_tool_timeout(self):
        from cryptotrader.mcp.registry import MCPRegistry, MCPToolTimeoutError

        cfg = MagicMock()
        cfg.call_timeout_s = 0.01
        registry = MCPRegistry(cfg)
        mock_server = MagicMock()
        mock_server.list_tools.return_value = [_FakeTool("slow_tool")]

        async def slow_call(tool_name, args):
            await asyncio.sleep(10)
            return {}

        mock_server.call_tool = slow_call
        registry.register_server("test", mock_server)
        with pytest.raises(MCPToolTimeoutError):
            await registry.call_tool("slow_tool", {})

    @pytest.mark.asyncio
    async def test_health_check(self):
        from cryptotrader.mcp.registry import MCPRegistry

        cfg = MagicMock()
        cfg.call_timeout_s = 30
        registry = MCPRegistry(cfg)
        ok_server = MagicMock()
        ok_server.list_tools.return_value = []
        registry.register_server("ok", ok_server)
        # Now make bad_server fail health check — but register first with good list_tools
        bad_server = MagicMock()
        bad_server.list_tools.return_value = []
        registry.register_server("bad", bad_server)
        # Then make list_tools fail for health check
        bad_server.list_tools.side_effect = Exception("broken")
        result = await registry.health_check()
        assert result["ok"] is True
        assert result["bad"] is False

    def test_from_config_disabled_server(self):
        from cryptotrader.mcp.registry import MCPRegistry

        cfg = MagicMock()
        cfg.call_timeout_s = 30
        sc = MagicMock()
        sc.enabled = False
        sc.name = "disabled-server"
        cfg.servers = [sc]
        registry = MCPRegistry.from_config(cfg)
        assert len(registry._servers) == 0

    @pytest.mark.asyncio
    async def test_call_tool_non_dict_result(self):
        from cryptotrader.mcp.registry import MCPRegistry

        cfg = MagicMock()
        cfg.call_timeout_s = 30
        registry = MCPRegistry(cfg)
        mock_server = MagicMock()
        mock_server.list_tools.return_value = [_FakeTool("string_tool")]

        async def string_call(tool_name, args):
            return "just a string"

        mock_server.call_tool = string_call
        registry.register_server("test", mock_server)
        result = await registry.call_tool("string_tool", {})
        assert result == {"result": "just a string"}
