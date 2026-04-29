"""Miscellaneous coverage tests for small modules to push past 70%."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── api/dependencies.py ──


class TestVerifyApiKey:
    @pytest.mark.asyncio
    async def test_disabled_mode_short_circuits(self):
        """AUTH_MODE=disabled returns early regardless of header — that's the test-env default."""
        from api.dependencies import verify_api_key

        with patch("api.dependencies.AUTH_MODE", "disabled"):
            req = MagicMock()
            req.method = "GET"
            req.url.path = "/test"
            await verify_api_key(req)  # no exception

    @pytest.mark.asyncio
    async def test_valid_key(self):
        from api.dependencies import verify_api_key

        req = MagicMock()
        req.headers.get.return_value = "secret123"
        with patch("api.dependencies.AUTH_MODE", "enabled"), patch("api.dependencies.API_KEY", "secret123"):
            await verify_api_key(req)

    @pytest.mark.asyncio
    async def test_invalid_key(self):
        from fastapi import HTTPException

        from api.dependencies import verify_api_key

        req = MagicMock()
        req.headers.get.return_value = "wrong"
        with patch("api.dependencies.AUTH_MODE", "enabled"), patch("api.dependencies.API_KEY", "secret123"):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(req)
            assert exc_info.value.status_code == 401


# ── cryptotrader/mcp/__init__.py ──


class TestMCPLazyImports:
    def test_mcp_adapter(self):
        import cryptotrader.mcp as mcp_mod

        assert mcp_mod.MCPAdapter is not None

    def test_mcp_config(self):
        import cryptotrader.mcp as mcp_mod

        assert mcp_mod.MCPConfig is not None
        assert mcp_mod.MCPServerConfig is not None

    def test_mcp_registry(self):
        import cryptotrader.mcp as mcp_mod

        assert mcp_mod.MCPRegistry is not None

    def test_mcp_errors(self):
        import cryptotrader.mcp as mcp_mod

        assert mcp_mod.MCPToolNotFoundError is not None
        assert mcp_mod.MCPToolTimeoutError is not None

    def test_mcp_unknown_attribute(self):
        import cryptotrader.mcp as mcp_mod

        with pytest.raises(AttributeError):
            _ = mcp_mod.NonexistentAttribute


# ── cryptotrader/data/binance_audit.py ──


class TestBinanceAudit:
    @pytest.mark.asyncio
    async def test_audit_token_high_risk(self):
        from unittest.mock import AsyncMock

        from cryptotrader.data.binance_audit import BinanceAudit

        audit = BinanceAudit()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "code": "000000",
            "data": {"isHoneypot": True, "buyTax": "50", "sellTax": "50"},
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        cfg = MagicMock()
        cfg.providers.token_tax_threshold = 10.0

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("cryptotrader.config.load_config", return_value=cfg),
        ):
            result = await audit.audit_token("TOKEN", "0xabc", "BSC")
        assert result["risk_level"] == "HIGH"
        assert result["is_honeypot"] is True

    @pytest.mark.asyncio
    async def test_audit_token_api_error(self):
        from unittest.mock import AsyncMock

        from cryptotrader.data.binance_audit import BinanceAudit

        audit = BinanceAudit()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await audit.audit_token("TOKEN", "0xabc")
        assert result["risk_level"] == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_audit_token_bad_code(self):
        from unittest.mock import AsyncMock

        from cryptotrader.data.binance_audit import BinanceAudit

        audit = BinanceAudit()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": "999999"}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await audit.audit_token("TOKEN", "0xabc")
        assert result["risk_level"] == "UNKNOWN"


# ── cryptotrader/chat/event_bus.py ──


class TestEventBus:
    def _make_bus(self):
        from unittest.mock import AsyncMock

        from cryptotrader.chat.event_bus import EventBus

        buffer = AsyncMock()
        buffer.next_event_id = AsyncMock(return_value=1)
        buffer.push = AsyncMock()
        return EventBus(session_id="test-session", buffer=buffer)

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        bus = self._make_bus()
        q = bus.subscribe()
        await bus.publish("test_event", {"key": "val"})
        envelope = q.get_nowait()
        assert envelope.type == "test_event"
        assert envelope.data == {"key": "val"}

    @pytest.mark.asyncio
    async def test_publish_no_subscribers(self):
        bus = self._make_bus()
        envelope = await bus.publish("unsubscribed", {})
        assert envelope.event_id == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = self._make_bus()
        q = bus.subscribe()
        bus.unsubscribe(q)
        await bus.publish("evt", {})
        assert q.empty()


# ── cryptotrader/mcp/compat.py ──


class TestMCPCompat:
    def test_fastmcp_stub_tool_registration(self):
        from cryptotrader.mcp.compat import FastMCP

        server = FastMCP("test-server")
        assert server.name == "test-server"

        @server.tool()
        async def my_tool(x: int) -> int:
            return x + 1

        tools = server.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "my_tool"

    @pytest.mark.asyncio
    async def test_fastmcp_stub_call_tool(self):
        from cryptotrader.mcp.compat import FastMCP

        server = FastMCP("test")

        @server.tool()
        async def add(a: int, b: int) -> int:
            return a + b

        result = await server.call_tool("add", {"a": 2, "b": 3})
        assert result == 5

    @pytest.mark.asyncio
    async def test_fastmcp_stub_call_unknown_tool(self):
        from cryptotrader.mcp.compat import FastMCP

        server = FastMCP("test")
        with pytest.raises(KeyError):
            await server.call_tool("nope", {})
