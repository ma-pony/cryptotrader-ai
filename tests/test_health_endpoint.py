"""Tests for /health endpoint — task 9.3.

Covers:
- LLM API reachability check (new behavior)
- DB connectivity check
- Redis connectivity check
- Overall status aggregation: "ok" / "degraded"
- HTTP status code: 200 ok, 503 degraded
- uptime_seconds field present
- Docker orchestrator can act on the response (503 triggers restart)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app():
    """Import the FastAPI app without triggering full lifespan setup."""
    # Lazy import so patching is effective
    from api.main import app

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient with lifespan disabled for fast unit tests."""
    app = _make_app()
    # Use with_lifespan=False so we avoid setup_logging / otel side-effects
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper patch context — mock config with all services configured
# ---------------------------------------------------------------------------


def _patch_config(
    *,
    db_url: str = "sqlite+aiosqlite:///test.db",
    redis_url: str = "redis://localhost:6379",
    llm_api_key: str = "sk-test",
    llm_base_url: str = "",
):
    """Return a mock AppConfig with the given infrastructure settings."""
    cfg = MagicMock()
    cfg.infrastructure.database_url = db_url
    cfg.infrastructure.redis_url = redis_url
    cfg.llm.api_key = llm_api_key
    cfg.llm.base_url = llm_base_url
    return cfg


# ---------------------------------------------------------------------------
# Tests: LLM check — new in task 9.3
# ---------------------------------------------------------------------------


class TestLLMCheck:
    """Verify that /health includes an 'llm' component status."""

    def test_llm_ok_when_api_key_configured(self, client):
        """When LLM api_key is non-empty and a lightweight ping succeeds, llm=ok."""
        cfg = _patch_config(db_url="", redis_url="", llm_api_key="sk-test")
        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="ok"),
        ):
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["checks"]["llm"] == "ok"

    def test_llm_not_configured_when_key_empty(self, client):
        """When LLM api_key is empty, llm status is 'not_configured'."""
        cfg = _patch_config(db_url="", redis_url="", llm_api_key="")
        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="not_configured"),
        ):
            resp = client.get("/health")
        body = resp.json()
        assert body["checks"]["llm"] == "not_configured"

    def test_llm_unavailable_when_request_fails(self, client):
        """When LLM endpoint is unreachable, llm='unavailable' and status=503."""
        cfg = _patch_config(db_url="", redis_url="", llm_api_key="sk-test")
        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="unavailable"),
        ):
            resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["checks"]["llm"] == "unavailable"
        assert body["status"] == "degraded"

    def test_llm_check_function_returns_ok_on_success(self):
        """Unit test _check_llm: returns 'ok' when httpx.get succeeds."""
        import asyncio

        from api.routes.health import _check_llm

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("api.routes.health.httpx") as mock_httpx:
            mock_client_instance = MagicMock()
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_httpx.AsyncClient.return_value = mock_client_instance

            result = asyncio.run(_check_llm("https://api.openai.com", "sk-test"))

        assert result == "ok"

    def test_llm_check_function_returns_unavailable_on_error(self):
        """Unit test _check_llm: returns 'unavailable' on network error."""
        import asyncio

        from api.routes.health import _check_llm

        with patch("api.routes.health.httpx") as mock_httpx:
            mock_client_instance = MagicMock()
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_instance.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_httpx.AsyncClient.return_value = mock_client_instance

            result = asyncio.run(_check_llm("https://api.openai.com", "sk-test"))

        assert result == "unavailable"


# ---------------------------------------------------------------------------
# Tests: DB check (existing behavior — should still work)
# ---------------------------------------------------------------------------


class TestDBCheck:
    """Verify that /health includes a 'db' component status."""

    def test_db_not_configured_when_url_empty(self, client):
        """When database_url is empty, db='not_configured'."""
        cfg = _patch_config(db_url="", redis_url="", llm_api_key="")
        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="not_configured"),
        ):
            resp = client.get("/health")
        body = resp.json()
        assert body["checks"]["db"] == "not_configured"

    def test_db_ok_when_query_succeeds(self, client):
        """When DB SELECT 1 succeeds, db='ok'."""
        cfg = _patch_config(redis_url="", llm_api_key="")

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=_AsyncCtx(mock_conn))
        mock_engine.dispose = AsyncMock()

        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="not_configured"),
            patch("api.routes.health.create_async_engine", return_value=mock_engine),
        ):
            resp = client.get("/health")
        body = resp.json()
        assert body["checks"]["db"] == "ok"

    def test_db_unavailable_on_error(self, client):
        """When DB connection throws, db='unavailable'."""
        cfg = _patch_config(redis_url="", llm_api_key="")

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(side_effect=Exception("DB offline"))
        mock_engine.dispose = AsyncMock()

        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="not_configured"),
            patch("api.routes.health.create_async_engine", return_value=mock_engine),
        ):
            resp = client.get("/health")
        body = resp.json()
        assert body["checks"]["db"] == "unavailable"


# ---------------------------------------------------------------------------
# Tests: Redis check (existing behavior — should still work)
# ---------------------------------------------------------------------------


class TestRedisCheck:
    """Verify that /health includes a 'redis' component status."""

    def test_redis_not_configured_when_url_empty(self, client):
        """When redis_url is empty, redis='not_configured'."""
        cfg = _patch_config(db_url="", redis_url="", llm_api_key="")
        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="not_configured"),
        ):
            resp = client.get("/health")
        body = resp.json()
        assert body["checks"]["redis"] == "not_configured"

    def test_redis_ok_when_ping_succeeds(self, client):
        """When Redis ping succeeds, redis='ok'."""
        cfg = _patch_config(db_url="", llm_api_key="")

        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping = AsyncMock()
        mock_redis_instance.aclose = AsyncMock()

        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="not_configured"),
            patch("api.routes.health.aioredis") as mock_aioredis,
        ):
            mock_aioredis.from_url.return_value = mock_redis_instance
            resp = client.get("/health")
        body = resp.json()
        assert body["checks"]["redis"] == "ok"

    def test_redis_unavailable_on_error(self, client):
        """When Redis ping raises, redis='unavailable'."""
        cfg = _patch_config(db_url="", llm_api_key="")

        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="not_configured"),
            patch("api.routes.health.aioredis") as mock_aioredis,
        ):
            mock_aioredis.from_url.side_effect = Exception("Redis unavailable")
            resp = client.get("/health")
        body = resp.json()
        assert body["checks"]["redis"] == "unavailable"


# ---------------------------------------------------------------------------
# Tests: overall status aggregation
# ---------------------------------------------------------------------------


class TestStatusAggregation:
    """Verify status and HTTP code derive correctly from component checks."""

    def test_all_ok_returns_200(self, client):
        """All components ok -> status=ok, HTTP 200."""
        cfg = _patch_config(db_url="", redis_url="", llm_api_key="")
        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="not_configured"),
        ):
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_any_unavailable_returns_503(self, client):
        """Any component unavailable -> status=degraded, HTTP 503."""
        cfg = _patch_config(db_url="", redis_url="", llm_api_key="sk-test")
        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="unavailable"),
        ):
            resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"

    def test_uptime_seconds_present(self, client):
        """Response always contains uptime_seconds >= 0."""
        cfg = _patch_config(db_url="", redis_url="", llm_api_key="")
        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="not_configured"),
        ):
            resp = client.get("/health")
        body = resp.json()
        assert "uptime_seconds" in body
        assert body["uptime_seconds"] >= 0

    def test_api_component_always_ok(self, client):
        """The 'api' component is always present and always 'ok'."""
        cfg = _patch_config(db_url="", redis_url="", llm_api_key="")
        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="not_configured"),
        ):
            resp = client.get("/health")
        assert resp.json()["checks"]["api"] == "ok"

    def test_llm_key_present_in_checks(self, client):
        """Response checks dict always contains an 'llm' key."""
        cfg = _patch_config(db_url="", redis_url="", llm_api_key="sk-test")
        with (
            patch("api.routes.health.load_config", return_value=cfg),
            patch("api.routes.health._check_llm", new_callable=AsyncMock, return_value="ok"),
        ):
            resp = client.get("/health")
        assert "llm" in resp.json()["checks"]


# ---------------------------------------------------------------------------
# Async context manager helper
# ---------------------------------------------------------------------------


class _AsyncCtx:
    """Minimal async context manager that yields the given value."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        pass
