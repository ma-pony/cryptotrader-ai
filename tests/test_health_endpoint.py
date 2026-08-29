"""Health probes consume the already-built database Runtime."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _runtime(*, db_url="", redis_url="", llm_base_url=""):
    document = SimpleNamespace(
        infrastructure=SimpleNamespace(redis_url=redis_url),
        llm=SimpleNamespace(base_url=llm_base_url),
        triggers=SimpleNamespace(enabled=False),
        scheduler=SimpleNamespace(enabled=False),
    )
    return SimpleNamespace(
        snapshot=SimpleNamespace(document=document),
        repository=SimpleNamespace(database_url=db_url),
        cycle=None,
        close=AsyncMock(),
    )


@pytest.fixture(autouse=True)
def _clear_health_clients():
    from api.routes.health import _reset_health_clients

    _reset_health_clients()
    yield
    _reset_health_clients()


@pytest.fixture
def client():
    from api.main import app

    runtime = _runtime()
    with (
        patch("cryptotrader.runtime.build_runtime", new=AsyncMock(return_value=runtime)),
        TestClient(app, raise_server_exceptions=False) as test_client,
    ):
        yield test_client


def _use(client, **values):
    runtime = _runtime(**values)
    client.app.state.runtime = runtime
    return runtime


def test_health_uses_runtime_llm_base_url(client):
    _use(client, llm_base_url="https://llm.example/v1")
    with patch("api.routes.health._check_llm", new=AsyncMock(return_value="ok")) as check:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["checks"]["llm"] == "ok"
    check.assert_awaited_once_with("https://llm.example/v1", "")


def test_health_reports_unconfigured_optional_services(client):
    _use(client)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["checks"] == {
        "api": "ok",
        "redis": "not_configured",
        "db": "not_configured",
        "llm": "not_configured",
    }


def test_health_returns_503_for_unavailable_llm(client):
    _use(client, llm_base_url="https://llm.example")
    with patch("api.routes.health._check_llm", new=AsyncMock(return_value="unavailable")):
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_health_checks_runtime_database(client):
    _use(client, db_url="sqlite+aiosqlite:///runtime.db")
    connection = AsyncMock()
    engine = MagicMock()
    engine.connect.return_value = _AsyncCtx(connection)
    engine.dispose = AsyncMock()
    with patch("api.routes.health.create_async_engine", return_value=engine):
        response = client.get("/health")

    assert response.json()["checks"]["db"] == "ok"
    connection.execute.assert_awaited_once()


def test_health_checks_runtime_redis(client):
    _use(client, redis_url="redis://runtime")
    redis = AsyncMock()
    redis.ping = AsyncMock()
    with patch("api.routes.health.aioredis") as module:
        module.from_url.return_value = redis
        response = client.get("/health")

    assert response.json()["checks"]["redis"] == "ok"
    redis.ping.assert_awaited_once()


def test_health_response_includes_uptime(client):
    _use(client)

    response = client.get("/health")

    assert response.json()["checks"]["api"] == "ok"
    assert response.json()["uptime_seconds"] >= 0


@pytest.mark.parametrize("status_code", [200, 401, 404])
def test_llm_check_treats_reachable_endpoint_as_ok(status_code):
    import asyncio

    from api.routes.health import _check_llm

    response = SimpleNamespace(status_code=status_code)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=response)
    with patch("api.routes.health.httpx.AsyncClient", return_value=client):
        result = asyncio.run(_check_llm("https://llm.example", ""))

    assert result == "ok"


class _AsyncCtx:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_args):
        return None
