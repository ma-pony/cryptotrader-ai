"""Health 和 ASGI lifespan 只消费已装配的数据库 Runtime。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _runtime(*, setup_required=False, db_url="", redis_url="", llm_base_url=""):
    document = SimpleNamespace(
        infrastructure=SimpleNamespace(redis_url=redis_url),
        llm=SimpleNamespace(base_url=llm_base_url),
        triggers=SimpleNamespace(enabled=False),
        scheduler=SimpleNamespace(enabled=False),
    )
    return SimpleNamespace(
        snapshot=SimpleNamespace(document=document, setup_required=setup_required),
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


def test_setup_required_health_is_truthful_without_dependency_probes(client):
    _use(
        client,
        setup_required=True,
        db_url="sqlite+aiosqlite:///must-not-probe.db",
        redis_url="redis://must-not-probe",
        llm_base_url="https://must-not-probe.example",
    )
    with (
        patch("api.routes.health.create_async_engine") as db,
        patch("api.routes.health.aioredis") as redis,
        patch("api.routes.health._check_llm", new=AsyncMock()) as llm,
    ):
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "setup_required"
    assert response.json()["checks"] == {"api": "ok", "runtime": "setup_required"}
    db.assert_not_called()
    redis.from_url.assert_not_called()
    llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_setup_lifespan_builds_once_skips_scheduler_and_triggers_and_closes_runtime():
    from api import main

    runtime = _runtime(setup_required=True)
    application = FastAPI()
    build = AsyncMock(return_value=runtime)
    with (
        patch("cryptotrader.runtime.build_runtime", new=build),
        patch.object(main, "_init_trigger_engine", new=AsyncMock()) as init_triggers,
        patch.object(main, "_init_scheduler", new=AsyncMock()) as init_scheduler,
        patch.object(main, "_shutdown_scheduler", new=AsyncMock()) as shutdown_scheduler,
    ):
        async with main.lifespan(application):
            assert application.state.runtime is runtime

    build.assert_awaited_once_with()
    init_triggers.assert_not_awaited()
    init_scheduler.assert_not_awaited()
    shutdown_scheduler.assert_not_awaited()
    runtime.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_active_lifespan_starts_and_stops_explicit_owners_once():
    from api import main

    runtime = _runtime()
    application = FastAPI()
    trigger = SimpleNamespace(stop=AsyncMock())

    async def init_trigger(app):
        app.state.trigger_engine = trigger

    with (
        patch("cryptotrader.runtime.build_runtime", new=AsyncMock(return_value=runtime)) as build,
        patch.object(main, "_init_trigger_engine", new=AsyncMock(side_effect=init_trigger)) as init_triggers,
        patch.object(main, "_init_scheduler", new=AsyncMock()) as init_scheduler,
        patch.object(main, "_shutdown_scheduler", new=AsyncMock()) as shutdown_scheduler,
    ):
        async with main.lifespan(application):
            assert application.state.runtime is runtime

    build.assert_awaited_once_with()
    init_triggers.assert_awaited_once_with(application)
    init_scheduler.assert_awaited_once_with(application)
    shutdown_scheduler.assert_awaited_once_with(application)
    trigger.stop.assert_awaited_once_with()
    runtime.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_active_lifespan_attempts_all_owner_shutdowns_when_scheduler_stop_fails():
    from api import main

    runtime = _runtime()
    application = FastAPI()
    trigger = SimpleNamespace(stop=AsyncMock())

    async def init_trigger(app):
        app.state.trigger_engine = trigger

    with (
        patch("cryptotrader.runtime.build_runtime", new=AsyncMock(return_value=runtime)),
        patch.object(main, "_init_trigger_engine", new=AsyncMock(side_effect=init_trigger)),
        patch.object(main, "_init_scheduler", new=AsyncMock()),
        patch.object(
            main,
            "_shutdown_scheduler",
            new=AsyncMock(side_effect=RuntimeError("scheduler stop failed")),
        ),
        pytest.raises(RuntimeError, match="scheduler stop failed"),
    ):
        async with main.lifespan(application):
            pass

    trigger.stop.assert_awaited_once_with()
    runtime.close.assert_awaited_once_with()


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
