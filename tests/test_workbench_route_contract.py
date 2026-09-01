"""Hard-cutover contract for the canonical six-entry workbench API."""

import httpx
import pytest

from api.dependencies import verify_api_key
from tests.workbench_app import ACCESS_KEY, build_workbench_app


def test_openapi_contains_canonical_routes_and_no_retired_routes():
    from api.main import app

    paths = app.openapi()["paths"]
    assert {
        "/api/analyses",
        "/api/trading-runs",
        "/api/decisions",
        "/api/accounts",
        "/api/backtest/runs",
        "/api/runtime/status",
    } <= paths.keys()
    assert {
        "/api/chat/stream",
        "/api/chat/interrupt/{session_id}",
        "/api/cycles",
        "/api/cycles/{cycle_id}",
        "/api/portfolio/snapshot",
        "/api/portfolio/equity-curve",
        "/api/risk/status",
        "/api/risk/circuit-breaker/reset",
        "/api/backtest/run",
        "/api/backtest/sessions",
        "/scheduler/status",
    }.isdisjoint(paths)


@pytest.mark.asyncio
async def test_retired_public_scheduler_status_is_not_served():
    from api.main import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/scheduler/status")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_read_only_decisions_are_available_without_trading_runtime():
    app = build_workbench_app()
    app.dependency_overrides[verify_api_key] = lambda: None
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.get("/api/decisions")
    assert response.status_code == 200
    assert response.json()["items"] == []


@pytest.mark.asyncio
async def test_read_only_decisions_still_require_configured_authentication():
    app = build_workbench_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        unauthorized = await client.get("/api/decisions")
        authorized = await client.get(
            "/api/decisions",
            headers={"X-API-Key": ACCESS_KEY},
        )
    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
