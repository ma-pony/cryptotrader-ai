"""Backtest HTTP history has an explicit response contract at every GET route."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
from fastapi import FastAPI


def test_backtest_get_routes_declare_response_models():
    from api.routes.backtest import router

    models = {route.path: route.response_model for route in router.routes if "GET" in getattr(route, "methods", set())}

    assert models["/api/backtest/runs"] is not None
    assert models["/api/backtest/runs/compare"] is not None
    assert models["/api/backtest/runs/{run_id}"] is not None


async def test_backtest_detail_rejects_a_malformed_persisted_decision(tmp_path):
    from api.routes.backtest import router
    from cryptotrader.backtest.models import BacktestParams
    from cryptotrader.backtest.result import BacktestResult
    from cryptotrader.backtest.store import BacktestStore
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'malformed-backtest.sqlite'}"
    await migrate_workbench_schema(database_url)
    store = BacktestStore(database_url)
    run_id = await store.create(
        BacktestParams(pair="BTC/USDT", start="2024-01-01", end="2024-01-02"),
        None,
    )
    await store.update(
        run_id,
        "completed",
        1,
        BacktestResult(
            decisions=[
                {
                    "cycle_id": "cycle-1",
                    "status": "completed",
                    "config_revision": 1,
                    "unexpected_legacy_field": True,
                }
            ]
        ),
    )
    app = FastAPI()
    app.include_router(router)
    app.state.runtime = SimpleNamespace(backtest_service=SimpleNamespace(store=store))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/backtest/runs/{run_id}")

    assert response.status_code == 500
    assert "unexpected_legacy_field" not in response.text
