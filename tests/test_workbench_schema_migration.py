"""Workbench schema is installed explicitly, never by an ordinary store call."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import Depends, FastAPI

from api.dependencies import verify_api_key


async def test_clean_backtest_startup_requires_migration_without_creating_sqlite(tmp_path):
    from cryptotrader.backtest.store import BacktestStore

    path = tmp_path / "unmigrated-backtest.sqlite"
    store = BacktestStore(f"sqlite+aiosqlite:///{path}")

    with pytest.raises(RuntimeError, match="migration required"):
        await store.recover_interrupted()

    assert not path.exists()


async def test_clean_runtime_config_read_requires_migration_without_creating_sqlite(tmp_path):
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository
    from cryptotrader.runtime_config.secrets import CredentialVault

    path = tmp_path / "unmigrated-runtime.sqlite"
    repository = RuntimeConfigRepository(
        f"sqlite+aiosqlite:///{path}",
        CredentialVault(base64.urlsafe_b64encode(b"m" * 32).decode()),
    )

    with pytest.raises(RuntimeError, match="migration required"):
        await repository.get_or_create()

    assert not path.exists()


async def test_explicit_workbench_schema_migration_enables_runtime_and_backtest(tmp_path):
    from cryptotrader.backtest.store import BacktestStore
    from cryptotrader.migrations import workbench
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository
    from cryptotrader.runtime_config.secrets import CredentialVault

    path = tmp_path / "migrated-workbench.sqlite"
    database_url = f"sqlite+aiosqlite:///{path}"
    migrate = getattr(workbench, "migrate_workbench_schema", None)
    assert migrate is not None, "explicit workbench schema migration is required"

    await migrate(database_url)
    repository = RuntimeConfigRepository(
        database_url,
        CredentialVault(base64.urlsafe_b64encode(b"m" * 32).decode()),
    )

    assert (await repository.get_or_create()).revision == 1
    await BacktestStore(database_url).recover_interrupted()
    assert path.exists()


async def test_lifespan_stays_readable_and_health_reports_migration_required():
    from api import main
    from api.routes.health import router as health_router
    from cryptotrader.migrations.schema import MigrationRequired

    application = FastAPI()
    application.include_router(health_router)
    with patch("cryptotrader.runtime.build_runtime", new=AsyncMock(side_effect=MigrationRequired(("runtime_config",)))):
        async with main.lifespan(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://test"
            ) as client:
                response = await client.get("/health")

    assert response.status_code == 503
    assert response.json()["checks"]["database_schema"] == "migration_required"


async def test_partial_schema_keeps_lifespan_health_and_business_error_readable(tmp_path, monkeypatch):
    from sqlalchemy import text

    from api import main
    from api.routes.health import router as health_router
    from cryptotrader.db import get_engine
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'partial-workbench.sqlite'}"
    await migrate_workbench_schema(database_url)
    engine = await get_engine(database_url)
    async with engine.begin() as connection:
        await connection.execute(text("DROP TABLE backtest_runs"))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("CONFIG_MASTER_KEY", base64.urlsafe_b64encode(b"p" * 32).decode())

    application = FastAPI()
    application.include_router(health_router)

    @application.get("/api/protected", dependencies=[Depends(verify_api_key)])
    async def protected():
        return {"ok": True}

    async with main.lifespan(application):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as client:
            health = await client.get("/health")
            business = await client.get("/api/protected")
        assert application.state.runtime is None
        assert application.state.migration_required.missing_tables == ("backtest_runs",)

    assert health.status_code == 503
    assert health.json()["checks"]["database_schema"] == "migration_required"
    assert business.status_code == 503
    assert business.json()["detail"] == "Workbench database migration required"


async def test_recovery_migration_failure_also_yields_degraded_lifespan(tmp_path, monkeypatch):
    from api import main
    from api.routes.health import router as health_router
    from cryptotrader.migrations.schema import MigrationRequired
    from cryptotrader.migrations.workbench import migrate_workbench_schema
    from cryptotrader.runtime_config.defaults import minimal_runtime_document
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'recovery-race.sqlite'}"
    await migrate_workbench_schema(database_url)
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("CONFIG_MASTER_KEY", base64.urlsafe_b64encode(b"r" * 32).decode())
    runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(1, minimal_runtime_document(), None),
        repository=SimpleNamespace(database_url=database_url),
        backtest_service=SimpleNamespace(
            store=SimpleNamespace(recover_interrupted=AsyncMock(side_effect=MigrationRequired(("backtest_runs",))))
        ),
        close=AsyncMock(),
    )

    application = FastAPI()
    application.include_router(health_router)
    with patch("cryptotrader.runtime.build_runtime", new=AsyncMock(return_value=runtime)):
        async with main.lifespan(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://test"
            ) as client:
                response = await client.get("/health")
            assert application.state.runtime is None
            assert application.state.migration_required.missing_tables == ("backtest_runs",)

    assert response.status_code == 503
    runtime.close.assert_awaited_once()


async def test_backtest_read_reports_migration_required_as_503_without_creating_sqlite(tmp_path, monkeypatch):
    from api.main import app
    from cryptotrader.backtest.store import BacktestStore
    from cryptotrader.runtime_config.defaults import minimal_runtime_document
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot

    path = tmp_path / "unmigrated-http.sqlite"
    runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(1, minimal_runtime_document(), None),
        backtest_service=SimpleNamespace(store=BacktestStore(f"sqlite+aiosqlite:///{path}")),
    )
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.get("/api/backtest/runs")

    assert response.status_code == 503
    assert response.json()["detail"] == "Workbench database migration required"
    assert not path.exists()
