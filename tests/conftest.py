"""Shared test fixtures.

Resets per-process API rate-limit state between tests so that the in-process
sliding-window limiter (``api.main._rate_buckets``) does not leak across
unrelated test functions.  Without this, large suites of TestClient calls from
the same client host (``testclient``) fail with 429 once the 60/min window
fills.
"""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest_plugins = ("tests.test_runtime_config_api",)

_test_runtime_database: Path | None = None
if "DATABASE_URL" not in os.environ:
    _test_runtime_database = Path(tempfile.gettempdir()) / f"cryptotrader-test-runtime-{os.getpid()}.db"
    _test_runtime_database.unlink(missing_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_test_runtime_database}"
os.environ.setdefault("CONFIG_MASTER_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")


def pytest_sessionstart() -> None:
    """Explicitly migrate the one SQLite database owned by the test session."""
    if _test_runtime_database is None:
        return

    async def migrate() -> None:
        from cryptotrader.db import dispose_engine
        from cryptotrader.migrations.workbench import migrate_workbench_schema

        database_url = os.environ["DATABASE_URL"]
        await migrate_workbench_schema(database_url)
        await dispose_engine(database_url)

    asyncio.run(migrate())


def pytest_sessionfinish() -> None:
    if _test_runtime_database is not None:
        _test_runtime_database.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _install_minimal_runtime_for_api_clients():
    """TestClient routes read an explicit database-runtime security document."""
    from api.main import app
    from cryptotrader.runtime_config.defaults import minimal_runtime_document
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot

    previous = getattr(app.state, "runtime", None)
    previous_migration = getattr(app.state, "migration_required", None)
    document = minimal_runtime_document().model_copy(update={})
    app.state.runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(1, document, datetime.now(UTC)),
        repository=SimpleNamespace(database_url=os.environ["DATABASE_URL"]),
        cycle=None,
    )
    app.state.migration_required = None
    yield
    app.state.runtime = previous
    app.state.migration_required = previous_migration


@pytest.fixture(autouse=True)
def _offline_process_boundary(monkeypatch) -> None:
    """Ordinary tests fail closed before any real socket or Redis connection."""
    import api.main as api_main
    from api.routes.health import _reset_health_clients

    def deny_external_socket(*_args, **_kwargs):
        raise AssertionError("tests must inject an offline transport before network access")

    monkeypatch.setattr(socket.socket, "connect", deny_external_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_external_socket)
    monkeypatch.setattr(api_main, "_get_redis_for_rate_limit", lambda: None)
    api_main._rate_buckets.clear()
    api_main._redis_client = None
    _reset_health_clients()
