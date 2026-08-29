"""Shared test fixtures.

Resets per-process API rate-limit state between tests so that the in-process
sliding-window limiter (``api.main._rate_buckets``) does not leak across
unrelated test functions.  Without this, large suites of TestClient calls from
the same client host (``testclient``) fail with 429 once the 60/min window
fills.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/cryptotrader-test-runtime.db")
os.environ.setdefault("CONFIG_MASTER_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")


@pytest.fixture(autouse=True)
def _install_minimal_runtime_for_api_clients():
    """TestClient routes read an explicit database-runtime security document."""
    try:
        from api.main import app
        from cryptotrader.runtime_config.defaults import minimal_runtime_document
        from cryptotrader.runtime_config.models import RuntimeConfigSnapshot

        previous = getattr(app.state, "runtime", None)
        app.state.runtime = SimpleNamespace(
            snapshot=RuntimeConfigSnapshot(1, minimal_runtime_document(), datetime.now(UTC)),
            repository=SimpleNamespace(database_url=os.environ["DATABASE_URL"]),
            cycle=None,
        )
        yield
        app.state.runtime = previous
    except ImportError:
        yield


@pytest.fixture(autouse=True)
def _reset_okx_portfolio_cache() -> None:
    """Reset the per-process OKX live-portfolio cache between tests.

    portfolio_v2 caches a successful read for 30s to avoid pounding OKX on
    every poll. Without this reset, tests that mock ``read_portfolio_from_exchange``
    to return None see the previous test's cached dict instead of going through
    the mocked path. Added 2026-05-07 with the cache.
    """
    try:
        import api.routes.portfolio_v2 as p

        p._OKX_LAST_FAIL_AT = 0.0
        p._OKX_LAST_OK_AT = 0.0
        p._OKX_LAST_OK_RESULT = None
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_api_rate_limiter() -> None:
    """Clear API rate-limit buckets, backtest run state, and health caches before each test."""
    try:
        import api.main as api_main

        api_main._rate_buckets.clear()
        api_main._redis_client = None
    except Exception:
        pass

    try:
        from api.routes.backtest import _RUNS, _TASKS

        _RUNS.clear()
        _TASKS.clear()
    except Exception:
        pass

    try:
        # #8-RC1: /health caches Redis client + DB engine across requests.
        # Reset between tests so mocked/patched fixtures take effect on each
        # call (otherwise a previous test's success leaves a live client cached).
        from api.routes.health import _reset_health_clients

        _reset_health_clients()
    except Exception:
        pass
