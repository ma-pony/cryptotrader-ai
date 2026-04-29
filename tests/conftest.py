"""Shared test fixtures.

Resets per-process API rate-limit state between tests so that the in-process
sliding-window limiter (``api.main._rate_buckets``) does not leak across
unrelated test functions.  Without this, large suites of TestClient calls from
the same client host (``testclient``) fail with 429 once the 60/min window
fills.
"""

from __future__ import annotations

import os

import pytest

# SEC-I5: api.dependencies fail-closes when AUTH_MODE=enabled (default) and
# API_KEY is empty. Tests bypass auth via TestClient without setting API_KEY,
# so we explicitly opt the test environment into AUTH_MODE=disabled before any
# `api.main` import collects this module. pytest itself does not read AUTH_MODE,
# so it is safe to import pytest first.
os.environ.setdefault("AUTH_MODE", "disabled")


@pytest.fixture(autouse=True)
def _reset_api_rate_limiter() -> None:
    """Clear API rate-limit buckets, backtest run state, and health caches before each test."""
    try:
        from api.main import _rate_buckets

        _rate_buckets.clear()
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
