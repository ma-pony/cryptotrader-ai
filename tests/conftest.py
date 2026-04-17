"""Shared test fixtures.

Resets per-process API rate-limit state between tests so that the in-process
sliding-window limiter (``api.main._rate_buckets``) does not leak across
unrelated test functions.  Without this, large suites of TestClient calls from
the same client host (``testclient``) fail with 429 once the 60/min window
fills.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_api_rate_limiter() -> None:
    """Clear API rate-limit buckets and backtest run state before each test."""
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
