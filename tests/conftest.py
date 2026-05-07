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
def _disable_scheduler_autostart_for_tests():
    """Disable scheduler.enabled before every test.

    The FastAPI lifespan autostarts ``Scheduler`` in a background task when
    ``config.scheduler.enabled`` is true. Inside ``TestClient`` the lifespan
    runs in a worker thread, where APScheduler's ``add_signal_handler`` raises
    ``ValueError: set_wakeup_fd only works in main thread`` during teardown.
    Tests that need a real scheduler attach one explicitly to ``app.state``.

    Function scope (not session) because ``test_config_validation.py`` and
    ``test_agent_integration.py`` reset ``cryptotrader.config._cached_config``
    to None to force reload, which would re-pick up ``enabled=true`` from
    ``local.toml`` and undo a session-scoped fixture.
    """
    from cryptotrader.config import load_config

    cfg = load_config()
    original = cfg.scheduler.enabled
    cfg.scheduler.enabled = False
    yield
    cfg.scheduler.enabled = original


@pytest.fixture(autouse=True)
def _isolate_agent_memory_dir(tmp_path_factory, monkeypatch):
    """Redirect spec 014 agent_memory writes to tmp during tests.

    Prevents production code paths (nodes/journal.py:journal_trade) from
    leaking case files into the real ``agent_memory/cases/`` when invoked
    from unit tests via test_nodes.py / test_llm_timeout.py / etc. Tests
    that explicitly pass ``memory_dir=tmp_path`` are unaffected.

    Each test session gets one shared tmp dir; per-test isolation is the
    test's responsibility (most tests use their own tmp_path anyway).
    """
    isolated = tmp_path_factory.mktemp("agent_memory_isolated")
    # Point production default at tmp so any leaking call writes into the
    # ephemeral session dir instead of repo-relative ``agent_memory/``.
    monkeypatch.setattr(
        "cryptotrader.agents.skills._constants.DEFAULT_AGENT_MEMORY_DIR",
        isolated,
        raising=False,
    )
    monkeypatch.setattr(
        "cryptotrader.learning.memory.DEFAULT_AGENT_MEMORY_DIR",
        isolated,
        raising=False,
    )


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
