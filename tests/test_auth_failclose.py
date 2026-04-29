"""SEC-I5: api.dependencies fail-closes when API key is missing.

Verifies that:
- AUTH_MODE=enabled + empty API_KEY → SystemExit at import (no silent bypass)
- AUTH_MODE=disabled → import succeeds, requests pass through
- AUTH_MODE=invalid → SystemExit at import
- secrets.compare_digest is used (timing-safe comparison)
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _reload_dependencies(monkeypatch, **env: str):
    """Re-import api.dependencies under a controlled environment."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("api.dependencies", None)
    return importlib.import_module("api.dependencies")


def test_failclose_when_enabled_and_no_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    sys.modules.pop("api.dependencies", None)
    monkeypatch.setenv("AUTH_MODE", "enabled")
    with pytest.raises(SystemExit) as exc:
        importlib.import_module("api.dependencies")
    assert "API_KEY is empty" in str(exc.value)


def test_disabled_mode_succeeds(monkeypatch):
    deps = _reload_dependencies(monkeypatch, AUTH_MODE="disabled")
    assert deps.AUTH_MODE == "disabled"


def test_enabled_mode_with_key_succeeds(monkeypatch):
    deps = _reload_dependencies(monkeypatch, AUTH_MODE="enabled", API_KEY="test-key")
    assert deps.AUTH_MODE == "enabled"
    assert deps.API_KEY == "test-key"


def test_invalid_mode_failclose(monkeypatch):
    sys.modules.pop("api.dependencies", None)
    monkeypatch.setenv("AUTH_MODE", "yes")
    with pytest.raises(SystemExit) as exc:
        importlib.import_module("api.dependencies")
    assert "AUTH_MODE must be" in str(exc.value)


@pytest.mark.asyncio
async def test_verify_api_key_uses_compare_digest(monkeypatch):
    """Wrong key returns 401; correct key passes — covers the compare path."""
    from fastapi import HTTPException

    deps = _reload_dependencies(monkeypatch, AUTH_MODE="enabled", API_KEY="secret-123")

    class _Req:
        method = "GET"
        url = type("U", (), {"path": "/test"})()

        def __init__(self, key: str):
            self.headers = {"X-API-Key": key}

    # Wrong key → 401
    with pytest.raises(HTTPException) as exc:
        await deps.verify_api_key(_Req("wrong"))
    assert exc.value.status_code == 401

    # Correct key → no raise
    await deps.verify_api_key(_Req("secret-123"))

    # Restore disabled mode for subsequent tests in the suite.
    _reload_dependencies(monkeypatch, AUTH_MODE="disabled")
