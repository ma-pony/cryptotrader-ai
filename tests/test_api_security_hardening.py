"""Tests for task 6.2: FastAPI production hardening.

Covers:
- DOCS_ENABLED env var disables /docs and /redoc when false (default)
- DOCS_ENABLED=true enables /docs and /redoc
- RequestValidationError handler returns 422 with sanitized log (no raw body)
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers to rebuild app with specific env vars
# ---------------------------------------------------------------------------


def _build_client(env_vars: dict[str, str]) -> TestClient:
    """Import api.main with patched env vars and return a fresh TestClient.

    We must reload the module so the FastAPI() constructor picks up the new
    env var values (they are read at import time).
    """
    # Remove cached modules so reload picks up the new environment
    for mod_name in list(sys.modules):
        if mod_name == "api.main" or mod_name.startswith("api.main."):
            del sys.modules[mod_name]

    with patch.dict("os.environ", env_vars, clear=False):
        import api.main as main_mod

        return TestClient(main_mod.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests: DOCS_ENABLED controls /docs and /redoc endpoints
# ---------------------------------------------------------------------------


class TestDocsEndpoints:
    """DOCS_ENABLED env var controls Swagger UI and ReDoc availability."""

    def test_docs_disabled_by_default(self):
        """When DOCS_ENABLED is absent, /docs should return 404."""
        # Reload without DOCS_ENABLED set (or explicitly set to "false")
        for mod_name in list(sys.modules):
            if mod_name == "api.main" or mod_name.startswith("api.main."):
                del sys.modules[mod_name]

        import os

        env_without_docs = {k: v for k, v in os.environ.items() if k != "DOCS_ENABLED"}
        with patch.dict("os.environ", env_without_docs, clear=True):
            import api.main as main_mod

            client = TestClient(main_mod.app, raise_server_exceptions=False)

        r = client.get("/docs")
        assert r.status_code == 404, f"Expected 404 for /docs when DOCS_ENABLED unset, got {r.status_code}"

    def test_docs_disabled_when_false(self):
        """When DOCS_ENABLED=false, /docs and /redoc return 404."""
        client = _build_client({"DOCS_ENABLED": "false"})
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404

    def test_docs_disabled_when_false_uppercase(self):
        """DOCS_ENABLED is case-insensitive: 'False' => disabled."""
        client = _build_client({"DOCS_ENABLED": "False"})
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404

    def test_docs_enabled_when_true(self):
        """When DOCS_ENABLED=true, /docs and /redoc are accessible."""
        client = _build_client({"DOCS_ENABLED": "true"})
        # FastAPI Swagger UI page returns 200
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200

    def test_docs_enabled_when_true_uppercase(self):
        """DOCS_ENABLED=True (capital T) enables docs."""
        client = _build_client({"DOCS_ENABLED": "True"})
        assert client.get("/docs").status_code == 200

    def test_other_routes_unaffected_by_docs_disabled(self):
        """Health and metrics endpoints still work when docs are disabled."""
        client = _build_client({"DOCS_ENABLED": "false"})
        r = client.get("/health")
        assert r.status_code in (200, 503)

    def test_app_docs_url_is_none_when_disabled(self):
        """FastAPI app.docs_url should be None when docs are disabled."""
        for mod_name in list(sys.modules):
            if mod_name == "api.main" or mod_name.startswith("api.main."):
                del sys.modules[mod_name]

        with patch.dict("os.environ", {"DOCS_ENABLED": "false"}, clear=False):
            import api.main as main_mod

            assert main_mod.app.docs_url is None
            assert main_mod.app.redoc_url is None

    def test_app_docs_url_set_when_enabled(self):
        """FastAPI app.docs_url should be '/docs' when docs are enabled."""
        for mod_name in list(sys.modules):
            if mod_name == "api.main" or mod_name.startswith("api.main."):
                del sys.modules[mod_name]

        with patch.dict("os.environ", {"DOCS_ENABLED": "true"}, clear=False):
            import api.main as main_mod

            assert main_mod.app.docs_url == "/docs"
            assert main_mod.app.redoc_url == "/redoc"


# ---------------------------------------------------------------------------
# Tests: RequestValidationError handler returns 422 with sanitized log
# ---------------------------------------------------------------------------


class TestRequestValidationErrorHandler:
    """RequestValidationError must return 422 and log a sanitized request summary.

    Triggers: send a request body that cannot be parsed as JSON (Content-Type
    application/json but body is plain text).  FastAPI raises RequestValidationError
    before calling the route handler.
    """

    @pytest.fixture
    def client(self):
        """Return a TestClient using the current api.main app."""
        # Re-use the existing import if available
        if "api.main" not in sys.modules:
            import api.main  # noqa: F401
        from api.main import app

        return TestClient(app, raise_server_exceptions=False)

    def _post_invalid_json(self, client):
        """POST non-JSON bytes with application/json content-type -> 422."""
        return client.post(
            "/analyze",
            content=b"THIS IS NOT JSON",
            headers={"Content-Type": "application/json"},
        )

    def test_invalid_request_returns_422(self, client):
        """Sending malformed JSON to /analyze should yield 422."""
        r = self._post_invalid_json(client)
        assert r.status_code == 422

    def test_422_response_has_detail_field(self, client):
        """422 response body must include a 'detail' field."""
        r = self._post_invalid_json(client)
        assert r.status_code == 422
        body = r.json()
        assert "detail" in body

    def test_422_does_not_expose_raw_body(self, client):
        """422 response detail must NOT echo back raw request body."""
        sensitive_payload = b"NOT-JSON password=supersecret"
        r = client.post(
            "/analyze",
            content=sensitive_payload,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422
        # Raw body values should not appear in the response
        assert "supersecret" not in r.text

    def test_validation_error_is_logged(self, client):
        """RequestValidationError must be logged (not silently swallowed)."""
        import logging

        with patch.object(logging.getLogger("api.main"), "warning") as mock_warn:
            r = self._post_invalid_json(client)
            assert r.status_code == 422
            assert mock_warn.called, "Expected logger.warning to be called on 422"

    def test_log_contains_method_and_path(self, client):
        """The validation-error log entry must contain method and path."""
        import logging

        log_calls: list[tuple] = []

        original_warning = logging.getLogger("api.main").warning

        def capturing_warning(msg, *args, **kwargs):
            log_calls.append((msg, args, kwargs))
            original_warning(msg, *args, **kwargs)

        with patch.object(logging.getLogger("api.main"), "warning", side_effect=capturing_warning):
            self._post_invalid_json(client)

        assert log_calls, "No warning was logged for the validation error"
        # The logged message or args should mention method/path info
        all_log_text = " ".join(str(c) for c in log_calls)
        assert "POST" in all_log_text or "/analyze" in all_log_text
