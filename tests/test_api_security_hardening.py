"""FastAPI validation and RuntimeConfig security boundary tests."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

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
    def client(self, monkeypatch):
        """Return a TestClient using the current api.main app."""
        # Re-use the existing import if available
        if "api.main" not in sys.modules:
            import api.main  # noqa: F401
        from api.main import app

        monkeypatch.setattr("api.main._get_redis_for_rate_limit", lambda: None)

        return TestClient(app, raise_server_exceptions=False)

    def _post_invalid_json(self, client):
        """POST non-JSON bytes with application/json content-type -> 422."""
        return client.post(
            "/api/backtest/runs",
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
            "/api/backtest/runs",
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
        assert "POST" in all_log_text or "/api/backtest/runs" in all_log_text


@pytest.mark.asyncio
async def test_rate_limit_uses_an_explicit_offline_redis_transport(monkeypatch):
    """The process-wide network guard must not replace dedicated fake transport coverage."""
    import api.main as api_main

    class FakeRedis:
        def __init__(self) -> None:
            self.counts: dict[str, int] = {}
            self.expirations: list[tuple[str, int]] = []

        async def incr(self, key: str) -> int:
            self.counts[key] = self.counts.get(key, 0) + 1
            return self.counts[key]

        async def expire(self, key: str, seconds: int) -> None:
            self.expirations.append((key, seconds))

    transport = FakeRedis()
    monkeypatch.setattr(api_main, "_get_redis_for_rate_limit", lambda: transport)

    assert all([await api_main._check_rate_limit("offline-test") for _ in range(api_main.RATE_LIMIT)])
    assert not await api_main._check_rate_limit("offline-test")
    assert len(transport.counts) == 1
    assert len(transport.expirations) == 1


def test_runtime_api_routes_are_api_key_protected_and_signal_profile_is_absent():
    from api.dependencies import verify_api_key
    from api.main import app

    expected = {
        "/api/config",
        "/api/config/catalog",
        "/api/config/catalog/venues/{adapter_id}",
        "/api/venue-connections",
        "/api/venue-connections/{connection_id}",
        "/api/venue-connections/{connection_id}/credentials",
        "/api/venue-connections/{connection_id}/test",
        "/api/venue-connections/{connection_id}/check",
        "/api/portfolio/books",
        "/api/portfolio/books/{book_id}",
        "/api/decisions",
        "/api/decisions/{decision_id}",
        "/api/analyses",
    }
    routes = {route.path: route for route in app.routes if route.path in expected}

    assert set(routes) == expected
    assert "/api/signal-profile" not in {route.path for route in app.routes}
    for route in routes.values():
        assert any(dependency.call is verify_api_key for dependency in route.dependant.dependencies)


def test_openapi_response_schemas_never_declare_credential_or_ciphertext_fields():  # noqa: C901
    from api.main import app

    schema = app.openapi()
    forbidden = {"api_key", "secret", "passphrase", "encrypted_payload", "credential_ref"}
    runtime_paths = {
        path: operations
        for path, operations in schema["paths"].items()
        if path.startswith(
            (
                "/api/config",
                "/api/venue-connections",
                "/api/portfolio/books",
                "/api/accounts",
                "/api/decisions",
                "/api/hitl",
            )
        )
    }

    def property_names(node, seen_refs=frozenset()):
        if not isinstance(node, dict):
            return set()
        names = set(node.get("properties", {}))
        reference = node.get("$ref")
        if reference is not None and reference not in seen_refs:
            component = reference.rsplit("/", 1)[-1]
            names |= property_names(schema["components"]["schemas"][component], seen_refs | {reference})
        for key in ("items", "anyOf", "oneOf", "allOf"):
            value = node.get(key, ())
            children = value if isinstance(value, list) else (value,)
            for child in children:
                names |= property_names(child, seen_refs)
        for child in node.get("properties", {}).values():
            names |= property_names(child, seen_refs)
        return names

    response_fields = set()
    request_fields = set()
    for operations in runtime_paths.values():
        for operation in operations.values():
            for media in operation.get("requestBody", {}).get("content", {}).values():
                request_fields |= property_names(media.get("schema", {}))
            for response in operation.get("responses", {}).values():
                for media in response.get("content", {}).values():
                    response_fields |= property_names(media.get("schema", {}))

    assert runtime_paths
    assert not forbidden & response_fields
    assert "credential_ref" not in request_fields


def test_runtime_response_openapi_has_no_free_form_object_escape_hatches():  # noqa: C901
    from api.main import app

    schema = app.openapi()
    runtime_paths = {
        path: operations
        for path, operations in schema["paths"].items()
        if path.startswith(
            (
                "/api/config",
                "/api/venue-connections",
                "/api/portfolio/books",
                "/api/decisions",
                "/api/hitl",
            )
        )
    }

    def free_form_paths(node, location: str, seen_refs=frozenset()):
        if not isinstance(node, dict):
            return []
        failures = []
        reference = node.get("$ref")
        if reference is not None and reference not in seen_refs:
            component = reference.rsplit("/", 1)[-1]
            failures.extend(
                free_form_paths(
                    schema["components"]["schemas"][component],
                    f"{location}->{component}",
                    seen_refs | {reference},
                )
            )
        additional = node.get("additionalProperties")
        if additional not in (None, False):
            failures.append(location)
        for key in ("items", "anyOf", "oneOf", "allOf"):
            value = node.get(key, ())
            children = value if isinstance(value, list) else (value,)
            for index, child in enumerate(children):
                failures.extend(free_form_paths(child, f"{location}.{key}[{index}]", seen_refs))
        for name, child in node.get("properties", {}).items():
            failures.extend(free_form_paths(child, f"{location}.{name}", seen_refs))
        return failures

    failures = []
    for path, operations in runtime_paths.items():
        for method, operation in operations.items():
            for status, response in operation.get("responses", {}).items():
                for media in response.get("content", {}).values():
                    failures.extend(free_form_paths(media.get("schema", {}), f"{method.upper()} {path} {status}"))

    assert runtime_paths
    assert failures == []
