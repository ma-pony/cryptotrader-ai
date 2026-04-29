"""Tests for LLM error classification and Retry-After extraction."""

from __future__ import annotations

from cryptotrader.llm.errors import (
    LLMProvidersExhaustedError,
    classify_error,
    extract_retry_after,
)


class TestClassifyError:
    def test_providers_exhausted(self):
        exc = LLMProvidersExhaustedError(role="test", providers_tried=["a"], last_error=RuntimeError())
        cat, retryable = classify_error(exc)
        assert cat == "providers_exhausted"
        assert retryable is False

    def test_rate_limit_by_openai_type(self):
        from openai import RateLimitError

        request_mock = type("Req", (), {"url": "http://test", "method": "POST", "headers": {}})()
        response_mock = type("Resp", (), {"status_code": 429, "headers": {}, "request": request_mock})()
        exc = RateLimitError(
            message="rate limited",
            response=response_mock,
            body=None,
        )
        cat, retryable = classify_error(exc)
        assert cat == "rate_limit"
        assert retryable is True

    def test_auth_error_by_openai_type(self):
        from openai import AuthenticationError

        request_mock = type("Req", (), {"url": "http://test", "method": "POST", "headers": {}})()
        response_mock = type("Resp", (), {"status_code": 401, "headers": {}, "request": request_mock})()
        exc = AuthenticationError(
            message="invalid key",
            response=response_mock,
            body=None,
        )
        cat, retryable = classify_error(exc)
        assert cat == "auth_error"
        assert retryable is False

    def test_timeout_by_openai_type(self):
        from openai import APITimeoutError

        exc = APITimeoutError(request=type("R", (), {"url": "http://test"})())
        cat, retryable = classify_error(exc)
        assert cat == "timeout"
        assert retryable is True

    def test_status_code_429(self):
        exc = Exception("rate limited")
        exc.status_code = 429  # type: ignore[attr-defined]
        cat, retryable = classify_error(exc)
        assert cat == "rate_limit"
        assert retryable is True

    def test_status_code_500(self):
        exc = Exception("internal error")
        exc.status_code = 500  # type: ignore[attr-defined]
        cat, retryable = classify_error(exc)
        assert cat == "server_error"
        assert retryable is True

    def test_status_code_401(self):
        exc = Exception("unauthorized")
        exc.status_code = 401  # type: ignore[attr-defined]
        cat, retryable = classify_error(exc)
        assert cat == "auth_error"
        assert retryable is False

    def test_unknown_error(self):
        cat, retryable = classify_error(ValueError("some error"))
        assert cat == "unknown"
        assert retryable is False


class TestExtractRetryAfter:
    def test_headers_attribute(self):
        exc = Exception()
        exc.headers = {"retry-after": "5"}  # type: ignore[attr-defined]
        assert extract_retry_after(exc) == 5.0

    def test_retry_after_case_insensitive(self):
        exc = Exception()
        exc.headers = {"Retry-After": "10"}  # type: ignore[attr-defined]
        assert extract_retry_after(exc) == 10.0

    def test_response_headers(self):
        resp = type("Resp", (), {"headers": {"retry-after": "3.5"}})()
        exc = Exception()
        exc.response = resp  # type: ignore[attr-defined]
        assert extract_retry_after(exc) == 3.5

    def test_no_retry_after(self):
        assert extract_retry_after(ValueError("no headers")) is None

    def test_non_numeric_retry_after(self):
        exc = Exception()
        exc.headers = {"retry-after": "not-a-number"}  # type: ignore[attr-defined]
        assert extract_retry_after(exc) is None

    def test_empty_headers(self):
        exc = Exception()
        exc.headers = {}  # type: ignore[attr-defined]
        assert extract_retry_after(exc) is None
