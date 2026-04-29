"""Custom exceptions and error classification for the LLM resilience layer."""

from __future__ import annotations

from typing import Literal


class LLMProvidersExhaustedError(RuntimeError):
    """All providers in the fallback chain have been exhausted."""

    def __init__(
        self,
        role: str,
        providers_tried: list[str],
        last_error: Exception,
    ) -> None:
        self.role = role
        self.providers_tried = providers_tried
        self.last_error = last_error
        tried = ", ".join(providers_tried) if providers_tried else "none"
        super().__init__(f"All providers exhausted for role '{role}': tried [{tried}]")


ErrorCategory = Literal[
    "rate_limit",
    "auth_error",
    "timeout",
    "bad_request",
    "server_error",
    "connection_error",
    "providers_exhausted",
    "unknown",
]


_STATUS_MAP: dict[int, tuple[ErrorCategory, bool]] = {
    429: ("rate_limit", True),
    401: ("auth_error", False),
    400: ("bad_request", False),
}


def _classify_by_openai_type(exc: Exception) -> tuple[ErrorCategory, bool] | None:
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            RateLimitError,
        )
    except ImportError:
        return None

    _type_map: list[tuple[type, ErrorCategory, bool]] = [
        (RateLimitError, "rate_limit", True),
        (AuthenticationError, "auth_error", False),
        (BadRequestError, "bad_request", False),
        (APITimeoutError, "timeout", True),
        (APIConnectionError, "connection_error", True),
    ]
    for exc_type, cat, retryable in _type_map:
        if isinstance(exc, exc_type):
            return cat, retryable
    return None


def classify_error(exc: Exception) -> tuple[ErrorCategory, bool]:
    """Classify an LLM error into a category and retryable flag."""
    if isinstance(exc, LLMProvidersExhaustedError):
        return "providers_exhausted", False

    result = _classify_by_openai_type(exc)
    if result is not None:
        return result

    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is not None:
        code = int(status)
        if code in _STATUS_MAP:
            return _STATUS_MAP[code]
        if 500 <= code <= 599:
            return "server_error", True

    return "unknown", False


def extract_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After delay (seconds) from a rate-limit error.

    Returns None if not available.
    """
    headers = getattr(exc, "headers", None) or {}
    if not isinstance(headers, dict):
        headers = dict(headers) if hasattr(headers, "__iter__") else {}

    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        response = getattr(exc, "response", None)
        if response is not None:
            resp_headers = getattr(response, "headers", None) or {}
            raw = resp_headers.get("retry-after") or resp_headers.get("Retry-After")

    if raw is None:
        return None

    try:
        return float(raw)
    except (ValueError, TypeError):
        return None
