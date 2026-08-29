"""Retry behavior for the database-configured OpenAI-compatible gateway."""

from __future__ import annotations

from typing import Any

import structlog

_slog = structlog.get_logger(__name__)


def _is_retryable(exc: Exception) -> bool:
    """Return whether a gateway failure may succeed on another attempt."""
    from openai import APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError, RateLimitError

    if isinstance(exc, AuthenticationError | BadRequestError):
        return False
    if isinstance(exc, RateLimitError | APIConnectionError | APITimeoutError):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status is not None and 500 <= int(status) <= 599


class _RetryingLLMWrapper:
    """Copy a LangChain model while retrying its asynchronous invocations."""

    def __new__(cls, llm: Any, retry_decorator: Any) -> Any:
        import copy

        wrapped = copy.copy(llm)
        original_ainvoke = llm.ainvoke

        async def retrying_ainvoke(*args: Any, **kwargs: Any) -> Any:
            return await retry_decorator(original_ainvoke)(*args, **kwargs)

        object.__setattr__(wrapped, "ainvoke", retrying_ainvoke)
        return wrapped


class _RetryAfterWait:
    """Respect a bounded Retry-After response before using exponential backoff."""

    def __init__(self, base_wait: Any) -> None:
        self._base = base_wait

    def __call__(self, retry_state: Any) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if exc is not None:
            from cryptotrader.llm.errors import extract_retry_after

            server_delay = extract_retry_after(exc)
            if server_delay is not None and server_delay > 0:
                _slog.info("retry_after_header", delay_seconds=server_delay, attempt=retry_state.attempt_number)
                return min(server_delay, 60.0)
        return self._base(retry_state=retry_state)


def _log_retry_attempt(retry_state: Any) -> None:
    """Log retry category without exposing gateway credentials."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    error_category = "unknown"
    if exc is not None:
        from cryptotrader.llm.errors import classify_error

        error_category, _ = classify_error(exc)
    _slog.warning(
        "llm_retry",
        attempt=retry_state.attempt_number,
        error=str(exc) if exc else "unknown",
        error_type=type(exc).__name__ if exc else "unknown",
        error_category=error_category,
    )


def wrap_with_retry(llm: Any, retry_config: Any) -> Any:
    """Apply the runtime document's retry policy to one gateway model."""
    import tenacity

    base_wait = tenacity.wait_exponential(
        multiplier=retry_config.retry_backoff_factor,
        min=retry_config.retry_base_delay_s,
    )
    if retry_config.retry_jitter:
        base_wait = base_wait + tenacity.wait_random(0, retry_config.retry_base_delay_s * 0.5)
    retry_decorator = tenacity.retry(
        retry=tenacity.retry_if_exception(_is_retryable),
        wait=_RetryAfterWait(base_wait),
        stop=tenacity.stop_after_attempt(retry_config.max_attempts),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )
    return _RetryingLLMWrapper(llm, retry_decorator)
