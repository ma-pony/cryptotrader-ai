"""LLM factory — provider dispatch, retry middleware, resilient chain builder."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import structlog

from cryptotrader.llm.errors import LLMProvidersExhaustedError

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from cryptotrader.config import RetryConfig
    from cryptotrader.llm.registry import ModelRoleConfig, ModelsManifest, ProviderEntry

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)


def _build_provider_llm(
    entry: ProviderEntry,
    temperature: float,
    timeout: int,
    json_mode: bool = False,
) -> BaseChatModel:
    """Instantiate the correct LangChain chat model for a provider entry."""
    pt = entry.provider_type
    model_name = entry.model_id.split("/", 1)[-1] if "/" in entry.model_id else entry.model_id

    api_key = ""
    if entry.api_key_env:
        api_key = os.environ.get(entry.api_key_env, "")

    if pt in ("openai", "openai_compatible"):
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "timeout": timeout,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if entry.base_url:
            kwargs["base_url"] = entry.base_url
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    if pt == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "timeout": timeout,
        }
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    if pt == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "timeout": timeout,
        }
        if api_key:
            kwargs["google_api_key"] = api_key
        return ChatGoogleGenerativeAI(**kwargs)

    raise ValueError(f"Unknown provider_type: {pt!r}")


def _is_retryable(exc: Exception) -> bool:
    """Determine if an LLM error is retryable."""
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        RateLimitError,
    )

    if isinstance(exc, AuthenticationError | BadRequestError):
        return False
    if isinstance(exc, RateLimitError | APIConnectionError | APITimeoutError):
        return True

    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status is not None and 500 <= int(status) <= 599


class _RetryingLLMWrapper:
    """Thin wrapper that adds tenacity retry around an LLM's ainvoke.

    Subclasses the inner LLM's class dynamically so Pydantic validation
    in RunnableWithFallbacks accepts it as a Runnable. The retry decorator
    is stored outside of Pydantic-managed fields.
    """

    def __new__(cls, llm: BaseChatModel, retry_decorator: Any) -> BaseChatModel:
        """Return a copy of the LLM with ainvoke patched via __dict__ bypass."""
        import copy

        wrapped = copy.copy(llm)
        original_ainvoke = llm.ainvoke

        async def retrying_ainvoke(*args: Any, **kwargs: Any) -> Any:
            return await retry_decorator(original_ainvoke)(*args, **kwargs)

        object.__setattr__(wrapped, "ainvoke", retrying_ainvoke)
        return wrapped


def _wrap_with_retry(llm: BaseChatModel, retry_cfg: RetryConfig) -> BaseChatModel:
    """Wrap an LLM with tenacity-based exponential backoff retry.

    Respects ``Retry-After`` headers from 429 responses when available.
    """
    import tenacity

    base_wait = tenacity.wait_exponential(
        multiplier=retry_cfg.retry_backoff_factor,
        min=retry_cfg.retry_base_delay_s,
    )
    if retry_cfg.retry_jitter:
        base_wait = base_wait + tenacity.wait_random(0, retry_cfg.retry_base_delay_s * 0.5)

    wait_strategy = _RetryAfterWait(base_wait)

    retry_decorator = tenacity.retry(
        retry=tenacity.retry_if_exception(_is_retryable),
        wait=wait_strategy,
        stop=tenacity.stop_after_attempt(retry_cfg.max_attempts),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )

    return _RetryingLLMWrapper(llm, retry_decorator)  # type: ignore[return-value]


class _RetryAfterWait:
    """Wait strategy that respects Retry-After headers, falling back to base strategy."""

    def __init__(self, base_wait: Any) -> None:
        self._base = base_wait

    def __call__(self, retry_state: Any) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if exc is not None:
            from cryptotrader.llm.errors import extract_retry_after

            server_delay = extract_retry_after(exc)
            if server_delay is not None and server_delay > 0:
                _slog.info(
                    "retry_after_header",
                    delay_seconds=server_delay,
                    attempt=retry_state.attempt_number,
                )
                return min(server_delay, 60.0)
        return self._base(retry_state=retry_state)


def _log_retry_attempt(retry_state: Any) -> None:
    """Log retry attempts via structlog."""
    attempt = retry_state.attempt_number
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    error_category = "unknown"
    if exc is not None:
        from cryptotrader.llm.errors import classify_error

        error_category, _ = classify_error(exc)
    _slog.warning(
        "llm_retry",
        attempt=attempt,
        error=str(exc) if exc else "unknown",
        error_type=type(exc).__name__ if exc else "unknown",
        error_category=error_category,
    )


def build_resilient_llm(
    role_cfg: ModelRoleConfig,
    manifest: ModelsManifest,
    temperature: float,
    timeout: int,
    json_mode: bool,
    retry_cfg: RetryConfig,
    role: str = "",
) -> BaseChatModel:
    """Build a multi-provider fallback LLM chain with retry on each provider."""
    llms: list[BaseChatModel] = []
    provider_ids: list[str] = []

    for model_id in role_cfg.provider_chain:
        entry = manifest.get_provider(model_id)
        if entry is None:
            continue
        try:
            llm = _build_provider_llm(entry, temperature, timeout, json_mode)
            llm = _wrap_with_retry(llm, retry_cfg)
            llms.append(llm)
            provider_ids.append(model_id)
        except Exception:
            logger.warning("Failed to build provider %s, skipping", model_id, exc_info=True)

    if not llms:
        raise LLMProvidersExhaustedError(role=role, providers_tried=provider_ids, last_error=ValueError("no providers"))

    if len(llms) == 1:
        return _wrap_exhausted_catcher(llms[0], role, provider_ids)

    from langchain_core.runnables import RunnableWithFallbacks

    primary = llms[0]
    fallbacks = llms[1:]
    chain = RunnableWithFallbacks(
        runnable=primary,
        fallbacks=fallbacks,
        exceptions_to_handle=(Exception,),
    )
    return _wrap_exhausted_catcher(chain, role, provider_ids)


class _ExhaustedCatcherWrapper:
    """Wraps an LLM chain to convert final failures to LLMProvidersExhaustedError."""

    def __init__(self, llm: Any, role: str, provider_ids: list[str]) -> None:
        object.__setattr__(self, "_inner", llm)
        object.__setattr__(self, "_role", role)
        object.__setattr__(self, "_provider_ids", provider_ids)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await self._inner.ainvoke(*args, **kwargs)
        except LLMProvidersExhaustedError:
            raise
        except Exception as e:
            raise LLMProvidersExhaustedError(
                role=self._role,
                providers_tried=self._provider_ids,
                last_error=e,
            ) from e

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _wrap_exhausted_catcher(llm: BaseChatModel, role: str, provider_ids: list[str]) -> BaseChatModel:
    """Wrap the final chain so that total failure raises LLMProvidersExhaustedError."""
    return _ExhaustedCatcherWrapper(llm, role, provider_ids)  # type: ignore[return-value]
