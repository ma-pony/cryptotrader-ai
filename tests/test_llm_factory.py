"""Tests for LLM factory — provider dispatch, retry, fallback chains."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.llm.errors import LLMProvidersExhaustedError
from cryptotrader.llm.factory import (
    _build_provider_llm,
    _is_retryable,
    _wrap_with_retry,
    build_resilient_llm,
)
from cryptotrader.llm.registry import ModelRoleConfig, ModelsManifest, ProviderEntry


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


@dataclass
class _RetryConfig:
    max_attempts: int = 2
    retry_base_delay_s: float = 0.01
    retry_backoff_factor: float = 1.0
    retry_jitter: bool = False


class TestIsRetryable:
    def test_rate_limit_is_retryable(self):
        from openai import RateLimitError

        exc = RateLimitError("rate limit", response=MagicMock(status_code=429), body=None)
        assert _is_retryable(exc) is True

    def test_auth_error_not_retryable(self):
        from openai import AuthenticationError

        exc = AuthenticationError("bad key", response=MagicMock(status_code=401), body=None)
        assert _is_retryable(exc) is False

    def test_bad_request_not_retryable(self):
        from openai import BadRequestError

        exc = BadRequestError("bad", response=MagicMock(status_code=400), body=None)
        assert _is_retryable(exc) is False

    def test_timeout_is_retryable(self):
        from openai import APITimeoutError

        exc = APITimeoutError(request=MagicMock())
        assert _is_retryable(exc) is True

    def test_connection_error_is_retryable(self):
        from openai import APIConnectionError

        exc = APIConnectionError(request=MagicMock())
        assert _is_retryable(exc) is True

    def test_server_error_retryable(self):
        exc = Exception("server error")
        exc.status_code = 500
        assert _is_retryable(exc) is True

    def test_502_retryable(self):
        exc = Exception("bad gateway")
        exc.status_code = 502
        assert _is_retryable(exc) is True

    def test_generic_exception_not_retryable(self):
        assert _is_retryable(ValueError("nope")) is False


class TestBuildProviderLLM:
    @patch("langchain_openai.ChatOpenAI")
    def test_openai_provider(self, mock_cls):
        entry = ProviderEntry(model_id="openai/gpt-4o", provider_type="openai", api_key_env="OPENAI_API_KEY")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            _build_provider_llm(entry, temperature=0.2, timeout=120)
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["api_key"] == "sk-test"

    @patch("langchain_openai.ChatOpenAI")
    def test_openai_compatible_with_base_url(self, mock_cls):
        entry = ProviderEntry(
            model_id="custom/model",
            provider_type="openai_compatible",
            base_url="https://custom.api.com/v1",
            api_key_env="CUSTOM_KEY",
        )
        with patch.dict("os.environ", {"CUSTOM_KEY": "ck-test"}):
            _build_provider_llm(entry, temperature=0.5, timeout=60)
        kwargs = mock_cls.call_args[1]
        assert kwargs["base_url"] == "https://custom.api.com/v1"

    @pytest.mark.skipif(
        not _has_module("langchain_anthropic"),
        reason="langchain_anthropic not installed",
    )
    @patch("langchain_anthropic.ChatAnthropic")
    def test_anthropic_provider(self, mock_cls):
        entry = ProviderEntry(
            model_id="anthropic/claude-sonnet",
            provider_type="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
        )
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ak-test"}):
            _build_provider_llm(entry, temperature=0.0, timeout=90)
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "claude-sonnet"

    @pytest.mark.skipif(
        not _has_module("langchain_google_genai"),
        reason="langchain_google_genai not installed",
    )
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_google_provider(self, mock_cls):
        entry = ProviderEntry(model_id="google/gemini-pro", provider_type="google", api_key_env="GOOGLE_API_KEY")
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-test"}):
            _build_provider_llm(entry, temperature=0.3, timeout=120)
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "gemini-pro"
        assert kwargs["google_api_key"] == "gk-test"

    def test_unknown_provider_raises(self):
        entry = ProviderEntry(model_id="x/y", provider_type="unknown_provider")
        with pytest.raises(ValueError, match="Unknown provider_type"):
            _build_provider_llm(entry, temperature=0.2, timeout=120)

    @patch("langchain_openai.ChatOpenAI")
    def test_json_mode(self, mock_cls):
        entry = ProviderEntry(model_id="openai/gpt-4o", provider_type="openai")
        _build_provider_llm(entry, temperature=0.2, timeout=120, json_mode=True)
        kwargs = mock_cls.call_args[1]
        assert kwargs["model_kwargs"]["response_format"]["type"] == "json_object"

    @patch("langchain_openai.ChatOpenAI")
    def test_model_id_slash_stripping(self, mock_cls):
        entry = ProviderEntry(model_id="openai/gpt-4o-mini", provider_type="openai")
        _build_provider_llm(entry, temperature=0.2, timeout=120)
        assert mock_cls.call_args[1]["model"] == "gpt-4o-mini"


class TestWrapWithRetry:
    async def test_retry_on_retryable_error(self):
        from openai import RateLimitError

        mock_llm = MagicMock()
        call_count = 0

        async def failing_then_ok(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError("rate limit", response=MagicMock(status_code=429), body=None)
            return "success"

        mock_llm.ainvoke = failing_then_ok
        wrapped = _wrap_with_retry(mock_llm, _RetryConfig())
        result = await wrapped.ainvoke("test")
        assert result == "success"
        assert call_count == 2

    async def test_no_retry_on_auth_error(self):
        from openai import AuthenticationError

        mock_llm = MagicMock()

        async def auth_fail(*args, **kwargs):
            raise AuthenticationError("bad key", response=MagicMock(status_code=401), body=None)

        mock_llm.ainvoke = auth_fail
        wrapped = _wrap_with_retry(mock_llm, _RetryConfig())
        with pytest.raises(AuthenticationError):
            await wrapped.ainvoke("test")


class TestBuildResilientLLM:
    @patch("cryptotrader.llm.factory._build_provider_llm")
    def test_single_provider(self, mock_build):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value="ok")
        mock_build.return_value = mock_llm

        entry = ProviderEntry(model_id="p1", provider_type="openai")
        manifest = ModelsManifest(providers={"p1": entry})
        role_cfg = ModelRoleConfig(primary_model="p1", provider_chain=["p1"])

        result = build_resilient_llm(role_cfg, manifest, 0.2, 120, False, _RetryConfig())
        assert result is not None

    def test_multi_provider_chain(self):
        from langchain_openai import ChatOpenAI

        llm1 = ChatOpenAI(model="gpt-4o-mini", api_key="sk-fake1")
        llm2 = ChatOpenAI(model="gpt-4o-mini", api_key="sk-fake2")

        with patch("cryptotrader.llm.factory._build_provider_llm", side_effect=[llm1, llm2]):
            manifest = ModelsManifest(
                providers={
                    "p1": ProviderEntry(model_id="p1", provider_type="openai"),
                    "p2": ProviderEntry(model_id="p2", provider_type="openai"),
                },
            )
            role_cfg = ModelRoleConfig(primary_model="p1", provider_chain=["p1", "p2"])
            result = build_resilient_llm(role_cfg, manifest, 0.2, 120, False, _RetryConfig())
            assert result is not None

    def test_no_providers_raises(self):
        manifest = ModelsManifest()
        role_cfg = ModelRoleConfig(primary_model="missing", provider_chain=["missing"])
        with pytest.raises(LLMProvidersExhaustedError, match="All providers exhausted"):
            build_resilient_llm(role_cfg, manifest, 0.2, 120, False, _RetryConfig(), role="test")


class TestLLMProvidersExhaustedError:
    def test_message_format(self):
        err = LLMProvidersExhaustedError(
            role="analysis",
            providers_tried=["p1", "p2"],
            last_error=ValueError("fail"),
        )
        assert "analysis" in str(err)
        assert "p1, p2" in str(err)
        assert err.role == "analysis"
        assert err.providers_tried == ["p1", "p2"]
        assert isinstance(err.last_error, ValueError)

    def test_empty_providers(self):
        err = LLMProvidersExhaustedError(role="test", providers_tried=[], last_error=ValueError("x"))
        assert "none" in str(err)
