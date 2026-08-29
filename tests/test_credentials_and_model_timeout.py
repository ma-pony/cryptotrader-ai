"""Task 1.3: Unit tests for missing credential detection and ModelConfig timeout field.

Coverage:
- ModelConfig.timeout_seconds field exists with default value 60
- _build_config correctly parses models.timeout_seconds from TOML
- arena live-check iterates all configured exchanges, outputs clear message on missing api_key/secret
- _check_credentials returns correct failure message (not a KeyError)

Requirements: 3.4, 8.7
"""

from __future__ import annotations

from cryptotrader.config import (
    ModelConfig,
    _build_config,
)

# -- ModelConfig.timeout_seconds --


def test_model_config_has_timeout_seconds():
    """ModelConfig must have timeout_seconds field with default value 90."""
    cfg = ModelConfig()
    assert hasattr(cfg, "timeout_seconds")
    assert cfg.timeout_seconds == 90


def test_model_config_timeout_seconds_is_int():
    """ModelConfig.timeout_seconds must be int type."""
    cfg = ModelConfig()
    assert isinstance(cfg.timeout_seconds, int)


def test_model_config_timeout_seconds_custom():
    """ModelConfig allows custom timeout_seconds."""
    cfg = ModelConfig(timeout_seconds=120)
    assert cfg.timeout_seconds == 120


def test_build_config_parses_model_timeout():
    """_build_config reads models.timeout_seconds from TOML dict."""
    toml_data = {"models": {"fallback": "gpt-4o-mini", "timeout_seconds": 90}}
    cfg = _build_config(toml_data)
    assert cfg.models.timeout_seconds == 90


def test_build_config_model_timeout_default():
    """_build_config uses default 90 when TOML lacks timeout_seconds."""
    toml_data = {"models": {"fallback": "gpt-4o-mini"}}
    cfg = _build_config(toml_data)
    assert cfg.models.timeout_seconds == 90


# -- RetryConfig env var overrides --


def test_retry_max_attempts_env_override():
    """CRYPTOTRADER_LLM__RETRY__MAX_ATTEMPTS overrides default."""
    from unittest.mock import patch as mock_patch

    from cryptotrader.config import _build_config, apply_env_overrides

    with mock_patch.dict("os.environ", {"CRYPTOTRADER_LLM__RETRY__MAX_ATTEMPTS": "5"}):
        cfg = _build_config(apply_env_overrides({}))
    assert cfg.llm.retry.max_attempts == 5


def test_retry_base_delay_env_override():
    """CRYPTOTRADER_LLM__RETRY__RETRY_BASE_DELAY_S overrides default."""
    from unittest.mock import patch as mock_patch

    from cryptotrader.config import _build_config, apply_env_overrides

    with mock_patch.dict("os.environ", {"CRYPTOTRADER_LLM__RETRY__RETRY_BASE_DELAY_S": "2.0"}):
        cfg = _build_config(apply_env_overrides({}))
    assert cfg.llm.retry.retry_base_delay_s == 2.0
