"""Task 1.3: Unit tests for missing credential detection and ModelConfig timeout field.

Coverage:
- ModelConfig.timeout_seconds field exists with default value 60
- _build_config correctly parses models.timeout_seconds from TOML
- arena live-check iterates all configured exchanges, outputs clear message on missing api_key/secret
- _check_credentials returns correct failure message (not a KeyError)

Requirements: 3.4, 8.7
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cryptotrader.config import (
    ExchangeCredentials,
    ExchangesConfig,
    ModelConfig,
    _build_config,
)

# -- ModelConfig.timeout_seconds --


def test_model_config_has_timeout_seconds():
    """ModelConfig must have timeout_seconds field with default value 60."""
    cfg = ModelConfig()
    assert hasattr(cfg, "timeout_seconds")
    assert cfg.timeout_seconds == 60


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
    """_build_config uses default 60 when TOML lacks timeout_seconds."""
    toml_data = {"models": {"fallback": "gpt-4o-mini"}}
    cfg = _build_config(toml_data)
    assert cfg.models.timeout_seconds == 60


# -- live-check credential detection --


def _make_config(exchanges: dict) -> MagicMock:
    """Create mock config object with specified exchange credentials."""
    ex_cfg = ExchangesConfig(_exchanges=exchanges)
    cfg = MagicMock()
    cfg.exchanges = ex_cfg
    cfg.exchange_id = "binance"
    cfg.infrastructure.redis_url = ""
    cfg.infrastructure.database_url = ""
    return cfg


def test_check_credentials_returns_fail_when_no_creds():
    """When exchange has no credentials, _check_credentials should return FAIL (not KeyError)."""
    from cli.main import _check_credentials

    cfg = _make_config({})
    name, ok, detail = _check_credentials(cfg, "binance")
    assert name == "Credentials"
    assert ok is False
    assert "binance" in detail
    # Should not raise KeyError
    assert "KeyError" not in detail


def test_check_credentials_returns_fail_when_api_key_empty():
    """When api_key is empty, _check_credentials should return FAIL."""
    from cli.main import _check_credentials

    creds = ExchangeCredentials(api_key="", secret="some_secret", sandbox=True)
    cfg = _make_config({"binance": creds})
    _name, ok, detail = _check_credentials(cfg, "binance")
    assert not ok
    assert "binance" in detail


def test_check_credentials_returns_fail_when_secret_empty():
    """When secret is empty, _check_credentials should return FAIL."""
    from cli.main import _check_credentials

    creds = ExchangeCredentials(api_key="some_key", secret="", sandbox=True)
    cfg = _make_config({"binance": creds})
    _name, ok, detail = _check_credentials(cfg, "binance")
    assert not ok
    assert "binance" in detail


def test_check_credentials_passes_with_valid_creds():
    """When api_key and secret are both non-empty, _check_credentials should return PASS."""
    from cli.main import _check_credentials

    creds = ExchangeCredentials(api_key="key123", secret="sec456", sandbox=True)
    cfg = _make_config({"binance": creds})
    _name, ok, detail = _check_credentials(cfg, "binance")
    assert ok is True
    assert "binance" in detail
    assert "SANDBOX" in detail  # sandbox=True should be noted


def test_check_credentials_no_sandbox_note_when_live():
    """When sandbox=False, detail should not contain SANDBOX."""
    from cli.main import _check_credentials

    creds = ExchangeCredentials(api_key="key123", secret="sec456", sandbox=False)
    cfg = _make_config({"binance": creds})
    _, ok, detail = _check_credentials(cfg, "binance")
    assert ok is True
    assert "SANDBOX" not in detail


def test_check_credentials_message_contains_guidance():
    """When credentials are missing, detail should contain guidance for the user."""
    from cli.main import _check_credentials

    cfg = _make_config({})
    _, ok, detail = _check_credentials(cfg, "okx")
    assert not ok
    # Message should be clear enough, containing the exchange name
    assert "okx" in detail.lower() or "okx" in detail


# -- live-check does not crash during full iteration --


def test_live_check_no_keyerror_for_unconfigured_exchange():
    """_check_credentials should not raise KeyError for unconfigured exchanges."""
    from cli.main import _check_credentials

    # Exchange has no registered credentials
    cfg = _make_config({})
    cfg.exchange_id = "binance"

    # Should not raise any exception
    _name, ok, detail = _check_credentials(cfg, "binance")
    assert not ok
    assert "KeyError" not in detail
    assert "binance" in detail


@pytest.mark.asyncio
async def test_live_check_all_exchanges_iterated_safely():
    """When multiple exchanges lack credentials, _check_credentials returns safely for each."""
    from cli.main import _check_credentials

    # Simulate multiple exchanges with no credentials
    cfg = _make_config({})
    for ex_id in ("binance", "okx", "kraken"):
        _name, ok, detail = _check_credentials(cfg, ex_id)
        assert not ok
        assert ex_id in detail or "No credentials" in detail
