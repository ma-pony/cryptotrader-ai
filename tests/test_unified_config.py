"""Test unified configuration system."""

import os

import pytest

from cryptotrader.config import AppConfig, ProvidersConfig, load_config


def test_providers_config_from_env():
    """Test ProvidersConfig loads OKX from environment."""
    # Set test env vars
    os.environ["PROVIDER_OKX_API_KEY"] = "test_key"
    os.environ["PROVIDER_OKX_SECRET_KEY"] = "test_secret"
    os.environ["PROVIDER_OKX_PASSPHRASE"] = "test_pass"

    config = ProvidersConfig()

    assert config.okx_api_key == "test_key"
    assert config.okx_secret_key == "test_secret"
    assert config.okx_passphrase == "test_pass"
    assert config.has_okx_credentials() is True
    assert config.okx_enabled is True  # Auto-enabled

    # Cleanup
    del os.environ["PROVIDER_OKX_API_KEY"]
    del os.environ["PROVIDER_OKX_SECRET_KEY"]
    del os.environ["PROVIDER_OKX_PASSPHRASE"]


def test_providers_config_without_credentials(monkeypatch, tmp_path):
    """Test ProvidersConfig without credentials."""
    # Clear env vars AND prevent .env file from injecting values
    for key in ["PROVIDER_OKX_API_KEY", "PROVIDER_OKX_SECRET_KEY", "PROVIDER_OKX_PASSPHRASE", "PROVIDER_OKX_ENABLED"]:
        monkeypatch.delenv(key, raising=False)
    # Point pydantic-settings to a non-existent .env so it won't read the real one
    monkeypatch.chdir(tmp_path)

    config = ProvidersConfig()

    assert config.okx_api_key == ""
    assert config.has_okx_credentials() is False
    assert config.okx_enabled is False


def test_app_config_includes_providers():
    """Test AppConfig includes ProvidersConfig."""
    config = AppConfig()

    assert hasattr(config, "providers")
    assert isinstance(config.providers, ProvidersConfig)
    assert config.providers.binance_audit_enabled is True
    assert config.providers.enforce_token_security is True


def test_load_config_integration():
    """Test load_config includes providers."""
    config = load_config()

    assert hasattr(config, "providers")
    assert config.providers.max_acceptable_risk == "MEDIUM"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
