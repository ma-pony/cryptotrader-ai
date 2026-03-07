"""Test unified configuration system."""

import pytest

from cryptotrader.config import AppConfig, ProvidersConfig, load_config


def test_providers_config_with_credentials():
    """Test ProvidersConfig with OKX credentials."""
    config = ProvidersConfig(
        okx_api_key="test_key",
        okx_secret_key="test_secret",
        okx_passphrase="test_pass",
    )

    assert config.okx_api_key == "test_key"
    assert config.okx_secret_key == "test_secret"
    assert config.okx_passphrase == "test_pass"
    assert config.has_okx_credentials() is True


def test_providers_config_without_credentials():
    """Test ProvidersConfig defaults have no credentials."""
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
