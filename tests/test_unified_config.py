"""Test unified configuration system."""

import pytest

from cryptotrader.config import AppConfig, ExperienceConfig, ProvidersConfig, RegimeThresholdsConfig, load_config


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


def test_experience_config_parsed():
    """Test experience config is parsed from TOML."""
    config = load_config()

    assert hasattr(config, "experience")
    assert isinstance(config.experience, ExperienceConfig)
    assert config.experience.enabled is True
    assert config.experience.every_n_cycles == 20
    assert config.experience.token_budget_pct == 0.30
    assert config.experience.verify_win_rate_tolerance == 0.15


def test_regime_thresholds_parsed():
    """Test regime thresholds are parsed as nested config."""
    config = load_config()

    rt = config.experience.regime_thresholds
    assert isinstance(rt, RegimeThresholdsConfig)
    assert rt.high_funding == 0.0003
    assert rt.negative_funding == -0.0001
    assert rt.high_vol == 0.025
    assert rt.extreme_fear_fng == 25


def test_reflection_backward_compat():
    """Test config.reflection property still works as alias."""
    config = load_config()
    assert config.reflection is config.experience


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
