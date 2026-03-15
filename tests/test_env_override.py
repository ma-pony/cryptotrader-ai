"""Tests for environment variable override mechanism (task 1.1).

Coverage:
- apply_env_overrides(): path parsing (double-underscore nesting), type coercion
  (bool/int/float/str), type-coercion failure -> warning + skip, immutability.
- Priority: env vars > local.toml > default.toml (end-to-end via load_config).
"""

import os
from unittest.mock import patch

import pytest

from cryptotrader.config import apply_env_overrides, load_config

# ── Path parsing ─────────────────────────────────────────────────────────────


def test_apply_env_overrides_single_level():
    """Single-level path override: CRYPTOTRADER_MODE=live -> data['mode']='live'."""
    toml_data = {"mode": "paper"}
    env = {"CRYPTOTRADER_MODE": "live"}
    with patch.dict(os.environ, env, clear=False):
        result = apply_env_overrides(toml_data)
    assert result["mode"] == "live"


def test_apply_env_overrides_nested_two_levels():
    """Two-level nested path: CRYPTOTRADER_RISK__MAX_STOP_LOSS_PCT."""
    toml_data = {"risk": {"max_stop_loss_pct": 0.05}}
    env = {"CRYPTOTRADER_RISK__MAX_STOP_LOSS_PCT": "0.08"}
    with patch.dict(os.environ, env, clear=False):
        result = apply_env_overrides(toml_data)
    assert result["risk"]["max_stop_loss_pct"] == pytest.approx(0.08)


def test_apply_env_overrides_nested_three_levels():
    """Three-level nested path: CRYPTOTRADER_RISK__LOSS__MAX_DAILY_LOSS_PCT."""
    toml_data = {"risk": {"loss": {"max_daily_loss_pct": 0.03}}}
    env = {"CRYPTOTRADER_RISK__LOSS__MAX_DAILY_LOSS_PCT": "0.01"}
    with patch.dict(os.environ, env, clear=False):
        result = apply_env_overrides(toml_data)
    assert result["risk"]["loss"]["max_daily_loss_pct"] == pytest.approx(0.01)


def test_apply_env_overrides_creates_missing_nested_keys():
    """Missing intermediate keys are auto-created."""
    toml_data: dict = {}
    env = {"CRYPTOTRADER_NEW__SECTION__KEY": "value"}
    with patch.dict(os.environ, env, clear=False):
        result = apply_env_overrides(toml_data)
    assert result["new"]["section"]["key"] == "value"


def test_apply_env_overrides_ignores_non_prefixed():
    """Environment variables without CRYPTOTRADER_ prefix are ignored."""
    toml_data = {"mode": "paper"}
    env = {"OTHER_VAR": "should_be_ignored", "CRYPTOTRADER_MODE": "live"}
    with patch.dict(os.environ, env, clear=False):
        result = apply_env_overrides(toml_data)
    assert result["mode"] == "live"
    assert "other_var" not in result


# ── Type coercion ─────────────────────────────────────────────────────────────


def test_apply_env_overrides_bool_true():
    """String 'true' (case-insensitive) is converted to Python True."""
    toml_data = {"scheduler": {"enabled": False}}
    env = {"CRYPTOTRADER_SCHEDULER__ENABLED": "true"}
    with patch.dict(os.environ, env, clear=False):
        result = apply_env_overrides(toml_data)
    assert result["scheduler"]["enabled"] is True


def test_apply_env_overrides_bool_false():
    """String 'False' (case-insensitive) is converted to Python False."""
    toml_data = {"scheduler": {"enabled": True}}
    env = {"CRYPTOTRADER_SCHEDULER__ENABLED": "False"}
    with patch.dict(os.environ, env, clear=False):
        result = apply_env_overrides(toml_data)
    assert result["scheduler"]["enabled"] is False


def test_apply_env_overrides_int():
    """Pure integer string is converted to int."""
    toml_data = {"scheduler": {"interval_minutes": 240}}
    env = {"CRYPTOTRADER_SCHEDULER__INTERVAL_MINUTES": "60"}
    with patch.dict(os.environ, env, clear=False):
        result = apply_env_overrides(toml_data)
    assert result["scheduler"]["interval_minutes"] == 60
    assert isinstance(result["scheduler"]["interval_minutes"], int)


def test_apply_env_overrides_float():
    """Decimal numeric string is converted to float."""
    toml_data = {"risk": {"max_stop_loss_pct": 0.05}}
    env = {"CRYPTOTRADER_RISK__MAX_STOP_LOSS_PCT": "0.10"}
    with patch.dict(os.environ, env, clear=False):
        result = apply_env_overrides(toml_data)
    assert result["risk"]["max_stop_loss_pct"] == pytest.approx(0.10)
    assert isinstance(result["risk"]["max_stop_loss_pct"], float)


def test_apply_env_overrides_string_retained():
    """Non-numeric, non-bool strings remain as str."""
    toml_data = {"mode": "paper"}
    env = {"CRYPTOTRADER_MODE": "live"}
    with patch.dict(os.environ, env, clear=False):
        result = apply_env_overrides(toml_data)
    assert result["mode"] == "live"
    assert isinstance(result["mode"], str)


def test_apply_env_overrides_invalid_type_logs_warning_and_skips(caplog):
    """When the original TOML value is numeric but env var cannot be parsed as a number,
    emit logger.warning and skip the key (preserve original value).
    """
    import logging

    toml_data = {"risk": {"loss": {"max_daily_loss_pct": 0.03}}}
    env = {"CRYPTOTRADER_RISK__LOSS__MAX_DAILY_LOSS_PCT": "notanumber"}
    with patch.dict(os.environ, env, clear=False), caplog.at_level(logging.WARNING):
        result = apply_env_overrides(toml_data)
    # Original value must be preserved
    assert result["risk"]["loss"]["max_daily_loss_pct"] == pytest.approx(0.03)
    # A warning must be logged
    assert any("notanumber" in r.message or "max_daily_loss_pct" in r.message.lower() for r in caplog.records)


# ── Immutability ──────────────────────────────────────────────────────────────


def test_apply_env_overrides_does_not_mutate_input():
    """apply_env_overrides must not modify the original toml_data dict."""
    toml_data = {"risk": {"loss": {"max_daily_loss_pct": 0.03}}}
    original_value = toml_data["risk"]["loss"]["max_daily_loss_pct"]
    env = {"CRYPTOTRADER_RISK__LOSS__MAX_DAILY_LOSS_PCT": "0.01"}
    with patch.dict(os.environ, env, clear=False):
        apply_env_overrides(toml_data)
    assert toml_data["risk"]["loss"]["max_daily_loss_pct"] == original_value


def test_apply_env_overrides_no_env_vars_returns_equivalent():
    """When no CRYPTOTRADER_* vars exist, result is equivalent to input."""
    toml_data = {"mode": "paper", "risk": {"max_stop_loss_pct": 0.05}}
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("CRYPTOTRADER_")}
    with patch.dict(os.environ, clean_env, clear=True):
        result = apply_env_overrides(toml_data)
    assert result["mode"] == "paper"
    assert result["risk"]["max_stop_loss_pct"] == pytest.approx(0.05)


# ── End-to-end integration ────────────────────────────────────────────────────


def test_load_config_env_override_end_to_end():
    """End-to-end: env var takes priority over TOML file values after load_config().

    Requirement 3.6: CRYPTOTRADER_* priority > file config.
    """
    import cryptotrader.config as config_module

    original_cache = config_module._cached_config
    config_module._cached_config = None

    try:
        env = {"CRYPTOTRADER_RISK__LOSS__MAX_DAILY_LOSS_PCT": "0.01"}
        with patch.dict(os.environ, env, clear=False):
            cfg = load_config()
        assert cfg.risk.loss.max_daily_loss_pct == pytest.approx(0.01)
    finally:
        config_module._cached_config = original_cache


def test_load_config_env_override_mode():
    """End-to-end: override top-level app field 'mode' via env var."""
    import cryptotrader.config as config_module

    original_cache = config_module._cached_config
    config_module._cached_config = None

    try:
        env = {"CRYPTOTRADER_APP__MODE": "live"}
        with patch.dict(os.environ, env, clear=False):
            cfg = load_config()
        assert cfg.mode == "live"
    finally:
        config_module._cached_config = original_cache


def test_load_config_env_override_bool():
    """End-to-end: override boolean field via env var."""
    import cryptotrader.config as config_module

    original_cache = config_module._cached_config
    config_module._cached_config = None

    try:
        env = {"CRYPTOTRADER_SCHEDULER__ENABLED": "true"}
        with patch.dict(os.environ, env, clear=False):
            cfg = load_config()
        assert cfg.scheduler.enabled is True
    finally:
        config_module._cached_config = original_cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
