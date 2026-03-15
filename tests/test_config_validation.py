"""Tests for declarative configuration validation at startup (task 1.2).

Coverage:
- ConfigurationError: is subclass of ValueError, carries field_path and expected attributes.
- validate_config(): raises ConfigurationError for out-of-range values on each guarded field.
- validate_config(): passes silently for valid config.
- load_config(): calls validate_config after building AppConfig; out-of-range env override
  causes ConfigurationError to propagate on startup.

Requirements: 3.3, 3.5
"""

import os
from unittest.mock import patch

import pytest

import cryptotrader.config as config_module
from cryptotrader.config import (
    AppConfig,
    ConfigurationError,
    DebateConfig,
    LossConfig,
    ModelConfig,
    PositionConfig,
    RiskConfig,
    validate_config,
)

# ── ConfigurationError class ──────────────────────────────────────────────────


def test_configuration_error_is_value_error():
    """ConfigurationError must be a subclass of ValueError."""
    err = ConfigurationError(field_path="risk.loss.max_daily_loss_pct", expected="value in (0, 1)")
    assert isinstance(err, ValueError)


def test_configuration_error_has_field_path():
    """ConfigurationError exposes field_path attribute."""
    err = ConfigurationError(field_path="models.fallback", expected="non-empty string")
    assert err.field_path == "models.fallback"


def test_configuration_error_has_expected():
    """ConfigurationError exposes expected attribute."""
    err = ConfigurationError(field_path="debate.consensus_skip_threshold", expected="value in (0, 1)")
    assert err.expected == "value in (0, 1)"


def test_configuration_error_str_contains_field_path():
    """ConfigurationError string representation includes the field path."""
    err = ConfigurationError(field_path="risk.position.max_single_pct", expected="value in (0, 1)")
    assert "risk.position.max_single_pct" in str(err)


# ── validate_config: valid config passes silently ─────────────────────────────


def test_validate_config_passes_for_valid_defaults():
    """validate_config() must not raise for a default AppConfig (all constraints satisfied)."""
    cfg = AppConfig()
    validate_config(cfg)  # must not raise


def test_validate_config_passes_for_edge_case_valid_values():
    """Values just inside the open interval (0, 1) must pass validation."""
    cfg = AppConfig(
        risk=RiskConfig(
            loss=LossConfig(max_daily_loss_pct=0.001),
            position=PositionConfig(max_single_pct=0.999),
        ),
        debate=DebateConfig(consensus_skip_threshold=0.001),
        models=ModelConfig(fallback="gpt-4o-mini"),
    )
    validate_config(cfg)  # must not raise


# ── validate_config: max_daily_loss_pct ──────────────────────────────────────


def test_validate_config_rejects_max_daily_loss_pct_zero():
    """max_daily_loss_pct == 0 is out of open interval (0, 1)."""
    cfg = AppConfig(risk=RiskConfig(loss=LossConfig(max_daily_loss_pct=0.0)))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.field_path == "risk.loss.max_daily_loss_pct"


def test_validate_config_rejects_max_daily_loss_pct_one():
    """max_daily_loss_pct == 1 is out of open interval (0, 1)."""
    cfg = AppConfig(risk=RiskConfig(loss=LossConfig(max_daily_loss_pct=1.0)))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.field_path == "risk.loss.max_daily_loss_pct"


def test_validate_config_rejects_max_daily_loss_pct_negative():
    """Negative max_daily_loss_pct is out of range."""
    cfg = AppConfig(risk=RiskConfig(loss=LossConfig(max_daily_loss_pct=-0.01)))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.field_path == "risk.loss.max_daily_loss_pct"


def test_validate_config_rejects_max_daily_loss_pct_above_one():
    """max_daily_loss_pct > 1 is out of range."""
    cfg = AppConfig(risk=RiskConfig(loss=LossConfig(max_daily_loss_pct=1.5)))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.field_path == "risk.loss.max_daily_loss_pct"


# ── validate_config: max_single_pct ──────────────────────────────────────────


def test_validate_config_rejects_max_single_pct_zero():
    """max_single_pct == 0 is out of open interval (0, 1)."""
    cfg = AppConfig(risk=RiskConfig(position=PositionConfig(max_single_pct=0.0)))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.field_path == "risk.position.max_single_pct"


def test_validate_config_rejects_max_single_pct_one():
    """max_single_pct == 1 is out of open interval (0, 1)."""
    cfg = AppConfig(risk=RiskConfig(position=PositionConfig(max_single_pct=1.0)))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.field_path == "risk.position.max_single_pct"


def test_validate_config_rejects_max_single_pct_negative():
    """Negative max_single_pct is out of range."""
    cfg = AppConfig(risk=RiskConfig(position=PositionConfig(max_single_pct=-0.05)))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.field_path == "risk.position.max_single_pct"


# ── validate_config: consensus_skip_threshold ────────────────────────────────


def test_validate_config_rejects_consensus_skip_threshold_zero():
    """consensus_skip_threshold == 0 is out of open interval (0, 1)."""
    cfg = AppConfig(debate=DebateConfig(consensus_skip_threshold=0.0))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.field_path == "debate.consensus_skip_threshold"


def test_validate_config_rejects_consensus_skip_threshold_one():
    """consensus_skip_threshold == 1 is out of open interval (0, 1)."""
    cfg = AppConfig(debate=DebateConfig(consensus_skip_threshold=1.0))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.field_path == "debate.consensus_skip_threshold"


def test_validate_config_rejects_consensus_skip_threshold_above_one():
    """consensus_skip_threshold > 1 is out of range."""
    cfg = AppConfig(debate=DebateConfig(consensus_skip_threshold=2.5))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.field_path == "debate.consensus_skip_threshold"


# ── validate_config: models.fallback non-empty ───────────────────────────────


def test_validate_config_rejects_empty_fallback():
    """Empty string for models.fallback must raise ConfigurationError."""
    cfg = AppConfig(models=ModelConfig(fallback=""))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.field_path == "models.fallback"


def test_validate_config_rejects_whitespace_only_fallback():
    """Whitespace-only string for models.fallback must raise ConfigurationError."""
    cfg = AppConfig(models=ModelConfig(fallback="   "))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.field_path == "models.fallback"


def test_validate_config_accepts_non_empty_fallback():
    """Non-empty fallback model name passes validation."""
    cfg = AppConfig(models=ModelConfig(fallback="deepseek-chat"))
    validate_config(cfg)  # must not raise


# ── validate_config: error carries expected field ────────────────────────────


def test_validate_config_error_expected_contains_range_description():
    """ConfigurationError.expected must contain a meaningful description for range errors."""
    cfg = AppConfig(risk=RiskConfig(loss=LossConfig(max_daily_loss_pct=0.0)))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.expected != ""


def test_validate_config_error_expected_contains_nonempty_description():
    """ConfigurationError.expected must describe the constraint for fallback error."""
    cfg = AppConfig(models=ModelConfig(fallback=""))
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(cfg)
    assert exc_info.value.expected != ""


# ── load_config integration: validate_config called after build ───────────────


def _reset_cache(original: AppConfig | None) -> None:
    config_module._cached_config = original


def test_load_config_calls_validate_config_invalid_env_override():
    """load_config must propagate ConfigurationError when env override violates a constraint.

    Sets max_daily_loss_pct to 0 via env var (out-of-range) and verifies that
    load_config raises ConfigurationError instead of returning silently.

    Requirement 3.3: startup-phase validation with non-zero exit signal.
    """
    original = config_module._cached_config
    config_module._cached_config = None
    try:
        env = {"CRYPTOTRADER_RISK__LOSS__MAX_DAILY_LOSS_PCT": "0.0"}
        with patch.dict(os.environ, env, clear=False), pytest.raises(ConfigurationError) as exc_info:
            config_module.load_config()
        assert exc_info.value.field_path == "risk.loss.max_daily_loss_pct"
    finally:
        config_module._cached_config = original


def test_load_config_calls_validate_config_invalid_fallback_env_override():
    """load_config must propagate ConfigurationError when models.fallback is overridden to empty."""
    original = config_module._cached_config
    config_module._cached_config = None
    try:
        env = {"CRYPTOTRADER_MODELS__FALLBACK": ""}
        with patch.dict(os.environ, env, clear=False), pytest.raises(ConfigurationError) as exc_info:
            config_module.load_config()
        assert exc_info.value.field_path == "models.fallback"
    finally:
        config_module._cached_config = original


def test_load_config_valid_config_does_not_raise():
    """load_config returns AppConfig without exception when config is valid (default.toml)."""
    original = config_module._cached_config
    config_module._cached_config = None
    try:
        cfg = config_module.load_config()
        assert isinstance(cfg, AppConfig)
    finally:
        config_module._cached_config = original


# ── default.toml values satisfy constraints ───────────────────────────────────


def test_default_toml_max_daily_loss_pct_in_range():
    """Default max_daily_loss_pct (0.03) must be in open interval (0, 1)."""
    cfg = config_module.load_config()
    assert 0 < cfg.risk.loss.max_daily_loss_pct < 1


def test_default_toml_max_single_pct_in_range():
    """Default max_single_pct (0.10) must be in open interval (0, 1)."""
    cfg = config_module.load_config()
    assert 0 < cfg.risk.position.max_single_pct < 1


def test_default_toml_consensus_skip_threshold_in_range():
    """Default consensus_skip_threshold (0.5) must be in open interval (0, 1)."""
    cfg = config_module.load_config()
    assert 0 < cfg.debate.consensus_skip_threshold < 1


def test_default_toml_fallback_nonempty():
    """Default models.fallback must be a non-empty string."""
    cfg = config_module.load_config()
    assert cfg.models.fallback.strip() != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
