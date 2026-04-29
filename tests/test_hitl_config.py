"""Tests for HITL configuration — defaults, parsing, validation."""

from __future__ import annotations

import pytest

from cryptotrader.config import (
    AppConfig,
    ConfigurationError,
    HitlConfig,
    HitlTelegramConfig,
    _build_config,
    validate_config,
)


def test_defaults():
    cfg = HitlConfig()
    assert cfg.enabled is False
    assert cfg.min_position_scale == 0.5
    assert cfg.divergence_threshold == 0.6
    assert cfg.cold_start_min_trades == 5
    assert cfg.approval_timeout_seconds == 300
    assert cfg.telegram.enabled is False
    assert cfg.telegram.bot_token == ""
    assert cfg.telegram.chat_id == ""


def test_app_config_has_hitl_default():
    cfg = AppConfig()
    assert cfg.hitl.enabled is False
    assert isinstance(cfg.hitl.telegram, HitlTelegramConfig)


def test_build_config_no_hitl_section():
    toml_data = {"models": {"fallback": "test-model"}}
    cfg = _build_config(toml_data)
    assert cfg.hitl.enabled is False
    assert cfg.hitl.min_position_scale == 0.5


def test_build_config_with_hitl_section():
    toml_data = {
        "models": {"fallback": "test-model"},
        "hitl": {
            "enabled": True,
            "min_position_scale": 0.7,
            "telegram": {"enabled": True, "bot_token": "tok", "chat_id": "123"},
        },
    }
    cfg = _build_config(toml_data)
    assert cfg.hitl.enabled is True
    assert cfg.hitl.min_position_scale == 0.7
    assert cfg.hitl.telegram.enabled is True
    assert cfg.hitl.telegram.bot_token == "tok"


def test_validate_config_hitl_enabled_valid():
    cfg = AppConfig()
    cfg.hitl.enabled = True
    cfg.hitl.min_position_scale = 0.5
    validate_config(cfg)


def test_validate_config_hitl_enabled_invalid_scale():
    cfg = AppConfig()
    cfg.hitl.enabled = True
    cfg.hitl.min_position_scale = 1.5
    with pytest.raises(ConfigurationError, match=r"hitl\.min_position_scale"):
        validate_config(cfg)


def test_validate_config_hitl_disabled_skips_validation():
    cfg = AppConfig()
    cfg.hitl.enabled = False
    cfg.hitl.min_position_scale = 99.0
    validate_config(cfg)
