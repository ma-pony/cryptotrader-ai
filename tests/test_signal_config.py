"""可插拔信号组件的 TOML 静态配置契约。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import cryptotrader.config as config_module

if TYPE_CHECKING:
    from pathlib import Path


def _write_config(tmp_path: Path, *, kronos_weight: float = 0.6, llm_weight: float = 0.4) -> Path:
    path = tmp_path / "default.toml"
    path.write_text(
        f"""
[app]
mode = "standalone"
engine = "paper"
exchange_id = "okx"

[signal_plugins]
factories = ["tests.factories.fake_signal_plugin:create_component"]

[signal_profile]
neutral_threshold = 0.20
max_target_ratio = 1.0
atr_stop_multiplier = 2.0
reward_ratio = 2.0
hitl_required = true

[[signal_profile.components]]
component_id = "kronos"
enabled = true
weight = {kronos_weight}

[[signal_profile.components]]
component_id = "llm_committee"
enabled = true
weight = {llm_weight}
""".strip(),
        encoding="utf-8",
    )
    return path


def test_config_loads_plugin_factories_and_default_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config_module, "_cached_config", None)

    cfg = config_module.load_config(_write_config(tmp_path))

    assert cfg.signal_plugins.factories == ["tests.factories.fake_signal_plugin:create_component"]
    assert [item.component_id for item in cfg.signal_profile_defaults.components] == [
        "kronos",
        "llm_committee",
    ]
    profile = cfg.signal_profile_defaults.to_profile()
    assert profile.revision == 1
    assert profile.hitl_required is True
    assert [item.weight for item in profile.components] == [0.6, 0.4]


def test_config_rejects_default_profile_weights_not_equal_to_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config_module, "_cached_config", None)

    with pytest.raises(ValueError, match=r"1\.0"):
        config_module.load_config(_write_config(tmp_path, kronos_weight=0.6, llm_weight=0.3))


def test_project_default_config_enables_both_builtin_components(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config_module, "_cached_config", None)

    cfg = config_module.load_config()

    assert cfg.signal_plugins.factories == []
    assert [(item.component_id, item.enabled, item.weight) for item in cfg.signal_profile_defaults.components] == [
        ("kronos", True, 0.6),
        ("llm_committee", True, 0.4),
    ]
