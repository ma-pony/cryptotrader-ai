"""Tests for LLM provider registry — manifest loading, caching, parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cryptotrader.llm.registry import (
    ModelRoleConfig,
    ModelsManifest,
    ProviderEntry,
    _parse_manifest,
    load_manifest,
    reset_manifest_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_manifest_cache()
    yield
    reset_manifest_cache()


class TestProviderEntry:
    def test_defaults(self):
        e = ProviderEntry(model_id="openai/gpt-4o", provider_type="openai")
        assert e.model_id == "openai/gpt-4o"
        assert e.base_url == ""
        assert e.api_key_env == ""


class TestModelsManifest:
    def test_get_role_found(self):
        role = ModelRoleConfig(primary_model="m1", provider_chain=["m1"])
        m = ModelsManifest(providers={}, roles={"analysis": role})
        assert m.get_role("analysis") is role

    def test_get_role_missing(self):
        m = ModelsManifest()
        assert m.get_role("nonexistent") is None

    def test_get_provider_found(self):
        entry = ProviderEntry(model_id="openai/gpt-4o", provider_type="openai")
        m = ModelsManifest(providers={"openai/gpt-4o": entry})
        assert m.get_provider("openai/gpt-4o") is entry

    def test_get_provider_missing(self):
        m = ModelsManifest()
        assert m.get_provider("nonexistent") is None


class TestParseManifest:
    def test_basic_parse(self):
        raw = {
            "providers": [
                {"model_id": "openai/gpt-4o", "provider_type": "openai", "api_key_env": "OPENAI_API_KEY"},
                {"model_id": "anthropic/claude", "provider_type": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
            ],
            "roles": {
                "analysis": {
                    "primary_model": "openai/gpt-4o",
                    "provider_chain": ["openai/gpt-4o", "anthropic/claude"],
                },
            },
        }
        manifest = _parse_manifest(raw)
        assert len(manifest.providers) == 2
        assert manifest.get_role("analysis").primary_model == "openai/gpt-4o"
        assert len(manifest.get_role("analysis").provider_chain) == 2

    def test_truncates_long_chain(self):
        raw = {
            "providers": [{"model_id": f"p{i}", "provider_type": "openai"} for i in range(5)],
            "roles": {
                "test": {
                    "primary_model": "p0",
                    "provider_chain": ["p0", "p1", "p2", "p3", "p4"],
                },
            },
        }
        manifest = _parse_manifest(raw)
        assert len(manifest.get_role("test").provider_chain) == 3

    def test_skips_unknown_provider_in_chain(self):
        raw = {
            "providers": [
                {"model_id": "p0", "provider_type": "openai"},
            ],
            "roles": {
                "test": {
                    "primary_model": "p0",
                    "provider_chain": ["p0", "nonexistent"],
                },
            },
        }
        manifest = _parse_manifest(raw)
        assert manifest.get_role("test").provider_chain == ["p0"]

    def test_skips_entry_without_model_id(self):
        raw = {
            "providers": [
                {"provider_type": "openai"},
                {"model_id": "valid", "provider_type": "openai"},
            ],
            "roles": {},
        }
        manifest = _parse_manifest(raw)
        assert len(manifest.providers) == 1
        assert "valid" in manifest.providers

    def test_empty_manifest(self):
        manifest = _parse_manifest({})
        assert manifest.providers == {}
        assert manifest.roles == {}


class TestLoadManifest:
    def test_missing_file_returns_none(self, tmp_path: Path):
        result = load_manifest(tmp_path / "nonexistent.toml")
        assert result is None

    def test_valid_file(self, tmp_path: Path):
        toml_content = b"""
[[providers]]
model_id = "openai/gpt-4o"
provider_type = "openai"
api_key_env = "OPENAI_API_KEY"

[roles.analysis]
primary_model = "openai/gpt-4o"
provider_chain = ["openai/gpt-4o"]
"""
        p = tmp_path / "models.toml"
        p.write_bytes(toml_content)
        result = load_manifest(p)
        assert result is not None
        assert "openai/gpt-4o" in result.providers

    def test_invalid_toml_returns_none(self, tmp_path: Path):
        p = tmp_path / "models.toml"
        p.write_text("this is not valid toml {{{}}")
        result = load_manifest(p)
        assert result is None

    def test_caching(self, tmp_path: Path):
        toml_content = b"""
[[providers]]
model_id = "test/model"
provider_type = "openai"

[roles.test]
primary_model = "test/model"
provider_chain = ["test/model"]
"""
        p = tmp_path / "models.toml"
        p.write_bytes(toml_content)
        first = load_manifest(p)
        second = load_manifest(p)
        assert first is second

    def test_default_path_when_none(self):
        with patch("cryptotrader.llm.registry.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = load_manifest(None)
            assert result is None
