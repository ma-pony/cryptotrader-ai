"""Provider registry — load and cache models.toml manifest."""

from __future__ import annotations

import logging
import threading

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_CHAIN_LENGTH = 3


@dataclass
class ProviderEntry:
    model_id: str
    provider_type: str
    base_url: str = ""
    api_key_env: str = ""


@dataclass
class ModelRoleConfig:
    primary_model: str
    provider_chain: list[str] = field(default_factory=list)


@dataclass
class ModelsManifest:
    providers: dict[str, ProviderEntry] = field(default_factory=dict)
    roles: dict[str, ModelRoleConfig] = field(default_factory=dict)

    def get_role(self, role: str) -> ModelRoleConfig | None:
        return self.roles.get(role)

    def get_provider(self, model_id: str) -> ProviderEntry | None:
        return self.providers.get(model_id)


_manifest_cache: ModelsManifest | None = None
_manifest_lock = threading.Lock()
_manifest_loaded = False


def load_manifest(path: Path | None = None) -> ModelsManifest | None:
    """Load models.toml manifest. Returns None if file missing or invalid.

    Result is cached in a module-level variable (thread-safe).
    """
    global _manifest_cache, _manifest_loaded
    with _manifest_lock:
        if _manifest_loaded:
            return _manifest_cache
        _manifest_loaded = True

    if path is None:
        path = Path("config/models.toml")

    if not path.exists():
        logger.debug("models.toml not found at %s, using legacy mode", path)
        return None

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except Exception:
        logger.warning("Failed to parse models.toml at %s", path, exc_info=True)
        return None

    manifest = _parse_manifest(raw)
    with _manifest_lock:
        _manifest_cache = manifest
    return manifest


def reset_manifest_cache() -> None:
    """Reset the cached manifest (for testing)."""
    global _manifest_cache, _manifest_loaded
    with _manifest_lock:
        _manifest_cache = None
        _manifest_loaded = False


def _parse_manifest(raw: dict[str, Any]) -> ModelsManifest:
    providers: dict[str, ProviderEntry] = {}
    for entry in raw.get("providers", []):
        model_id = entry.get("model_id", "")
        if not model_id:
            continue
        providers[model_id] = ProviderEntry(
            model_id=model_id,
            provider_type=entry.get("provider_type", "openai"),
            base_url=entry.get("base_url", ""),
            api_key_env=entry.get("api_key_env", ""),
        )

    roles: dict[str, ModelRoleConfig] = {}
    for role_name, role_data in raw.get("roles", {}).items():
        chain = role_data.get("provider_chain", [])

        if len(chain) > _MAX_CHAIN_LENGTH:
            logger.warning(
                "Role '%s' has %d providers in chain, truncating to %d",
                role_name,
                len(chain),
                _MAX_CHAIN_LENGTH,
            )
            chain = chain[:_MAX_CHAIN_LENGTH]

        valid_chain = []
        for model_id in chain:
            if model_id not in providers:
                logger.warning(
                    "Role '%s' references unknown provider '%s', skipping",
                    role_name,
                    model_id,
                )
                continue
            valid_chain.append(model_id)

        roles[role_name] = ModelRoleConfig(
            primary_model=role_data.get("primary_model", valid_chain[0] if valid_chain else ""),
            provider_chain=valid_chain,
        )

    return ModelsManifest(providers=providers, roles=roles)
