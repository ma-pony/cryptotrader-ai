"""Runtime 唯一外部引导参数与公开装配入口。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from cryptotrader.runtime import Runtime, build_runtime


@dataclass(frozen=True)
class BootstrapSettings:
    """Build the database runtime from exactly two process-level values."""

    database_url: str
    config_master_key: str = field(repr=False)

    @classmethod
    def from_environment(cls) -> BootstrapSettings:
        database_url = os.environ.get("DATABASE_URL", "").strip()
        master_key = os.environ.get("CONFIG_MASTER_KEY", "").strip()
        if not database_url:
            raise RuntimeError("DATABASE_URL is required")
        if not master_key:
            raise RuntimeError("CONFIG_MASTER_KEY is required")
        return cls(database_url, master_key)


__all__ = ["BootstrapSettings", "Runtime", "build_runtime"]
