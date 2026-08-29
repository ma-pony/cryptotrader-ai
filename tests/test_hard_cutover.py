"""Static guard for the one-way database/venue/book runtime cutover."""

from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_RUNTIME_PATTERNS = (
    r"\bload_config\b",
    r"\bAppConfig\b",
    r"\bExchangeCredentials\b",
    r"\bExchangesConfig\b",
    r"\bCRYPTOTRADER_",
    r"config/(default|local)\.toml",
    r"\bcryptotrader\.config\b",
    r"config/models\.toml",
    r"\bmodels\.toml\b",
    r"\bapi_key_env\b",
    r"\bcryptotrader\.llm\.registry\b",
    r"\bcryptotrader\.llm\.factory\b",
    r"\b_try_manifest_llm\b",
    r"\bexchange_id\b",
    r"\bLiveExchange\b",
    r"\bsupports_protection_orders\b",
)

SCAN_PATHS = ("src", "tests", "scripts", "config", "web/src", "Dockerfile", "docker-compose.yml", "pyproject.toml")


def scan_paths(paths: tuple[str, ...], patterns: tuple[str, ...]) -> list[str]:
    compiled = re.compile("|".join(f"(?:{pattern})" for pattern in patterns))
    violations: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            if (
                not candidate.is_file()
                or candidate.suffix == ".lock"
                or "__pycache__" in candidate.parts
                or candidate.resolve() == Path(__file__).resolve()
            ):
                continue
            for line_number, line in enumerate(candidate.read_text(errors="replace").splitlines(), start=1):
                if compiled.search(line):
                    violations.append(f"{candidate}:{line_number}: {line.strip()}")
    return violations


def test_forbidden_symbols_are_absent_from_runtime_tests_docker_and_web():
    assert scan_paths(SCAN_PATHS, FORBIDDEN_RUNTIME_PATTERNS) == []


def test_config_directory_contains_no_runtime_toml():
    assert not Path("config/default.toml").exists()
    assert not Path("config/local.toml").exists()
    assert not tuple(Path("config").glob("*.toml"))


def test_removed_legacy_production_modules_do_not_exist():
    for path in (
        "src/cryptotrader/config.py",
        "src/cryptotrader/execution/exchange.py",
        "src/cryptotrader/execution/simulator.py",
        "src/cryptotrader/portfolio/exchange_reader.py",
        "src/cryptotrader/profiles/repository.py",
        "src/api/routes/signal_profile.py",
        "src/cryptotrader/hitl/gate.py",
        "tests/test_hitl_gate.py",
        "web/src/hooks/use-signal-profile.ts",
    ):
        assert not Path(path).exists()


def test_container_runtime_bootstrap_is_exactly_database_and_master_key():
    compose = Path("docker-compose.yml").read_text()
    assert "env_file:" not in compose
    assert "DATABASE_URL:" in compose
    assert "CONFIG_MASTER_KEY:" in compose
