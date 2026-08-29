"""Database-runtime staging gate contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from staging_validate import main, run_step


class _Runtime:
    async def close(self):
        return None


def test_run_step_reports_failure_without_stopping_the_gate():
    result = run_step(2, "runtime health", lambda: (_ for _ in ()).throw(RuntimeError("down")))

    assert result.status == "FAIL"
    assert result.error == "down"
    assert result.fmt().startswith("[step 2] runtime health: FAIL")


def test_main_runs_schema_runtime_then_each_enabled_connection(monkeypatch):
    calls: list[str] = []

    async def schema():
        calls.append("schema")
        return _Runtime()

    async def runtime(_runtime):
        calls.append("runtime")

    async def connections(_runtime):
        calls.append("connections")

    monkeypatch.setattr("staging_validate._load_runtime_config", schema)
    monkeypatch.setattr("staging_validate._check_runtime_health", runtime)
    monkeypatch.setattr("staging_validate._check_enabled_connections", connections)

    assert main() == 0
    assert calls == ["schema", "runtime", "connections"]


def test_main_fails_when_no_active_runtime_configuration(monkeypatch):
    async def missing_config():
        raise RuntimeError("runtime configuration is not active")

    monkeypatch.setattr("staging_validate._load_runtime_config", missing_config)

    assert main() == 1


def test_main_fails_when_an_enabled_connection_is_unhealthy(monkeypatch):
    async def schema():
        return _Runtime()

    async def runtime(_runtime):
        return None

    async def unhealthy(_runtime):
        raise RuntimeError("connection demo-okx is unhealthy")

    monkeypatch.setattr("staging_validate._load_runtime_config", schema)
    monkeypatch.setattr("staging_validate._check_runtime_health", runtime)
    monkeypatch.setattr("staging_validate._check_enabled_connections", unhealthy)

    assert main() == 1


def test_main_prints_a_three_step_summary(monkeypatch, capsys):
    async def schema():
        return _Runtime()

    async def nothing(_runtime):
        return None

    monkeypatch.setattr("staging_validate._load_runtime_config", schema)
    monkeypatch.setattr("staging_validate._check_runtime_health", nothing)
    monkeypatch.setattr("staging_validate._check_enabled_connections", nothing)

    assert main() == 0
    output = capsys.readouterr().out
    assert "[step 1] database schema and config revision: PASS" in output
    assert "[step 2] runtime health: PASS" in output
    assert "[step 3] enabled connection health: PASS" in output


@pytest.mark.parametrize("name", ["DATABASE_URL", "CONFIG_MASTER_KEY"])
def test_staging_gate_names_only_bootstrap_environment(name):
    source = (Path(__file__).parent.parent / "scripts" / "staging_validate.py").read_text()
    assert name in source
    assert "load_dotenv" not in source
