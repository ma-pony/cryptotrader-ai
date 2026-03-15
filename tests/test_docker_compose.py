"""Tests for docker-compose.yml structure (Task 9.2).

Validates:
- Service naming: api, scheduler, dashboard, redis, postgres
- Resource limits on api/scheduler/dashboard (memory: 512m, cpus: "1.0")
- ctdata named volume mounted at /home/appuser/.cryptotrader
- DOCS_ENABLED=false env var on api service
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE_PATH = Path(__file__).parent.parent / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose() -> dict:
    """Load and parse docker-compose.yml."""
    with COMPOSE_PATH.open() as f:
        return yaml.safe_load(f)


# ---- Service naming ----


def test_required_services_exist(compose):
    """All required service names must be present."""
    services = set(compose["services"].keys())
    assert "api" in services, "service 'api' not found"
    assert "scheduler" in services, "service 'scheduler' not found"
    assert "dashboard" in services, "service 'dashboard' not found"
    assert "redis" in services, "service 'redis' not found"
    assert "postgres" in services, "service 'postgres' not found"


def test_no_legacy_app_service(compose):
    """Legacy 'app' service must not exist -- renamed to 'api'."""
    assert "app" not in compose["services"], "legacy 'app' service should be removed"


# ---- Resource limits ----


@pytest.mark.parametrize("service_name", ["api", "scheduler", "dashboard"])
def test_resource_limits_memory(compose, service_name):
    """deploy.resources.limits.memory must be 512m for api/scheduler/dashboard."""
    svc = compose["services"][service_name]
    limits = svc.get("deploy", {}).get("resources", {}).get("limits", {})
    assert limits.get("memory") == "512m", (
        f"service '{service_name}' missing memory limit '512m', got: {limits.get('memory')}"
    )


@pytest.mark.parametrize("service_name", ["api", "scheduler", "dashboard"])
def test_resource_limits_cpus(compose, service_name):
    """deploy.resources.limits.cpus must be '1.0' for api/scheduler/dashboard."""
    svc = compose["services"][service_name]
    limits = svc.get("deploy", {}).get("resources", {}).get("limits", {})
    assert str(limits.get("cpus")) == "1.0", (
        f"service '{service_name}' missing cpus limit '1.0', got: {limits.get('cpus')}"
    )


# ---- ctdata named volume ----


def test_ctdata_volume_declared(compose):
    """Top-level named volume 'ctdata' must be declared."""
    volumes = compose.get("volumes", {})
    assert "ctdata" in volumes, "named volume 'ctdata' not declared at top level"


@pytest.mark.parametrize("service_name", ["api", "scheduler", "dashboard"])
def test_ctdata_volume_mounted(compose, service_name):
    """ctdata volume must be mounted at /home/appuser/.cryptotrader in api/scheduler/dashboard."""
    svc = compose["services"][service_name]
    vol_list = svc.get("volumes", [])
    # Accept both short ("ctdata:/home/appuser/.cryptotrader") and long form
    found = False
    for entry in vol_list:
        if isinstance(entry, str):
            if entry.startswith("ctdata:") and "/home/appuser/.cryptotrader" in entry:
                found = True
                break
        elif isinstance(entry, dict) and (
            entry.get("source") == "ctdata" and entry.get("target") == "/home/appuser/.cryptotrader"
        ):
            found = True
            break
    assert found, f"service '{service_name}' missing ctdata volume mount at /home/appuser/.cryptotrader"


# ---- DOCS_ENABLED on api service ----


def test_api_docs_enabled_false(compose):
    """api service must set DOCS_ENABLED=false to disable Swagger/ReDoc in production."""
    svc = compose["services"]["api"]
    env = svc.get("environment", {})
    if isinstance(env, list):
        # List form: "DOCS_ENABLED=false"
        assert any(e.startswith("DOCS_ENABLED=false") for e in env), (
            "api service environment missing DOCS_ENABLED=false"
        )
    else:
        # Dict form
        assert str(env.get("DOCS_ENABLED", "")).lower() == "false", (
            f"api service DOCS_ENABLED should be 'false', got: {env.get('DOCS_ENABLED')}"
        )
