"""Tests for docker-compose.yml structure (Task 9.2).

Validates:
- Service naming: api, scheduler, web, redis, postgres
- Resource limits on api/scheduler/web
- ctdata named volume mounted at /home/appuser/.cryptotrader
- DOCS_ENABLED=false env var on api service
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

COMPOSE_PATH = Path(__file__).parent.parent / "docker-compose.yml"
WEB_DOCKERFILE_PATH = Path(__file__).parent.parent / "web" / "Dockerfile"


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
    assert "web" in services, "service 'web' not found"
    assert "redis" in services, "service 'redis' not found"
    assert "postgres" in services, "service 'postgres' not found"


def test_no_legacy_app_service(compose):
    """Legacy 'app' service must not exist -- renamed to 'api'."""
    assert "app" not in compose["services"], "legacy 'app' service should be removed"


# ---- Resource limits ----


@pytest.mark.parametrize(("service_name", "expected"), [("api", "512m"), ("scheduler", "2g")])
def test_resource_limits_memory(compose, service_name, expected):
    """Each service must retain its workload-specific memory limit."""
    svc = compose["services"][service_name]
    limits = svc.get("deploy", {}).get("resources", {}).get("limits", {})
    assert limits.get("memory") == expected, (
        f"service '{service_name}' missing memory limit '{expected}', got: {limits.get('memory')}"
    )


@pytest.mark.parametrize(("service_name", "expected"), [("api", "1.0"), ("scheduler", "2.0")])
def test_resource_limits_cpus(compose, service_name, expected):
    """Each service must retain its workload-specific CPU limit."""
    svc = compose["services"][service_name]
    limits = svc.get("deploy", {}).get("resources", {}).get("limits", {})
    assert str(limits.get("cpus")) == expected, (
        f"service '{service_name}' missing cpus limit '{expected}', got: {limits.get('cpus')}"
    )


def test_web_resource_limits(compose):
    """web service must have resource limits configured."""
    svc = compose["services"]["web"]
    limits = svc.get("deploy", {}).get("resources", {}).get("limits", {})
    assert limits.get("memory"), "web service missing memory limit"
    assert limits.get("cpus"), "web service missing cpus limit"


# ---- ctdata named volume ----


def test_ctdata_volume_declared(compose):
    """Top-level named volume 'ctdata' must be declared."""
    volumes = compose.get("volumes", {})
    assert "ctdata" in volumes, "named volume 'ctdata' not declared at top level"


@pytest.mark.parametrize("service_name", ["api", "scheduler"])
def test_ctdata_volume_mounted(compose, service_name):
    """ctdata volume must be mounted at /home/appuser/.cryptotrader in api/scheduler."""
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


def test_clean_compose_config_needs_no_dotenv_and_exposes_no_browser_api_hostname(tmp_path):
    """A clean checkout must resolve to an explicit, same-origin local web stack."""
    if shutil.which("docker") is None:
        pytest.skip("docker CLI is unavailable")

    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    clean_env = os.environ.copy()
    for name in (
        "API_KEY",
        "AUTH_MODE",
        "COMPOSE_ENV_FILES",
        "POSTGRES_PASSWORD",
        "POSTGRES_USER",
        "VITE_API_BASE_URL",
    ):
        clean_env.pop(name, None)

    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(empty_env),
            "-f",
            str(COMPOSE_PATH),
            "config",
        ],
        cwd=COMPOSE_PATH.parent,
        env=clean_env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    resolved = yaml.safe_load(result.stdout)
    services = resolved["services"]
    postgres_password = services["postgres"]["environment"]["POSTGRES_PASSWORD"]
    assert postgres_password
    assert postgres_password in services["api"]["environment"]["CRYPTOTRADER_INFRASTRUCTURE__DATABASE_URL"]
    assert services["api"]["environment"]["AUTH_MODE"] == "disabled"
    assert services["api"]["environment"]["API_KEY"] == ""
    assert services["api"]["ports"][0]["host_ip"] == "127.0.0.1"
    assert services["web"]["ports"][0]["host_ip"] == "127.0.0.1"
    assert "VITE_API_BASE_URL" not in services["web"].get("environment", {})
    assert "http://api:8003" not in str(services["web"])


def test_web_healthcheck_targets_nginx_ipv4_listener():
    """Alpine resolves localhost to ::1 while this Nginx image listens on IPv4."""
    dockerfile = WEB_DOCKERFILE_PATH.read_text()

    assert "wget -q --spider http://127.0.0.1/" in dockerfile
    assert "wget -q --spider http://localhost/" not in dockerfile
