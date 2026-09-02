"""Tests for docker-compose.yml structure (Task 9.2).

Validates:
- Service naming: api, web, redis, postgres
- Resource limits on api/web
- ctdata named volume mounted at /home/appuser/.cryptotrader
- API receives only the two database bootstrap variables
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
DEPLOY_WORKFLOW_PATH = Path(__file__).parent.parent / ".github" / "workflows" / "deploy.yml"


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
    assert "scheduler" not in services, "API is the only scheduler owner"
    assert "web" in services, "service 'web' not found"
    assert "redis" in services, "service 'redis' not found"
    assert "postgres" in services, "service 'postgres' not found"


def test_no_legacy_app_service(compose):
    """Legacy 'app' service must not exist -- renamed to 'api'."""
    assert "app" not in compose["services"], "legacy 'app' service should be removed"


# ---- Resource limits ----


@pytest.mark.parametrize(("service_name", "expected"), [("api", "2g")])
def test_resource_limits_memory(compose, service_name, expected):
    """Each service must retain its workload-specific memory limit."""
    svc = compose["services"][service_name]
    limits = svc.get("deploy", {}).get("resources", {}).get("limits", {})
    assert limits.get("memory") == expected, (
        f"service '{service_name}' missing memory limit '{expected}', got: {limits.get('memory')}"
    )


@pytest.mark.parametrize(("service_name", "expected"), [("api", "2.0")])
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


@pytest.mark.parametrize("service_name", ["api"])
def test_ctdata_volume_mounted(compose, service_name):
    """ctdata volume must be mounted at /home/appuser/.cryptotrader in API."""
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


@pytest.mark.parametrize("service_name", ["api"])
def test_application_services_receive_only_database_runtime_bootstrap(compose, service_name):
    """Containers must not smuggle TOML, dotenv, or mode flags into runtime."""
    environment = compose["services"][service_name]["environment"]
    assert set(environment) == {"DATABASE_URL", "CONFIG_MASTER_KEY"}
    assert "env_file" not in compose["services"][service_name]


def test_clean_compose_config_needs_no_dotenv_and_exposes_no_browser_api_hostname(tmp_path):
    """A clean checkout must resolve to an explicit, same-origin local web stack."""
    if shutil.which("docker") is None:
        pytest.skip("docker CLI is unavailable")

    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    clean_env = os.environ.copy()
    for name in (
        "COMPOSE_ENV_FILES",
        "CONFIG_MASTER_KEY",
        "POSTGRES_PASSWORD",
        "POSTGRES_USER",
        "VITE_API_BASE_URL",
    ):
        clean_env.pop(name, None)

    clean_env["CONFIG_MASTER_KEY"] = "A" * 43 + "="
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
    assert postgres_password in services["api"]["environment"]["DATABASE_URL"]
    assert set(services["api"]["environment"]) == {"DATABASE_URL", "CONFIG_MASTER_KEY"}
    assert services["api"]["ports"][0]["host_ip"] == "127.0.0.1"
    assert services["web"]["ports"][0]["host_ip"] == "127.0.0.1"
    assert "VITE_API_BASE_URL" not in services["web"].get("environment", {})
    assert "http://api:8003" not in str(services["web"])


def test_web_healthcheck_targets_nginx_ipv4_listener():
    """Alpine resolves localhost to ::1 while this Nginx image listens on IPv4."""
    dockerfile = WEB_DOCKERFILE_PATH.read_text()

    assert "wget -q --spider http://127.0.0.1/" in dockerfile
    assert "wget -q --spider http://localhost/" not in dockerfile


@pytest.mark.parametrize(
    ("service_name", "image_variable"),
    [("api", "API_IMAGE"), ("web", "WEB_IMAGE")],
)
def test_application_images_can_be_promoted_without_building_on_the_server(compose, service_name, image_variable):
    """Production selects immutable CI images while local Compose retains builds."""
    service = compose["services"][service_name]
    assert service["image"] == f"${{{image_variable}:-cryptotrader-ai-{service_name}:local}}"
    assert service["pull_policy"] == "never"
    assert "build" in service


def test_deploy_workflow_publishes_images_and_does_not_build_on_the_server():
    """The constrained VPS must only pull CI-built artifacts, never compile Torch."""
    workflow = DEPLOY_WORKFLOW_PATH.read_text()

    assert workflow.count("docker/build-push-action@") == 2
    assert "push: true" in workflow
    assert "ghcr.io/${{ github.repository_owner }}/cryptotrader-ai-api:${{ github.sha }}" in workflow
    assert "ghcr.io/${{ github.repository_owner }}/cryptotrader-ai-web:${{ github.sha }}" in workflow
    assert 'docker pull "$API_IMAGE"' in workflow
    assert 'docker pull "$WEB_IMAGE"' in workflow
    assert "docker compose up -d --no-build --remove-orphans" in workflow
    assert "docker compose build" not in workflow
    assert '[ "$old_image" != "$API_IMAGE" ]' in workflow
    assert '[ "$old_image" != "$WEB_IMAGE" ]' in workflow
    assert 'case "$old_image" in' not in workflow


def test_runtime_image_does_not_copy_legacy_configuration_files():
    dockerfile = (COMPOSE_PATH.parent / "Dockerfile").read_text()
    assert "COPY config/ config/" not in dockerfile
