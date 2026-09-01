"""Contract for the isolated browser-acceptance application."""

from fastapi.testclient import TestClient

from tests import workbench_app


def test_workbench_fixture_exposes_the_real_workbench_routes() -> None:
    """The browser fixture must exercise canonical product routes, not a UI stub."""

    fixture = workbench_app.build_workbench_app()
    paths = fixture.openapi()["paths"]

    assert "/api/config" in paths
    assert "/api/config/catalog" in paths
    assert "/api/analyses" in paths
    assert "/api/trading-runs" in paths
    assert "/api/accounts" in paths
    assert "/api/accounts/{connection_id}/operations/prepare" in paths
    assert "/api/components/{component_id}/evaluations" in paths
    assert "/__workbench__/clock/advance" in paths


def test_workbench_fixture_registers_only_local_extensions() -> None:
    registry = workbench_app.workbench_registry()

    assert "sample_signal" in registry.components
    assert "sample_venue" in registry.venues
    assert registry.components["sample_signal"].configuration.fields[0].key == "window"
    assert [field.key for field in registry.venues["sample_venue"].configuration.credential_fields] == [
        "access_token",
        "tenant_pin",
    ]


def test_workbench_lifespan_uses_test_key_and_temporary_sqlite() -> None:
    with TestClient(workbench_app.build_workbench_app()) as client:
        assert client.get("/api/config").status_code == 401
        headers = {"X-API-Key": workbench_app.ACCESS_KEY}
        catalog = client.get("/api/config/catalog", headers=headers)
        state = client.get("/__workbench__/state")

    assert catalog.status_code == 200
    assert any(item["id"] == "sample_signal" for item in catalog.json()["components"])
    assert any(item["id"] == "sample_venue" for item in catalog.json()["venues"])
    assert state.status_code == 200
    assert state.json()["database_kind"] == "temporary_sqlite"
    assert state.json()["venue"] == {"connect_calls": 0, "account_reads": 0, "order_writes": 0}


def test_baseline_and_extension_apps_discover_different_code_registries() -> None:
    from tests import workbench_baseline_app

    headers = {"X-API-Key": workbench_app.ACCESS_KEY}
    baseline_registry = workbench_baseline_app.registry_factory()
    extension_registry = workbench_app.workbench_registry()
    baseline = workbench_app.build_workbench_app(registry_factory=workbench_baseline_app.registry_factory)
    extension = workbench_app.build_workbench_app()

    with TestClient(baseline) as client:
        baseline_catalog = client.get("/api/config/catalog", headers=headers).json()
    with TestClient(extension) as client:
        extension_catalog = client.get("/api/config/catalog", headers=headers).json()

    assert "sample_signal" not in {item["id"] for item in baseline_catalog["components"]}
    assert "sample_venue" not in {item["id"] for item in baseline_catalog["venues"]}
    assert "sample_signal" in {item["id"] for item in extension_catalog["components"]}
    assert "sample_venue" in {item["id"] for item in extension_catalog["venues"]}
    assert set(extension_registry.components) - set(baseline_registry.components) == {"sample_signal"}
    assert set(extension_registry.venues) - set(baseline_registry.venues) == {"sample_venue"}
    assert extension_registry.market_sources == baseline_registry.market_sources
    assert {
        key: extension_registry.components[key] for key in baseline_registry.components
    } == baseline_registry.components
    assert {key: extension_registry.venues[key] for key in baseline_registry.venues} == baseline_registry.venues


def test_workbench_sample_venue_passes_the_real_read_only_connection_check() -> None:
    headers = {"X-API-Key": workbench_app.ACCESS_KEY}
    connection_id = "venue-sample-check"

    with TestClient(workbench_app.build_workbench_app()) as client:
        revision = client.get("/api/config", headers=headers).json()["revision"]
        created = client.post(
            "/api/venue-connections",
            headers=headers,
            json={
                "expected_revision": revision,
                "id": connection_id,
                "label": "Read-only sample",
                "adapter_id": "sample_venue",
                "environment": "sandbox",
                "enabled": True,
                "leverage": 1,
                "margin_mode": "cross",
                "canary_only": False,
                "parameters": {"account_code": "sample"},
            },
        )
        saved = client.put(
            f"/api/venue-connections/{connection_id}/credentials",
            headers=headers,
            json={
                "expected_revision": created.json()["revision"],
                "values": {"access_token": "sample-token", "tenant_pin": "2468"},
            },
        )
        checked = client.post(
            f"/api/venue-connections/{connection_id}/test",
            headers=headers,
            json={},
        )

    assert created.status_code == 201
    assert saved.status_code == 200
    assert checked.status_code == 200
    assert checked.json()["healthy"] is True
    assert checked.json()["error_code"] is None
    assert checked.json()["capabilities"]["account_reads"]
