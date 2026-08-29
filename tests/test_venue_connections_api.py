"""平台连接写入、凭据与只读连通性测试 API 契约。"""

# ruff: noqa: F401, F811 - 导入 fixture 供本模块的 pytest 参数解析。

from __future__ import annotations

from unittest.mock import ANY, AsyncMock

from cryptotrader.pair import Pair
from tests.test_runtime_config_api import ApiHarness, api_harness, put_fixture_credentials

PAIR = Pair.parse("BTC/USDT:USDT")


def create_payload(
    revision: int,
    *,
    connection_id: str = "paper-extra",
    environment: str = "paper",
) -> dict:
    return {
        "expected_revision": revision,
        "id": connection_id,
        "label": connection_id,
        "adapter_id": "paper" if environment == "paper" else "okx",
        "environment": environment,
        "enabled": True,
        "credential_ref": None if environment == "paper" else f"{connection_id}-credentials",
        "leverage": 1,
        "margin_mode": "isolated",
        "parameters": {},
    }


def update_payload(revision: int, connection: dict, **overrides) -> dict:
    return {
        "expected_revision": revision,
        "label": connection["label"],
        "adapter_id": connection["adapter_id"],
        "environment": connection["environment"],
        "enabled": connection["enabled"],
        "credential_ref": connection["credential_ref"],
        "leverage": connection["leverage"],
        "margin_mode": connection["margin_mode"],
        "parameters": connection["parameters"],
    } | overrides


async def _connection_and_revision(harness: ApiHarness, connection_id: str) -> tuple[dict, int]:
    config = (await harness.client.get("/api/config")).json()
    connection = next(item for item in config["document"]["execution"]["connections"] if item["id"] == connection_id)
    return connection, config["revision"]


async def test_connection_create_cas_writes_the_complete_document(api_harness):
    _, revision = await _connection_and_revision(api_harness, "okx-demo")

    response = await api_harness.client.post("/api/venue-connections", json=create_payload(revision))

    assert response.status_code == 201
    assert response.json()["revision"] == revision + 1
    assert response.json()["connection"]["id"] == "paper-extra"
    latest = (await api_harness.client.get("/api/config")).json()
    assert [item["id"] for item in latest["document"]["execution"]["connections"]][-1] == "paper-extra"


async def test_environment_is_immutable_after_creation(api_harness):
    connection, revision = await _connection_and_revision(api_harness, "okx-demo")

    response = await api_harness.client.put(
        "/api/venue-connections/okx-demo",
        json=update_payload(revision, connection, environment="live"),
    )

    assert response.status_code == 422


async def test_connection_cannot_be_disabled_while_enabled_book_references_it(api_harness):
    connection, revision = await _connection_and_revision(api_harness, "okx-demo")

    response = await api_harness.client.put(
        "/api/venue-connections/okx-demo",
        json=update_payload(revision, connection, enabled=False),
    )

    assert response.status_code == 422


async def test_connection_update_rejects_stale_revision(api_harness):
    connection, revision = await _connection_and_revision(api_harness, "okx-demo")
    body = update_payload(revision, connection, label="Updated")
    assert (await api_harness.client.put("/api/venue-connections/okx-demo", json=body)).status_code == 200

    stale = await api_harness.client.put("/api/venue-connections/okx-demo", json=body)

    assert stale.status_code == 409


async def test_credentials_use_connection_credential_ref_and_increment_global_revision(api_harness):
    saved = await put_fixture_credentials(api_harness, "okx-demo", marker="credential-marker")

    assert saved.status_code == 200
    assert saved.json()["revision"] == 2
    assert saved.json()["credential"] == {
        "credential_ref": "okx-demo-credentials",
        "configured": True,
        "updated_at": ANY,
    }


async def test_paper_connection_rejects_credentials(api_harness):
    _, revision = await _connection_and_revision(api_harness, "okx-demo")
    created = await api_harness.client.post(
        "/api/venue-connections",
        json=create_payload(revision, connection_id="paper-no-credentials"),
    )
    assert created.status_code == 201

    response = await api_harness.client.put(
        "/api/venue-connections/paper-no-credentials/credentials",
        json={
            "expected_revision": created.json()["revision"],
            "credentials": {"api_key": "key", "secret": "value"},  # pragma: allowlist secret
        },
    )

    assert response.status_code == 422


async def test_credential_storage_failure_is_safe_service_unavailable(api_harness):
    marker = "raw-credential-storage-secret"
    _, revision = await _connection_and_revision(api_harness, "okx-demo")
    api_harness.runtime.repository.put_credentials = AsyncMock(side_effect=RuntimeError(marker))

    response = await api_harness.client.put(
        "/api/venue-connections/okx-demo/credentials",
        json={
            "expected_revision": revision,
            "credentials": {"api_key": "key", "secret": "value"},  # pragma: allowlist secret
        },
    )

    assert response.status_code == 503
    assert marker not in response.text


async def test_connection_test_requires_configured_credentials(api_harness):
    response = await api_harness.client.post("/api/venue-connections/bybit-testnet/test")

    assert response.status_code == 503
    assert "bybit-testnet-credentials" not in response.text


async def test_connection_test_uses_adapter_and_returns_normalized_health(api_harness):
    assert (await put_fixture_credentials(api_harness, "bybit-testnet")).status_code == 200

    response = await api_harness.client.post("/api/venue-connections/bybit-testnet/test")

    assert response.status_code == 200
    assert response.json() == {
        "connection_id": "bybit-testnet",
        "healthy": True,
        "environment": "testnet",
        "capabilities": {
            "market_types": ["swap"],
            "native_protection": True,
            "hedge_mode": False,
            "reduce_only": True,
            "supported_order_types": ["limit", "market"],
        },
        "credential_configured": True,
    }
    adapter = api_harness.adapters["bybit"]
    assert len(adapter.connect_calls) == 1
    assert adapter.opened_sessions[0].closed == 1


async def test_connection_test_returns_safe_bad_gateway_and_closes_session(api_harness):
    marker = "raw-adapter-secret-error"
    assert (await put_fixture_credentials(api_harness, "okx-demo")).status_code == 200
    adapter = api_harness.adapters["okx"]
    adapter.error = RuntimeError(marker)

    response = await api_harness.client.post("/api/venue-connections/okx-demo/test")

    assert response.status_code == 502
    assert marker not in response.text
