"""Removing an account requires a new empty read and preserves archived history."""
# ruff: noqa: F811 -- Pytest fixture imported and injected by name.

from dataclasses import replace

import pytest

from tests.test_account_operations import operation_harness  # noqa: F401 - shared offline fixture


async def test_disabled_flat_account_can_be_removed_and_history_remains(operation_harness):
    h = operation_harness
    from decimal import Decimal

    h.venue.amount = h.venue.available = Decimal("0")
    h.venue.ordinary = h.venue.protection = False
    current = await h.runtime.repository.get_existing()
    document = current.document.model_copy(
        update={
            "execution": current.document.execution.model_copy(
                update={
                    "connections": tuple(
                        replace(c, enabled=False) if c.id == "okx-demo" else c
                        for c in current.document.execution.connections
                    ),
                    "books": tuple(
                        replace(b, enabled=False) if b.id == "simulation" else b
                        for b in current.document.execution.books
                    ),
                }
            )
        }
    )
    saved = await h.runtime.repository.replace(current.revision, document)
    response = await h.client.delete("/api/venue-connections/okx-demo", params={"expected_revision": saved.revision})
    assert response.status_code == 200, response.text
    detail = await h.client.get("/api/accounts/okx-demo")
    assert detail.status_code == 200, detail.text
    assert detail.json()["archived"] is True
    assert "credential" not in str(detail.json())
    assert (await h.client.get("/api/accounts/okx-demo/fills")).status_code == 200
    assert (await h.client.post("/api/accounts/okx-demo/sync")).status_code == 409
    assert "okx-demo" not in {i["connection_id"] for i in (await h.client.get("/api/accounts")).json()["items"]}
    current = await h.runtime.repository.get_existing()
    with pytest.raises(ValueError, match="archived"):
        await h.runtime.repository.replace(current.revision, document)


async def test_remove_with_open_orders_is_rejected(operation_harness):
    h = operation_harness
    current = await h.runtime.repository.get_existing()
    document = current.document.model_copy(
        update={
            "execution": current.document.execution.model_copy(
                update={
                    "connections": tuple(
                        replace(c, enabled=False) if c.id == "okx-demo" else c
                        for c in current.document.execution.connections
                    ),
                    "books": tuple(
                        replace(b, enabled=False) if b.id == "simulation" else b
                        for b in current.document.execution.books
                    ),
                }
            )
        }
    )
    saved = await h.runtime.repository.replace(current.revision, document)
    response = await h.client.delete("/api/venue-connections/okx-demo", params={"expected_revision": saved.revision})
    assert response.status_code == 409, response.text
    assert (await h.runtime.repository.get_existing()).revision == saved.revision


@pytest.mark.parametrize("kind", ["delete", "unassign", "sync_failure"])
async def test_total_put_and_allocation_removal_share_fresh_safety_assertion(operation_harness, kind):
    from tests.test_runtime_config_api import active_payload

    h = operation_harness
    current = await h.runtime.repository.get_existing()
    document = current.document.model_copy(
        update={
            "execution": current.document.execution.model_copy(
                update={
                    "books": tuple(
                        replace(b, enabled=False) if b.id == "simulation" else b
                        for b in current.document.execution.books
                    ),
                }
            )
        }
    )
    saved = await h.runtime.repository.replace(current.revision, document)
    desired = active_payload()
    desired["execution"]["books"][0]["enabled"] = False
    desired["execution"]["books"][0]["allocations"] = [
        a for a in desired["execution"]["books"][0]["allocations"] if a["connection_id"] != "okx-demo"
    ]
    if kind != "unassign":
        desired["execution"]["connections"] = [c for c in desired["execution"]["connections"] if c["id"] != "okx-demo"]
    if kind == "sync_failure":
        h.venue.read_fail = True
    response = await h.client.put("/api/config", json={"expected_revision": saved.revision, "document": desired})
    assert response.status_code == 409, response.text
    assert (await h.runtime.repository.get_existing()).revision == saved.revision
    assert not any("cancel" in call or "close" in call for call in h.venue.calls)
