"""Canonical cycles 与 decisions 列表共享多资金池 Journal 契约。"""

# ruff: noqa: F401, F811 - 导入 fixture 供本模块的 pytest 参数解析。

from __future__ import annotations

from tests.test_multi_venue_journal import _record
from tests.test_runtime_config_api import api_harness


async def test_cycles_list_returns_shared_signals_books_and_connections(api_harness):
    await api_harness.runtime.cycle.journal.save(_record())

    response = await api_harness.client.get("/api/cycles")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["page"] == 1
    assert body["size"] == 20
    item = body["items"][0]
    assert item["cycle_id"] == "cycle-1"
    assert set(item["shared_signals"]) == {"components", "fused", "target_position"}
    assert item["shared_signals"]["components"][0]["component_id"] == "kronos"
    assert {entry["key"] for entry in item["shared_signals"]["components"][0]["details"]} == {
        "as_of",
        "direction",
        "pair",
        "threshold",
        "window",
    }
    assert item["shared_signals"]["fused"]["contributions"][0]["component_id"] == "kronos"
    assert item["books"][0]["book_id"] == "simulation"
    assert item["books"][0]["risk"]["connection_targets"][0]["connection_id"] == "sim-first"
    assert [connection["connection_id"] for connection in item["books"][0]["connections"]] == [
        "sim-first",
        "sim-second",
    ]
    assert "profile_revision" not in item
    assert "trade_plan" not in item


async def test_decisions_list_uses_the_same_new_journal_and_dto(api_harness):
    await api_harness.runtime.cycle.journal.save(_record())

    canonical = await api_harness.client.get("/api/cycles")
    decisions = await api_harness.client.get("/api/decisions")

    assert decisions.status_code == 200
    assert decisions.json() == canonical.json()


async def test_cycles_list_paginates_the_single_runtime_store(api_harness):
    await api_harness.runtime.cycle.journal.save(_record(cycle_id="cycle-1"))
    await api_harness.runtime.cycle.journal.save(_record(cycle_id="cycle-2"))

    response = await api_harness.client.get("/api/cycles", params={"page": 1, "size": 1})

    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert len(response.json()["items"]) == 1
    assert response.json()["has_next"] is True


async def test_cycles_require_active_runtime_journal(api_harness):
    api_harness.runtime.cycle = None

    response = await api_harness.client.get("/api/cycles")

    assert response.status_code == 503
