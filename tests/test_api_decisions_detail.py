"""Canonical cycles 与 decisions detail 返回同一严格多平台层级。"""

# ruff: noqa: F401, F811 - 导入 fixture 供本模块的 pytest 参数解析。

from __future__ import annotations

from tests.test_multi_venue_journal import _record
from tests.test_runtime_config_api import api_harness


async def test_cycle_detail_preserves_book_connection_execution_audit(api_harness):
    await api_harness.runtime.cycle.journal.save(_record())

    response = await api_harness.client.get("/api/cycles/cycle-1")

    assert response.status_code == 200
    body = response.json()
    book = body["books"][0]
    assert body["cycle_status"] == "partial"
    assert body["execution_status"] == "partial"
    assert body["requires_attention"] is True
    assert book["status"] == "partial"
    assert book["hitl"]["status"] == "not_required"
    assert book["portfolio_before"]["book_id"] == "simulation"
    assert book["portfolio_after"]["book_id"] == "simulation"
    assert book["connections"][0]["plan"]["connection_id"] == "sim-first"
    assert book["connections"][0]["plan"]["pair"] == {"symbol": "BTC/USDT:USDT"}
    assert book["connections"][0]["execution"]["connection_id"] == "sim-first"
    assert book["connections"][0]["execution"]["final_position"]["position"]["pair"] == {"symbol": "BTC/USDT:USDT"}


async def test_decision_detail_is_exactly_the_canonical_cycle_contract(api_harness):
    await api_harness.runtime.cycle.journal.save(_record())

    canonical = await api_harness.client.get("/api/cycles/cycle-1")
    decision = await api_harness.client.get("/api/decisions/cycle-1")

    assert decision.status_code == 200
    assert decision.json() == canonical.json()


async def test_cycle_and_decision_detail_return_not_found(api_harness):
    canonical = await api_harness.client.get("/api/cycles/missing")
    decision = await api_harness.client.get("/api/decisions/missing")

    assert canonical.status_code == 404
    assert decision.status_code == 404
