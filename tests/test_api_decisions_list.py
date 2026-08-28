from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.factories.signal_fusion import cycle_record


@pytest.fixture
def client() -> TestClient:
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


def _config() -> MagicMock:
    config = MagicMock()
    config.infrastructure.database_url = None
    return config


def _completed():
    return cycle_record(
        cycle_id="cycle-completed",
        status="completed",
        profile_revision=3,
        fused_signal={"score": 0.36, "direction": "long", "contributions": []},
        target_position={"side": "long", "size_ratio": 0.2},
        risk_result={"passed": True, "rejected_by": "", "reason": ""},
        execution_result={"succeeded": True, "orders": [{"status": "filled"}]},
    )


def _failed():
    return cycle_record(
        cycle_id="cycle-failed",
        status="component_failed",
        component_error={"llm_committee": "RuntimeError: timeout"},
    )


def _store(records):
    store = MagicMock()
    store.list = AsyncMock(return_value=records)
    store.count = AsyncMock(return_value=len(records))
    return store


def test_list_exposes_fusion_target_risk_and_execution(client: TestClient) -> None:
    store = _store([_completed()])
    with (
        patch("cryptotrader.config.load_config", return_value=_config()),
        patch("cryptotrader.journal.store.CycleJournalStore", return_value=store),
    ):
        response = client.get("/api/decisions")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["cycle_id"] == "cycle-completed"
    assert item["profile_revision"] == 3
    assert item["fused_score"] == pytest.approx(0.36)
    assert item["target_position"] == {"side": "long", "size_ratio": 0.2}
    assert item["risk_result"]["passed"] is True
    assert item["execution_result"]["succeeded"] is True


def test_list_includes_component_failed_cycles(client: TestClient) -> None:
    store = _store([_failed()])
    with (
        patch("cryptotrader.config.load_config", return_value=_config()),
        patch("cryptotrader.journal.store.CycleJournalStore", return_value=store),
    ):
        response = client.get("/api/decisions")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["status"] == "component_failed"
    assert item["component_error"] == {"llm_committee": "RuntimeError: timeout"}


def test_list_preserves_unknown_price_for_unavailable_context(client: TestClient) -> None:
    record = cycle_record(
        cycle_id="cycle-context-unavailable",
        status="cycle_failed",
        error="RuntimeError: context unavailable",
        context_summary={
            "available": False,
            "pair": "BTC/USDT:USDT",
            "as_of": None,
            "mode": "paper",
            "exchange_id": "okx",
        },
    )
    store = _store([record])
    with (
        patch("cryptotrader.config.load_config", return_value=_config()),
        patch("cryptotrader.journal.store.CycleJournalStore", return_value=store),
    ):
        response = client.get("/api/decisions")

    assert response.status_code == 200
    assert response.json()["items"][0]["price"] is None


def test_list_passes_pair_status_and_offset_to_cycle_store(client: TestClient) -> None:
    store = _store([])
    with (
        patch("cryptotrader.config.load_config", return_value=_config()),
        patch("cryptotrader.journal.store.CycleJournalStore", return_value=store),
    ):
        response = client.get("/api/decisions?pair=ETH/USDT&status=risk_rejected&page=2&size=10")

    assert response.status_code == 200
    store.list.assert_awaited_once_with(limit=10, offset=10, pair="ETH/USDT", status="risk_rejected")
    store.count.assert_awaited_once_with(pair="ETH/USDT", status="risk_rejected")


def test_list_pagination_uses_exact_total(client: TestClient) -> None:
    store = _store([_completed()])
    store.count.return_value = 21
    with (
        patch("cryptotrader.config.load_config", return_value=_config()),
        patch("cryptotrader.journal.store.CycleJournalStore", return_value=store),
    ):
        body = client.get("/api/decisions?page=2&size=10").json()

    assert body["total"] == 21
    assert body["has_next"] is True


def test_list_rejects_invalid_page_size(client: TestClient) -> None:
    assert client.get("/api/decisions?page=0").status_code == 422
    assert client.get("/api/decisions?size=101").status_code == 422
