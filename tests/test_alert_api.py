"""Read and acknowledgment APIs never drive the business or HTTP sender."""

from types import SimpleNamespace

import httpx
import pytest

from cryptotrader.alerts.models import BusinessAlertEvent
from cryptotrader.alerts.service import AlertService, DeliveryService
from cryptotrader.alerts.store import AlertStore
from cryptotrader.migrations.workbench import migrate_alerts
from tests.test_alert_lifecycle import NOW, enabled_config


@pytest.mark.asyncio
async def test_get_on_uninitialized_alert_database_is_safe_and_never_creates_tables(tmp_path, monkeypatch):
    import sqlite3

    import api.main as main
    from api.main import app

    monkeypatch.setattr(main, "_get_redis_for_rate_limit", lambda: None)
    path = tmp_path / "uninitialized-api.db"
    store = AlertStore(f"sqlite+aiosqlite:///{path}")
    monkeypatch.setattr(
        app.state,
        "runtime",
        SimpleNamespace(
            snapshot=app.state.runtime.snapshot,
            alerts=AlertService(store, enabled_config),
            alert_store=store,
            deliveries=DeliveryService(store, enabled_config),
            alert_owner=None,
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.get("/api/alerts")
    assert (response.status_code, response.json()) == (503, {"detail": "告警存储尚未就绪"})
    with sqlite3.connect(path) as connection:
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    assert tables == []


@pytest.mark.asyncio
async def test_api_refresh_read_and_retry_only_change_alert_delivery(tmp_path, monkeypatch):
    import api.main as main
    from api.main import app

    monkeypatch.setattr(main, "_get_redis_for_rate_limit", lambda: None)
    url = f"sqlite+aiosqlite:///{tmp_path}/api.db"
    await migrate_alerts(url)
    store = AlertStore(url, clock=lambda: NOW)
    alerts = AlertService(store, enabled_config)
    runtime = app.state.runtime
    monkeypatch.setattr(
        app.state,
        "runtime",
        SimpleNamespace(
            snapshot=runtime.snapshot,
            alerts=alerts,
            alert_store=store,
            deliveries=DeliveryService(store, enabled_config),
            alert_owner=None,
        ),
    )
    identity = await alerts.record(
        BusinessAlertEvent(event_key="summary:api", type="daily_summary", occurred_at=NOW, message="摘要")
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/alerts")
        assert response.status_code == 200
        assert response.json()["items"][0]["id"] == identity
        await client.get("/api/alerts")
        read = await client.post(f"/api/alerts/{identity}/read")
        assert read.status_code == 200
        assert read.json()["resolution"] == "informational"
        delivery = (await store.list_deliveries())[0]
        await store.claim_delivery(delivery.id)
        await store.finish_delivery(delivery.id, "Webhook HTTP 503")
        retry = await client.post(f"/api/alerts/deliveries/{delivery.id}/retry")
        assert retry.status_code == 200
        assert retry.json()["status"] == "pending"
    assert len(await store.list_alerts()) == 1
    assert (await store.get_delivery(delivery.id)).attempts == 1


@pytest.mark.asyncio
async def test_api_maps_missing_and_invalid_retry_without_internal_details(tmp_path, monkeypatch):
    import api.main as main
    from api.main import app

    monkeypatch.setattr(main, "_get_redis_for_rate_limit", lambda: None)
    url = f"sqlite+aiosqlite:///{tmp_path}/errors.db"
    await migrate_alerts(url)
    store = AlertStore(url, clock=lambda: NOW)
    alerts = AlertService(store, enabled_config)
    monkeypatch.setattr(
        app.state,
        "runtime",
        SimpleNamespace(
            snapshot=app.state.runtime.snapshot,
            alerts=alerts,
            alert_store=store,
            deliveries=DeliveryService(store, enabled_config),
            alert_owner=None,
        ),
    )
    alert_id = await alerts.record(
        BusinessAlertEvent(event_key="summary:conflict", type="daily_summary", occurred_at=NOW, message="摘要")
    )
    delivery = (await store.list_deliveries())[0]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        missing_alert = await client.post("/api/alerts/missing/read")
        missing_delivery = await client.post("/api/alerts/deliveries/missing/retry")
        conflict = await client.post(f"/api/alerts/deliveries/{delivery.id}/retry")
    assert (missing_alert.status_code, missing_alert.json()) == (404, {"detail": "告警不存在"})
    assert (missing_delivery.status_code, missing_delivery.json()) == (404, {"detail": "投递记录不存在"})
    assert (conflict.status_code, conflict.json()) == (409, {"detail": "仅失败投递可以重试"})
    assert (await store.get_alert(alert_id)).read_at is None
