"""Delivery checks use only fake HTTP; rejection must never count as delivered."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from cryptotrader.alerts.models import BusinessAlertEvent
from cryptotrader.alerts.owner import AlertOwner
from cryptotrader.alerts.service import AlertService, DeliveryService
from cryptotrader.alerts.store import AlertStore
from cryptotrader.migrations.workbench import migrate_alerts
from cryptotrader.notifications import WebhookBackend
from tests.test_alert_lifecycle import NOW, enabled_config


@pytest.mark.asyncio
async def test_failed_delivery_is_durable_retry_increments_without_copying_alert(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/alerts.db"
    await migrate_alerts(url)
    store = AlertStore(url, clock=lambda: NOW)
    alerts = AlertService(store, enabled_config)
    alert_id = await alerts.record(
        BusinessAlertEvent(event_key="summary:today", type="daily_summary", occurred_at=NOW, message="摘要")
    )
    deliveries = await store.list_deliveries()
    assert len(deliveries) == 1
    status = [503]

    def factory(config):
        return WebhookBackend(
            config.webhook_url,
            transport=httpx.MockTransport(lambda request: httpx.Response(status[0], text="private-secret-response")),
        )

    sender = DeliveryService(store, enabled_config, backend_factory=factory)
    await sender.send(deliveries[0].id)
    failed = (await store.list_deliveries())[0]
    assert failed.status == "failed"
    assert failed.attempts == 1
    assert failed.last_error == "Webhook HTTP 503"
    assert failed.last_attempt_at == NOW
    assert (await store.list_alerts())[0].id == alert_id
    status[0] = 204
    await sender.retry(failed.id)
    await sender.send(failed.id)
    delivered = (await store.list_deliveries())[0]
    assert delivered.status == "delivered"
    assert delivered.attempts == 2
    assert delivered.delivered_at == NOW
    assert len(await store.list_alerts()) == 1
    assert "secret" not in delivered.model_dump_json()


@pytest.mark.asyncio
async def test_webhook_non_success_is_a_delivery_failure(monkeypatch):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    with pytest.raises(httpx.HTTPStatusError):
        await WebhookBackend("https://example.test/private-token").send("execution_failed", {"message": "执行需核对"})


@pytest.mark.asyncio
async def test_background_owner_sends_persisted_pending_without_browser(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/owner.db"
    await migrate_alerts(url)
    store = AlertStore(url, clock=lambda: NOW)
    await AlertService(store, enabled_config).record(
        BusinessAlertEvent(event_key="summary:owner", type="daily_summary", occurred_at=NOW, message="摘要")
    )
    sent = asyncio.Event()

    def respond(request):
        sent.set()
        return httpx.Response(204)

    sender = DeliveryService(
        AlertStore(store.database_url),
        enabled_config,
        backend_factory=lambda config: WebhookBackend(config.webhook_url, transport=httpx.MockTransport(respond)),
    )
    owner = AlertOwner(AsyncMock(), sender, interval=0.01)
    owner.start()
    try:
        done, _ = await asyncio.wait({asyncio.create_task(sent.wait())}, timeout=0.2)
        assert done, "the independent owner must send persisted pending deliveries"
    finally:
        await owner.stop()
    assert (await store.list_deliveries())[0].status == "delivered"


@pytest.mark.asyncio
async def test_deliveries_are_newest_activity_first(tmp_path):
    from datetime import timedelta

    url = f"sqlite+aiosqlite:///{tmp_path}/ordered.db"
    await migrate_alerts(url)
    store = AlertStore(url, clock=lambda: NOW)
    service = AlertService(store, enabled_config)
    await service.record(
        BusinessAlertEvent(event_key="summary:old", type="daily_summary", occurred_at=NOW, message="旧")
    )
    store.clock = lambda: NOW + timedelta(minutes=1)
    await service.record(
        BusinessAlertEvent(event_key="summary:new", type="daily_summary", occurred_at=NOW, message="新")
    )
    old, new = reversed(await store.list_deliveries())
    store.clock = lambda: NOW + timedelta(minutes=2)
    await store.claim_delivery(old.id)
    ordered = await store.list_deliveries()
    assert [item.id for item in ordered] == [old.id, new.id]
