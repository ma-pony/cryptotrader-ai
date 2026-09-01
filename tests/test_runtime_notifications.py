"""Durable alert recording honors explicit event selection."""

from datetime import UTC, datetime

import pytest

from cryptotrader.alerts.models import BusinessAlertEvent
from cryptotrader.alerts.service import AlertService
from cryptotrader.alerts.store import AlertStore
from cryptotrader.migrations.workbench import migrate_alerts
from cryptotrader.runtime_config.models import NotificationConfig


@pytest.mark.asyncio
async def test_runtime_alert_filters_outbox_not_business_record(tmp_path):
    async def config():
        return NotificationConfig(enabled=True, webhook_url="https://example.test/hook", events=("daily_summary",))

    url = f"sqlite+aiosqlite:///{tmp_path}/notify.db"
    await migrate_alerts(url)
    service = AlertService(AlertStore(url), config)
    now = datetime(2026, 9, 1, tzinfo=UTC)
    await service.record(
        BusinessAlertEvent(
            event_key="connection:x:initial:failed",
            type="connection_failed",
            occurred_at=now,
            connection_id="x",
            message="连接失败",
        )
    )
    await service.record(
        BusinessAlertEvent(
            event_key="daily_summary:2026-09-01", type="daily_summary", occurred_at=now, message="每日摘要"
        )
    )
    assert len(await service.store.list_alerts()) == 2
    assert len(await service.store.list_deliveries()) == 1
