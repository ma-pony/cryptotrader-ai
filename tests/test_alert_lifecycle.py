"""Durable business alerts, independent of notification and trading state."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from cryptotrader.accounts.models import AccountOperationOut, AccountOperationResult
from cryptotrader.accounts.store import AccountOperationStore, AccountStore
from cryptotrader.alerts.models import BusinessAlertEvent
from cryptotrader.alerts.owner import AlertOwner
from cryptotrader.alerts.recovery import AlertRecovery
from cryptotrader.alerts.service import AlertService, DeliveryService
from cryptotrader.alerts.store import AlertStore
from cryptotrader.hitl.store import BookApprovalStore
from cryptotrader.journal.store import MultiVenueCycleStore
from cryptotrader.migrations.workbench import migrate_alerts, migrate_workbench_schema
from cryptotrader.runtime_config.models import NotificationConfig
from tests.test_multi_venue_journal import _book_cycle, _proposal_for, _record

NOW = datetime(2026, 9, 1, tzinfo=UTC)


async def enabled_config():
    return NotificationConfig(enabled=True, webhook_url="https://example.test/secret", events=("daily_summary",))


@pytest.mark.asyncio
async def test_unique_alert_and_outbox_survive_restart_and_reads_are_separate(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/alerts.db"
    await migrate_alerts(url)
    event = BusinessAlertEvent(
        event_key="summary:2026-09-01", type="daily_summary", occurred_at=NOW, message="当日运行摘要"
    )
    store = AlertStore(url, clock=lambda: NOW)
    service = AlertService(store, enabled_config)
    alert_id = await service.record(event)
    assert isinstance(alert_id, str)
    assert await service.record(event) == alert_id
    restarted = AlertStore(url, clock=lambda: NOW)
    assert len(await restarted.list_alerts()) == 1
    assert len(await restarted.list_deliveries()) == 1
    before = (await restarted.list_alerts())[0]
    await AlertService(restarted, enabled_config).mark_read(alert_id)
    after = (await restarted.list_alerts())[0]
    assert after.read_at == NOW
    assert after.resolution == before.resolution == "informational"


def test_business_event_selection_survives_configuration_round_trip():
    events = (
        "approval_pending",
        "execution_failed",
        "protection_failed",
        "risk_adjusted",
        "component_failed",
        "connection_failed",
        "daily_summary",
    )
    config = NotificationConfig(events=events)
    assert NotificationConfig.model_validate(config.model_dump()).events == events


async def make_recovery(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/recovery.db"
    await migrate_workbench_schema(url)
    store = AlertStore(url, clock=lambda: NOW)
    alerts = AlertService(store, enabled_config)
    journal = MultiVenueCycleStore(url)
    approvals = BookApprovalStore(url)
    accounts = AccountStore(url, clock=lambda: NOW)
    operations = AccountOperationStore(url)
    recovery = AlertRecovery(alerts, journal=journal, approvals=approvals, accounts=accounts, operations=operations)
    return store, journal, approvals, accounts, operations, recovery


@pytest.mark.asyncio
async def test_committed_execution_attention_is_recovered_without_changing_trade(tmp_path):
    store, journal, _, accounts, _, recovery = await make_recovery(tmp_path)
    await accounts.ensure_tables()
    book = replace(_book_cycle(status="completed"), reconciliation_required=True)
    record = _record(
        book_results=(book,), cycle_status="completed", execution_status="completed", requires_attention=True
    )
    await journal.save(record)
    await recovery.reconcile()
    await recovery.reconcile()
    alerts = await store.list_alerts()
    assert len(alerts) == 1
    assert alerts[0].event_key == "book:cycle-1:simulation:execution_failed"
    assert "核对" in alerts[0].message
    assert "未下单" not in alerts[0].message
    assert (await journal.get("cycle-1")).cycle_status == "completed"


@pytest.mark.asyncio
async def test_approval_read_does_not_approve_and_expiry_is_not_execution(tmp_path):
    store, _, approvals, accounts, _, recovery = await make_recovery(tmp_path)
    await accounts.ensure_tables()
    approval = await approvals.create(_proposal_for(), cycle_id="decision")
    store.clock = lambda: approval.created_at
    await recovery.reconcile()
    items = await store.list_alerts()
    assert len(items) == 1
    await store.mark_read(items[0].id)
    assert (await approvals.get(approval.id)).status == "pending"
    assert (await store.get_alert(items[0].id)).resolution == "open"
    store.clock = lambda: approval.created_at + timedelta(hours=2)
    await recovery.reconcile()
    assert (await store.get_alert(items[0].id)).resolution == "expired"
    assert (await approvals.get(approval.id)).status == "pending"


@pytest.mark.asyncio
async def test_connection_failure_poll_dedup_and_recurrence_after_recovery(tmp_path):
    from cryptotrader.accounts.store import SyncStatusRow
    from cryptotrader.db import get_async_session

    store, _, _, accounts, _, recovery = await make_recovery(tmp_path)
    await accounts.failed("connection", "safe reason")
    await recovery.reconcile()
    accounts._clock = lambda: NOW + timedelta(minutes=1)
    await accounts.failed("connection", "safe reason")
    await recovery.reconcile()
    assert len(await store.list_alerts()) == 1
    async with await get_async_session(store.database_url) as session, session.begin():
        row = await session.get(SyncStatusRow, "connection")
        row.last_success_at = NOW + timedelta(minutes=2)
        row.failure_reason = None
    # Even if recovery happened between scans, new failure has a new incident key.
    accounts._clock = lambda: NOW + timedelta(minutes=3)
    await accounts.failed("connection", "safe reason")
    await recovery.reconcile()
    alerts = await store.list_alerts()
    assert len(alerts) == 2
    assert sum(alert.resolution == "open" for alert in alerts) == 1
    assert sum(alert.resolution == "recovered" for alert in alerts) == 1


@pytest.mark.asyncio
async def test_interrupted_operation_uses_operation_identity_not_fake_decision(tmp_path):
    store, _, _, accounts, operations, recovery = await make_recovery(tmp_path)
    await accounts.ensure_tables()
    await operations.create(
        AccountOperationOut(
            operation_id="exit-1",
            connection_id="connection",
            pair="BTC/USDT",
            kind="flatten",
            status="preparing",
            created_at=NOW,
            updated_at=NOW,
        )
    )
    await operations.recover_interrupted()
    await recovery.reconcile()
    alerts = await store.list_alerts()
    assert len(alerts) == 1
    assert alerts[0].operation_id == "exit-1"
    assert alerts[0].decision_id is None
    assert alerts[0].resolution == "open"
    assert "已成交" not in alerts[0].message


@pytest.mark.asyncio
async def test_uninitialized_store_and_owner_never_create_alert_tables(tmp_path):
    import sqlite3

    path = tmp_path / "uninitialized.db"
    url = f"sqlite+aiosqlite:///{path}"
    store = AlertStore(url)
    service = AlertService(store, enabled_config)
    with pytest.raises(SQLAlchemyError):
        await store.list_alerts()
    with pytest.raises(SQLAlchemyError):
        await service.record(
            BusinessAlertEvent(event_key="summary:no-schema", type="daily_summary", occurred_at=NOW, message="摘要")
        )
    owner = AlertOwner(AsyncMock(), DeliveryService(store, enabled_config), interval=0.01)
    owner.start()
    await __import__("asyncio").sleep(0.03)
    await owner.stop()
    with sqlite3.connect(path) as connection:
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    assert tables == []
    await migrate_alerts(url)
    assert await store.list_alerts() == []


@pytest.mark.asyncio
async def test_atomic_alert_delivery_rolls_back_and_reconcile_repairs(monkeypatch, tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/atomic.db"
    await migrate_alerts(url)
    store = AlertStore(url)
    service = AlertService(store, enabled_config)
    event = BusinessAlertEvent(event_key="summary:atomic", type="daily_summary", occurred_at=NOW, message="摘要")

    def fail_delivery_stage(*args):
        raise RuntimeError("injected enqueue failure")

    monkeypatch.setattr(store, "_stage_delivery", fail_delivery_stage)
    with pytest.raises(RuntimeError, match="injected enqueue failure"):
        await service.record(event)
    assert await store.list_alerts() == []
    assert await store.list_deliveries() == []
    monkeypatch.undo()
    await service.record(event)
    await service.record(event)
    assert len(await store.list_alerts()) == 1
    assert len(await store.list_deliveries()) == 1


@pytest.mark.asyncio
async def test_existing_alert_without_selected_delivery_is_repaired_once(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/repair.db"
    await migrate_alerts(url)
    store = AlertStore(url)
    event = BusinessAlertEvent(event_key="summary:repair", type="daily_summary", occurred_at=NOW, message="摘要")
    alert_id = await store.record_once(event.event_key, event)
    assert await store.list_deliveries() == []

    service = AlertService(store, enabled_config)
    assert await service.record(event) == alert_id
    assert await service.record(event) == alert_id
    assert len(await store.list_alerts()) == 1
    assert len(await store.list_deliveries()) == 1


@pytest.mark.asyncio
async def test_completed_flatten_resolves_same_pair_only(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/pairs.db"
    await migrate_alerts(url)
    store = AlertStore(url, clock=lambda: NOW + timedelta(minutes=2))
    alerts = AlertService(store, enabled_config)
    for pair in ("BTC/USDT", "ETH/USDT"):
        await alerts.record(
            BusinessAlertEvent(
                event_key=f"book:decision:{pair}:execution_failed",
                type="execution_failed",
                occurred_at=NOW,
                connection_id="shared",
                pair=pair,
                message="执行需核对",
            )
        )
    operation = AccountOperationOut(
        operation_id="flatten-btc",
        connection_id="shared",
        pair="BTC/USDT",
        kind="flatten",
        status="completed",
        created_at=NOW,
        updated_at=NOW + timedelta(minutes=1),
        result=AccountOperationResult(remaining_position=Decimal("0")),
    )
    recovery = AlertRecovery(
        alerts,
        journal=AsyncMock(),
        approvals=AsyncMock(),
        operations=SimpleNamespace(list_all=AsyncMock(return_value=[operation])),
    )
    await recovery._operations({})
    items = {item.pair: item for item in await store.list_alerts()}
    assert items["BTC/USDT"].resolution == "exit_completed"
    assert items["ETH/USDT"].resolution == "open"
