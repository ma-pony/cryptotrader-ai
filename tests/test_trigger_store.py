"""Tests for TriggerRuleStore CRUD operations (T005).

Uses an in-memory SQLite database via aiosqlite + SQLAlchemy async.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from cryptotrader._compat import UTC
from cryptotrader.triggers.models import Base
from cryptotrader.triggers.store import TriggerRuleStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def session_factory():
    """Create in-memory SQLite engine and return an async session factory."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sm = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _factory() -> AsyncSession:
        return sm()

    yield _factory

    await engine.dispose()


@pytest.fixture
async def store(session_factory):
    return TriggerRuleStore(session_factory)


def _rule_data(**kwargs) -> dict:
    base = {
        "name": "BTC Drop Alert",
        "trigger_type": "price_threshold",
        "pair": "BTC/USDT",
        "parameters": {"direction": "below", "price": 50000},
        "cooldown_minutes": 30,
        "enabled": True,
        "created_by": "user",
        "schedule_depth": 0,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# create_rule + get_rule
# ---------------------------------------------------------------------------


class TestCreateAndGetRule:
    async def test_create_returns_rule_with_id(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        assert rule.id is not None
        assert len(rule.id) == 36  # UUID string
        assert rule.name == "BTC Drop Alert"
        assert rule.trigger_type == "price_threshold"
        assert rule.pair == "BTC/USDT"

    async def test_get_rule_returns_existing(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        fetched = await store.get_rule(rule.id)
        assert fetched is not None
        assert fetched.id == rule.id
        assert fetched.name == rule.name

    async def test_get_rule_returns_none_for_unknown_id(self, store: TriggerRuleStore) -> None:
        result = await store.get_rule("nonexistent-uuid-0000")
        assert result is None


# ---------------------------------------------------------------------------
# list_rules
# ---------------------------------------------------------------------------


class TestListRules:
    async def test_list_all_rules(self, store: TriggerRuleStore) -> None:
        await store.create_rule(_rule_data(name="Rule A"))
        await store.create_rule(_rule_data(name="Rule B", enabled=False))
        rules = await store.list_rules()
        assert len(rules) == 2

    async def test_list_enabled_only(self, store: TriggerRuleStore) -> None:
        await store.create_rule(_rule_data(name="Enabled"))
        await store.create_rule(_rule_data(name="Disabled", enabled=False))
        rules = await store.list_rules(enabled_only=True)
        assert len(rules) == 1
        assert rules[0].name == "Enabled"

    async def test_list_empty_returns_empty_list(self, store: TriggerRuleStore) -> None:
        rules = await store.list_rules()
        assert rules == []

    async def test_list_ordered_by_created_at_desc(self, store: TriggerRuleStore) -> None:
        await store.create_rule(_rule_data(name="First"))
        second = await store.create_rule(_rule_data(name="Second"))
        rules = await store.list_rules()
        # Most recently created appears first
        names = [r.name for r in rules]
        assert names.index("Second") < names.index("First") or second.id == rules[0].id


# ---------------------------------------------------------------------------
# update_rule
# ---------------------------------------------------------------------------


class TestUpdateRule:
    async def test_update_changes_fields(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        updated = await store.update_rule(rule.id, {"name": "New Name", "cooldown_minutes": 60})
        assert updated is not None
        assert updated.name == "New Name"
        assert updated.cooldown_minutes == 60

    async def test_update_nonexistent_returns_none(self, store: TriggerRuleStore) -> None:
        result = await store.update_rule("ghost-id", {"name": "X"})
        assert result is None

    async def test_update_sets_updated_at(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        before = rule.updated_at
        updated = await store.update_rule(rule.id, {"name": "Changed"})
        assert updated is not None
        # updated_at should be >= created_at
        assert updated.updated_at >= before


# ---------------------------------------------------------------------------
# toggle_rule
# ---------------------------------------------------------------------------


class TestToggleRule:
    async def test_toggle_disables_enabled_rule(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data(enabled=True))
        toggled = await store.toggle_rule(rule.id)
        assert toggled is not None
        assert toggled.enabled is False

    async def test_toggle_enables_disabled_rule(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data(enabled=False))
        toggled = await store.toggle_rule(rule.id)
        assert toggled is not None
        assert toggled.enabled is True

    async def test_toggle_twice_restores_original(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data(enabled=True))
        await store.toggle_rule(rule.id)
        toggled2 = await store.toggle_rule(rule.id)
        assert toggled2 is not None
        assert toggled2.enabled is True

    async def test_toggle_nonexistent_returns_none(self, store: TriggerRuleStore) -> None:
        result = await store.toggle_rule("no-such-id")
        assert result is None


# ---------------------------------------------------------------------------
# delete_rule
# ---------------------------------------------------------------------------


class TestDeleteRule:
    async def test_delete_existing_rule_returns_true(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        deleted = await store.delete_rule(rule.id)
        assert deleted is True

    async def test_deleted_rule_is_gone(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        await store.delete_rule(rule.id)
        assert await store.get_rule(rule.id) is None

    async def test_delete_nonexistent_returns_false(self, store: TriggerRuleStore) -> None:
        deleted = await store.delete_rule("does-not-exist")
        assert deleted is False


# ---------------------------------------------------------------------------
# record_event + list_events
# ---------------------------------------------------------------------------


class TestRecordAndListEvents:
    async def test_record_event_creates_record(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        event = await store.record_event(
            {
                "rule_id": rule.id,
                "trigger_reason": "BTC fell below 50000",
                "price_snapshot": {"price": 49500, "pair": "BTC/USDT"},
                "schedule_depth": 0,
                "cooldown_skipped": False,
            }
        )
        assert event.id is not None
        assert event.rule_id == rule.id
        assert event.cooldown_skipped is False

    async def test_list_events_returns_paginated(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        for i in range(5):
            await store.record_event(
                {
                    "rule_id": rule.id,
                    "trigger_reason": f"trigger {i}",
                    "price_snapshot": {},
                    "schedule_depth": 0,
                    "cooldown_skipped": False,
                }
            )
        events, total = await store.list_events(page=1, size=3)
        assert total == 5
        assert len(events) == 3

    async def test_list_events_second_page(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        for i in range(5):
            await store.record_event(
                {
                    "rule_id": rule.id,
                    "trigger_reason": f"trigger {i}",
                    "price_snapshot": {},
                    "schedule_depth": 0,
                    "cooldown_skipped": False,
                }
            )
        events, total = await store.list_events(page=2, size=3)
        assert total == 5
        assert len(events) == 2

    async def test_list_events_filtered_by_rule_id(self, store: TriggerRuleStore) -> None:
        r1 = await store.create_rule(_rule_data(name="R1"))
        r2 = await store.create_rule(_rule_data(name="R2"))
        await store.record_event(
            {
                "rule_id": r1.id,
                "trigger_reason": "a",
                "price_snapshot": {},
                "schedule_depth": 0,
                "cooldown_skipped": False,
            }
        )
        await store.record_event(
            {
                "rule_id": r2.id,
                "trigger_reason": "b",
                "price_snapshot": {},
                "schedule_depth": 0,
                "cooldown_skipped": False,
            }
        )
        events, total = await store.list_events(rule_id=r1.id)
        assert total == 1
        assert events[0].rule_id == r1.id


# ---------------------------------------------------------------------------
# cleanup_expired_rules
# ---------------------------------------------------------------------------


class TestCleanupExpiredRules:
    async def test_cleanup_removes_expired_agent_rules(self, store: TriggerRuleStore) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        await store.create_rule(_rule_data(name="Expired", created_by="agent", ttl_expires_at=past))
        await store.create_rule(_rule_data(name="Alive"))
        removed = await store.cleanup_expired_rules()
        assert removed == 1
        rules = await store.list_rules()
        assert len(rules) == 1
        assert rules[0].name == "Alive"

    async def test_cleanup_keeps_non_expired_agent_rules(self, store: TriggerRuleStore) -> None:
        future = datetime.now(UTC) + timedelta(hours=1)
        await store.create_rule(_rule_data(name="Future", created_by="agent", ttl_expires_at=future))
        removed = await store.cleanup_expired_rules()
        assert removed == 0

    async def test_cleanup_keeps_user_created_expired_rules(self, store: TriggerRuleStore) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        await store.create_rule(_rule_data(name="UserRule", created_by="user", ttl_expires_at=past))
        removed = await store.cleanup_expired_rules()
        assert removed == 0

    async def test_cleanup_no_rules_returns_zero(self, store: TriggerRuleStore) -> None:
        removed = await store.cleanup_expired_rules()
        assert removed == 0


# ---------------------------------------------------------------------------
# count_rules
# ---------------------------------------------------------------------------


class TestCountRules:
    async def test_count_empty(self, store: TriggerRuleStore) -> None:
        assert await store.count_rules() == 0

    async def test_count_after_creates(self, store: TriggerRuleStore) -> None:
        await store.create_rule(_rule_data(name="A"))
        await store.create_rule(_rule_data(name="B"))
        assert await store.count_rules() == 2

    async def test_count_decrements_after_delete(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        await store.delete_rule(rule.id)
        assert await store.count_rules() == 0


# ---------------------------------------------------------------------------
# get_last_triggered_at
# ---------------------------------------------------------------------------


class TestGetLastTriggeredAt:
    async def test_returns_none_when_no_events(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        result = await store.get_last_triggered_at(rule.id)
        assert result is None

    async def test_returns_most_recent_non_skipped_event(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        # Record a real trigger
        await store.record_event(
            {
                "rule_id": rule.id,
                "trigger_reason": "real",
                "price_snapshot": {},
                "schedule_depth": 0,
                "cooldown_skipped": False,
            }
        )
        # Record a skipped trigger (should not appear)
        await store.record_event(
            {
                "rule_id": rule.id,
                "trigger_reason": "skip",
                "price_snapshot": {},
                "schedule_depth": 0,
                "cooldown_skipped": True,
            }
        )
        last = await store.get_last_triggered_at(rule.id)
        assert last is not None

    async def test_only_skipped_events_returns_none(self, store: TriggerRuleStore) -> None:
        rule = await store.create_rule(_rule_data())
        await store.record_event(
            {
                "rule_id": rule.id,
                "trigger_reason": "skip",
                "price_snapshot": {},
                "schedule_depth": 0,
                "cooldown_skipped": True,
            }
        )
        last = await store.get_last_triggered_at(rule.id)
        assert last is None
