"""Integration tests for trigger rule CRUD API endpoints (T018).

Tests /api/scheduler/rules and /api/scheduler/triggers endpoints.
Mocks TriggerRuleStore, RedisStateManager, and load_config.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.scheduler import api_router
from cryptotrader._compat import UTC

# ---------------------------------------------------------------------------
# Minimal test app
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)
    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _mock_rule(
    rule_id: str = "rule-abc-123",
    name: str = "BTC Alert",
    trigger_type: str = "price_threshold",
    pair: str = "BTC/USDT",
    parameters: dict | None = None,
    cooldown_minutes: int = 30,
    enabled: bool = True,
    created_by: str = "user",
    schedule_depth: int = 0,
) -> MagicMock:
    rule = MagicMock()
    rule.id = rule_id
    rule.name = name
    rule.trigger_type = trigger_type
    rule.pair = pair
    rule.parameters = parameters or {"direction": "below", "price": 50000}
    rule.cooldown_minutes = cooldown_minutes
    rule.enabled = enabled
    rule.ttl_expires_at = None
    rule.created_by = created_by
    rule.schedule_depth = schedule_depth
    rule.created_at = _now()
    rule.updated_at = _now()
    return rule


def _mock_event(
    event_id: str = "evt-001",
    rule_id: str = "rule-abc-123",
) -> MagicMock:
    event = MagicMock()
    event.id = event_id
    event.rule_id = rule_id
    event.triggered_at = _now()
    event.trigger_reason = "BTC fell below 50000"
    event.price_snapshot = {"pair": "BTC/USDT", "price": 49000.0}
    event.analysis_commit_id = None
    event.schedule_depth = 0
    event.cooldown_skipped = False
    return event


def _mock_config(max_rules: int = 50) -> MagicMock:
    cfg = MagicMock()
    cfg.triggers.max_rules = max_rules
    cfg.infrastructure.redis_url = "redis://localhost:6379"
    return cfg


def _mock_redis(in_cooldown: bool = False) -> MagicMock:
    rsm = AsyncMock()
    rsm.get = AsyncMock(return_value="1" if in_cooldown else None)
    return rsm


@pytest.fixture
def app_and_store():
    app = _make_app()
    store = AsyncMock()
    store.get_last_triggered_at = AsyncMock(return_value=None)
    app.state.trigger_store = store
    return app, store


@pytest.fixture
def client(app_and_store):
    app, _ = app_and_store
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def store(app_and_store):
    _, s = app_and_store
    return s


# ---------------------------------------------------------------------------
# Shared patch context
# ---------------------------------------------------------------------------


def _config_and_redis_patch(max_rules: int = 50, in_cooldown: bool = False):
    """Return a tuple of patchers for load_config and RedisStateManager."""
    return (
        patch("cryptotrader.config.load_config", return_value=_mock_config(max_rules=max_rules)),
        patch("cryptotrader.risk.state.RedisStateManager", return_value=_mock_redis(in_cooldown)),
    )


# ---------------------------------------------------------------------------
# GET /api/scheduler/rules
# ---------------------------------------------------------------------------


class TestListRules:
    def test_returns_list_of_rules(self, client: TestClient, store: AsyncMock) -> None:
        rule = _mock_rule()
        store.list_rules = AsyncMock(return_value=[rule])
        store.get_last_triggered_at = AsyncMock(return_value=None)

        with _config_and_redis_patch()[0], _config_and_redis_patch()[1]:
            resp = client.get("/api/scheduler/rules")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "rule-abc-123"
        assert data[0]["name"] == "BTC Alert"

    def test_returns_empty_list_when_no_rules(self, client: TestClient, store: AsyncMock) -> None:
        store.list_rules = AsyncMock(return_value=[])
        with _config_and_redis_patch()[0], _config_and_redis_patch()[1]:
            resp = client.get("/api/scheduler/rules")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_503_when_store_not_initialized(self) -> None:
        app = _make_app()  # no trigger_store on state
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.get("/api/scheduler/rules")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/scheduler/rules
# ---------------------------------------------------------------------------


class TestCreateRule:
    def test_creates_rule_returns_201(self, client: TestClient, store: AsyncMock) -> None:
        rule = _mock_rule()
        store.count_rules = AsyncMock(return_value=0)
        store.create_rule = AsyncMock(return_value=rule)
        store.get_last_triggered_at = AsyncMock(return_value=None)

        body = {
            "name": "BTC Alert",
            "trigger_type": "price_threshold",
            "pair": "BTC/USDT",
            "parameters": {"direction": "below", "price": 50000},
            "cooldown_minutes": 30,
        }
        with _config_and_redis_patch()[0], _config_and_redis_patch()[1]:
            resp = client.post("/api/scheduler/rules", json=body)

        assert resp.status_code == 201
        assert resp.json()["id"] == "rule-abc-123"

    def test_returns_422_when_max_rules_exceeded(self, client: TestClient, store: AsyncMock) -> None:
        store.count_rules = AsyncMock(return_value=50)

        body = {
            "name": "New Rule",
            "trigger_type": "funding_rate",
            "pair": "ETH/USDT",
            "parameters": {"threshold_pct": 0.1},
            "cooldown_minutes": 30,
        }
        with patch("cryptotrader.config.load_config", return_value=_mock_config(max_rules=50)):
            resp = client.post("/api/scheduler/rules", json=body)

        assert resp.status_code == 422

    def test_invalid_trigger_type_returns_422(self, client: TestClient, store: AsyncMock) -> None:
        body = {
            "name": "Bad Rule",
            "trigger_type": "invalid_type",
            "pair": "BTC/USDT",
            "parameters": {},
            "cooldown_minutes": 30,
        }
        resp = client.post("/api/scheduler/rules", json=body)
        assert resp.status_code == 422

    def test_missing_required_fields_returns_422(self, client: TestClient, store: AsyncMock) -> None:
        resp = client.post("/api/scheduler/rules", json={"name": "Incomplete"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/scheduler/rules/{rule_id}
# ---------------------------------------------------------------------------


class TestGetRule:
    def test_returns_rule_by_id(self, client: TestClient, store: AsyncMock) -> None:
        rule = _mock_rule()
        store.get_rule = AsyncMock(return_value=rule)
        store.get_last_triggered_at = AsyncMock(return_value=None)

        with _config_and_redis_patch()[0], _config_and_redis_patch()[1]:
            resp = client.get("/api/scheduler/rules/rule-abc-123")

        assert resp.status_code == 200
        assert resp.json()["id"] == "rule-abc-123"

    def test_returns_404_for_unknown_rule(self, client: TestClient, store: AsyncMock) -> None:
        store.get_rule = AsyncMock(return_value=None)
        resp = client.get("/api/scheduler/rules/no-such-rule")
        assert resp.status_code == 404

    def test_returns_in_cooldown_flag(self, client: TestClient, store: AsyncMock) -> None:
        rule = _mock_rule()
        store.get_rule = AsyncMock(return_value=rule)
        store.get_last_triggered_at = AsyncMock(return_value=None)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=_mock_redis(in_cooldown=True)),
        ):
            resp = client.get("/api/scheduler/rules/rule-abc-123")

        assert resp.status_code == 200
        assert resp.json()["in_cooldown"] is True


# ---------------------------------------------------------------------------
# PUT /api/scheduler/rules/{rule_id}
# ---------------------------------------------------------------------------


class TestUpdateRule:
    def test_updates_rule_returns_200(self, client: TestClient, store: AsyncMock) -> None:
        updated = _mock_rule(name="Updated Name")
        store.update_rule = AsyncMock(return_value=updated)
        store.get_last_triggered_at = AsyncMock(return_value=None)

        body = {
            "name": "Updated Name",
            "trigger_type": "price_threshold",
            "pair": "BTC/USDT",
            "parameters": {"direction": "above", "price": 60000},
            "cooldown_minutes": 60,
        }
        with _config_and_redis_patch()[0], _config_and_redis_patch()[1]:
            resp = client.put("/api/scheduler/rules/rule-abc-123", json=body)

        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Name"

    def test_returns_404_when_rule_not_found(self, client: TestClient, store: AsyncMock) -> None:
        store.update_rule = AsyncMock(return_value=None)
        body = {
            "name": "Ghost",
            "trigger_type": "funding_rate",
            "pair": "BTC/USDT",
            "parameters": {},
            "cooldown_minutes": 30,
        }
        resp = client.put("/api/scheduler/rules/no-such-rule", json=body)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /api/scheduler/rules/{rule_id}/toggle
# ---------------------------------------------------------------------------


class TestToggleRule:
    def test_toggle_returns_updated_rule(self, client: TestClient, store: AsyncMock) -> None:
        toggled = _mock_rule(enabled=False)
        store.toggle_rule = AsyncMock(return_value=toggled)
        store.get_last_triggered_at = AsyncMock(return_value=None)

        with _config_and_redis_patch()[0], _config_and_redis_patch()[1]:
            resp = client.patch("/api/scheduler/rules/rule-abc-123/toggle")

        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_toggle_returns_404_for_unknown_rule(self, client: TestClient, store: AsyncMock) -> None:
        store.toggle_rule = AsyncMock(return_value=None)
        resp = client.patch("/api/scheduler/rules/ghost/toggle")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/scheduler/rules/{rule_id}
# ---------------------------------------------------------------------------


class TestDeleteRule:
    def test_delete_returns_204(self, client: TestClient, store: AsyncMock) -> None:
        store.delete_rule = AsyncMock(return_value=True)
        resp = client.delete("/api/scheduler/rules/rule-abc-123")
        assert resp.status_code == 204
        assert resp.content == b""

    def test_delete_returns_404_when_not_found(self, client: TestClient, store: AsyncMock) -> None:
        store.delete_rule = AsyncMock(return_value=False)
        resp = client.delete("/api/scheduler/rules/ghost")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/scheduler/triggers
# ---------------------------------------------------------------------------


class TestListTriggers:
    def test_returns_paginated_events(self, client: TestClient, store: AsyncMock) -> None:
        event = _mock_event()
        store.list_events = AsyncMock(return_value=([event], 1))

        resp = client.get("/api/scheduler/triggers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["page"] == 1
        assert data["size"] == 20
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == "evt-001"

    def test_pagination_params_forwarded(self, client: TestClient, store: AsyncMock) -> None:
        store.list_events = AsyncMock(return_value=([], 0))
        resp = client.get("/api/scheduler/triggers?page=2&size=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 2
        assert data["size"] == 10
        store.list_events.assert_awaited_once_with(2, 10, rule_id=None)

    def test_filter_by_rule_id(self, client: TestClient, store: AsyncMock) -> None:
        store.list_events = AsyncMock(return_value=([], 0))
        resp = client.get("/api/scheduler/triggers?rule_id=rule-abc-123")
        assert resp.status_code == 200
        store.list_events.assert_awaited_once_with(1, 20, rule_id="rule-abc-123")


# ---------------------------------------------------------------------------
# GET /api/scheduler/triggers/{event_id}
# ---------------------------------------------------------------------------


class TestGetTriggerEvent:
    def test_returns_event_by_id(self, client: TestClient, store: AsyncMock) -> None:
        event = _mock_event()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = AsyncMock(return_value=event)
        store._session = AsyncMock(return_value=mock_session)

        resp = client.get("/api/scheduler/triggers/evt-001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "evt-001"
        assert resp.json()["rule_id"] == "rule-abc-123"

    def test_returns_404_when_event_not_found(self, client: TestClient, store: AsyncMock) -> None:
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = AsyncMock(return_value=None)
        store._session = AsyncMock(return_value=mock_session)

        resp = client.get("/api/scheduler/triggers/nonexistent")
        assert resp.status_code == 404
