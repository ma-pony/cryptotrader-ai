"""spec 022 T015 — Tests for GET /api/events/heartbeat.

6 test cases:
  (a) basic poll returns recent events
  (b) cursor pagination — second poll with next_cursor returns 0 events when empty
  (c) since + cursor both provided → cursor wins
  (d) ?types=verdict,trade filter
  (e) future since returns empty (no 500)
  (f) limit upper bound
"""

from __future__ import annotations

import base64
import os
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from cryptotrader._compat import UTC

os.environ.setdefault("AUTH_MODE", "disabled")
os.environ.setdefault("API_KEY", "test-key-022")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(
    ts: datetime,
    trace_id: str,
    event_type: str = "phase1_rejected",
    pair: str | None = "BTC/USDT:USDT",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": ts,
        "trace_id": trace_id,
        "event_type": event_type,
        "pair": pair,
        "payload": payload or {"reason": "low_rr"},
    }


_NOW = datetime(2026, 5, 18, 10, 0, 0, tzinfo=UTC)

_SAMPLE_ROWS = [
    _make_row(_NOW - timedelta(seconds=30), "trace-001", "verdict_decision", payload={"action": "hold"}),
    _make_row(_NOW - timedelta(seconds=20), "trace-002", "journal_trade_committed", payload={"algo_id": "x1"}),
    _make_row(_NOW - timedelta(seconds=10), "trace-003", "risk_gate_rejected", payload={"check_name": "daily_loss"}),
    _make_row(_NOW - timedelta(seconds=5), "trace-004", "phase1_rejected", payload={"reason": "low_rr"}),
]


@pytest.fixture
def client():
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helper: encode cursor as events.py does
# ---------------------------------------------------------------------------


def _encode_cursor(ts: datetime, trace_id: str) -> str:
    raw = f"{ts.isoformat()}|{trace_id}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


# ---------------------------------------------------------------------------
# (a) basic poll returns recent events
# ---------------------------------------------------------------------------


class TestBasicPoll:
    def test_returns_recent_events(self, client: TestClient) -> None:
        """Poll without filter should return items list + next_cursor."""
        with patch(
            "api.routes.events._query_heartbeat_events",
            new=AsyncMock(return_value=_SAMPLE_ROWS),
        ):
            resp = client.get("/api/events/heartbeat")
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert "next_cursor" in body
        assert len(body["items"]) == 4
        # Check event_type mapping (internal → external)
        types_seen = {item["event_type"] for item in body["items"]}
        assert "verdict" in types_seen
        assert "trade" in types_seen
        assert "rejection" in types_seen
        assert "phase1_rejected" in types_seen

    def test_empty_db_returns_empty_items_null_cursor(self, client: TestClient) -> None:
        with patch(
            "api.routes.events._query_heartbeat_events",
            new=AsyncMock(return_value=[]),
        ):
            resp = client.get("/api/events/heartbeat")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []
        assert body["next_cursor"] is None


# ---------------------------------------------------------------------------
# (b) cursor pagination — second poll with next_cursor returns 0 events
# ---------------------------------------------------------------------------


class TestCursorPagination:
    def test_second_poll_with_cursor_returns_empty(self, client: TestClient) -> None:
        """After fetching all events, second poll with cursor should return no new events."""
        # First poll: get events
        with patch(
            "api.routes.events._query_heartbeat_events",
            new=AsyncMock(return_value=_SAMPLE_ROWS),
        ):
            resp1 = client.get("/api/events/heartbeat")
        assert resp1.status_code == 200
        cursor = resp1.json()["next_cursor"]
        assert cursor is not None

        # Second poll with cursor — no new events
        with patch(
            "api.routes.events._query_heartbeat_events",
            new=AsyncMock(return_value=[]),
        ):
            resp2 = client.get(f"/api/events/heartbeat?cursor={cursor}")
        assert resp2.status_code == 200
        body2 = resp2.json()
        assert body2["items"] == []
        assert body2["next_cursor"] is None

    def test_cursor_is_passed_to_query(self, client: TestClient) -> None:
        """Cursor parameter should be decoded and passed to _query_heartbeat_events."""
        cursor = _encode_cursor(_NOW - timedelta(seconds=5), "trace-004")
        mock_query = AsyncMock(return_value=[])

        with patch("api.routes.events._query_heartbeat_events", new=mock_query):
            resp = client.get(f"/api/events/heartbeat?cursor={cursor}")
        assert resp.status_code == 200
        # Verify cursor was passed correctly
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["cursor_ts"] is not None
        assert call_kwargs["cursor_trace_id"] == "trace-004"


# ---------------------------------------------------------------------------
# (c) since + cursor both provided → cursor wins
# ---------------------------------------------------------------------------


class TestCursorVsSince:
    def test_cursor_wins_over_since(self, client: TestClient) -> None:
        """When both cursor and since are provided, cursor takes precedence."""
        cursor = _encode_cursor(_NOW - timedelta(seconds=5), "trace-004")
        since_ts = (_NOW - timedelta(hours=1)).isoformat()
        mock_query = AsyncMock(return_value=[])

        with patch("api.routes.events._query_heartbeat_events", new=mock_query):
            resp = client.get(f"/api/events/heartbeat?cursor={cursor}&since={since_ts}")
        assert resp.status_code == 200
        call_kwargs = mock_query.call_args.kwargs
        # cursor_ts should be set (cursor wins)
        assert call_kwargs["cursor_ts"] is not None
        assert call_kwargs["cursor_trace_id"] == "trace-004"
        # since should be None because cursor won
        assert call_kwargs["since"] is None


# ---------------------------------------------------------------------------
# (d) ?types=verdict,trade filter
# ---------------------------------------------------------------------------


class TestTypesFilter:
    def test_types_filter_maps_to_db_event_types(self, client: TestClient) -> None:
        """?types=verdict,trade should filter to verdict_decision + journal_trade_committed."""
        mock_query = AsyncMock(return_value=[])

        with patch("api.routes.events._query_heartbeat_events", new=mock_query):
            resp = client.get("/api/events/heartbeat?types=verdict,trade")
        assert resp.status_code == 200
        call_kwargs = mock_query.call_args.kwargs
        db_types = set(call_kwargs["db_event_types"])
        assert "verdict_decision" in db_types
        assert "journal_trade_committed" in db_types
        # Should not include others
        assert "phase1_rejected" not in db_types
        assert "risk_gate_rejected" not in db_types

    def test_types_filter_response_items(self, client: TestClient) -> None:
        """Items should only contain mapped types when filter applied."""
        verdict_row = _make_row(
            _NOW - timedelta(seconds=10), "trace-v1", "verdict_decision", payload={"action": "long"}
        )
        trade_row = _make_row(_NOW - timedelta(seconds=5), "trace-t1", "journal_trade_committed", payload={"sz": 1.0})

        with patch(
            "api.routes.events._query_heartbeat_events",
            new=AsyncMock(return_value=[verdict_row, trade_row]),
        ):
            resp = client.get("/api/events/heartbeat?types=verdict,trade")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 2
        types_seen = {item["event_type"] for item in items}
        assert types_seen == {"verdict", "trade"}


# ---------------------------------------------------------------------------
# (e) future since returns empty (no 500)
# ---------------------------------------------------------------------------


class TestFutureSince:
    def test_future_since_returns_empty_not_500(self, client: TestClient) -> None:
        """A future `since` timestamp should return empty items without error."""
        future_ts = (datetime.now(UTC) + timedelta(hours=24)).isoformat()

        with patch(
            "api.routes.events._query_heartbeat_events",
            new=AsyncMock(return_value=[]),
        ):
            resp = client.get(f"/api/events/heartbeat?since={future_ts}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []
        assert body["next_cursor"] is None

    def test_invalid_since_is_ignored(self, client: TestClient) -> None:
        """An invalid `since` value should be ignored (not 422/500)."""
        with patch(
            "api.routes.events._query_heartbeat_events",
            new=AsyncMock(return_value=[]),
        ):
            resp = client.get("/api/events/heartbeat?since=not-a-date")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# (f) limit upper bound
# ---------------------------------------------------------------------------


class TestLimitBound:
    def test_limit_200_is_accepted(self, client: TestClient) -> None:
        """limit=200 (max) should not result in an error."""
        with patch(
            "api.routes.events._query_heartbeat_events",
            new=AsyncMock(return_value=[]),
        ):
            resp = client.get("/api/events/heartbeat?limit=200")
        assert resp.status_code == 200

    def test_limit_over_200_is_rejected(self, client: TestClient) -> None:
        """limit=201 should result in 422 (FastAPI validation)."""
        resp = client.get("/api/events/heartbeat?limit=201")
        assert resp.status_code == 422

    def test_limit_default_is_50(self, client: TestClient) -> None:
        """Default limit is 50; query should be called with limit=50."""
        mock_query = AsyncMock(return_value=[])

        with patch("api.routes.events._query_heartbeat_events", new=mock_query):
            resp = client.get("/api/events/heartbeat")
        assert resp.status_code == 200
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["limit"] == 50
