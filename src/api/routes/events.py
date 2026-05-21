"""spec 022 FR-022-10 — Heartbeat events pull-based subscription endpoint.

GET /api/events/heartbeat?since=&cursor=&types=&limit=

Reuses the `journal` table via the `events_heartbeat` SQL view.
Cursor-based pagination: base64-url "{timestamp.isoformat()}|{trace_id}".

Observability:
  - HeartbeatPollAggregator.record(client_identifier)
  - OTel span "events.heartbeat.poll" with attr client_identifier (API_KEY sha256[:8])
"""

from __future__ import annotations

import base64
import hashlib
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from api.dependencies import verify_api_key
from cryptotrader._compat import UTC

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Mapping from heartbeat event_type (external API name) to journal event_type
_TYPE_MAP: dict[str, str] = {
    "verdict": "verdict_decision",
    "trade": "journal_trade_committed",
    "rejection": "risk_gate_rejected",
    "phase1_rejected": "phase1_rejected",
    "evolution": "evolution_event",
    "oco_state_change": "oco_state_change",
}

# Inverse map: journal event_type → external API name
_TYPE_MAP_INV: dict[str, str] = {v: k for k, v in _TYPE_MAP.items()}

# Default included types (must 4)
_DEFAULT_TYPES = frozenset({"verdict", "trade", "rejection", "phase1_rejected"})


class HeartbeatEvent(BaseModel):
    """Single heartbeat event from the events_heartbeat view."""

    timestamp: datetime
    trace_id: str | None
    event_type: str
    pair: str | None
    payload: dict[str, Any]


class HeartbeatResponse(BaseModel):
    """Response from GET /api/events/heartbeat."""

    items: list[HeartbeatEvent]
    next_cursor: str | None


# ---------------------------------------------------------------------------
# Cursor helpers
# ---------------------------------------------------------------------------


def encode_cursor(timestamp: datetime, trace_id: str) -> str:
    """Encode (timestamp, trace_id) into a base64-url cursor token.

    Format: base64url("{timestamp.isoformat()}|{trace_id}")
    """
    raw = f"{timestamp.isoformat()}|{trace_id}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def decode_cursor(token: str) -> tuple[datetime, str]:
    """Decode a cursor token back to (timestamp, trace_id).

    Raises ValueError if the token is malformed.
    """
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        ts_str, trace_id = decoded.split("|", 1)
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return ts, trace_id
    except Exception as exc:
        raise ValueError(f"Invalid cursor token: {exc}") from exc


def _client_identifier(request: Request) -> str:
    """Derive an 8-char client identifier from the API_KEY header (sha256[:8])."""
    key = request.headers.get("X-API-Key", "") or request.headers.get("Authorization", "")
    # Strip "Bearer " prefix if present
    if key.startswith("Bearer "):
        key = key[7:]
    if not key:
        return "anonymous"
    return hashlib.sha256(key.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# DB helpers (in-process, no external tables except journal)
# ---------------------------------------------------------------------------


# Pre-defined SQL templates (no user input in table/column names)
_SQL_QUERY_BASE = (
    "SELECT timestamp, trace_id, event_type, pair, payload "
    "FROM journal "
    "WHERE event_type = ANY(:types) "
    "ORDER BY timestamp DESC, COALESCE(trace_id, '') DESC "
    "LIMIT :limit"
)

_SQL_QUERY_SINCE = (
    "SELECT timestamp, trace_id, event_type, pair, payload "
    "FROM journal "
    "WHERE event_type = ANY(:types) AND timestamp > :cutoff_ts "
    "ORDER BY timestamp DESC, COALESCE(trace_id, '') DESC "
    "LIMIT :limit"
)

_SQL_QUERY_CURSOR = (
    "SELECT timestamp, trace_id, event_type, pair, payload "
    "FROM journal "
    "WHERE event_type = ANY(:types) "
    "AND (timestamp, COALESCE(trace_id, '')) < (:cutoff_ts, :cursor_tid) "
    "ORDER BY timestamp DESC, COALESCE(trace_id, '') DESC "
    "LIMIT :limit"
)


async def _query_heartbeat_events(
    database_url: str | None,
    *,
    since: datetime | None,
    cursor_ts: datetime | None,
    cursor_trace_id: str | None,
    db_event_types: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Query journal table for heartbeat events.

    Returns a list of row dicts: {timestamp, trace_id, event_type, pair, payload}.
    cursor_ts takes priority over since when both are provided.
    """
    if not database_url:
        return []

    from sqlalchemy import text

    from cryptotrader.db import get_engine

    # Pick the right pre-built SQL template
    params: dict[str, Any] = {"types": db_event_types, "limit": limit}
    if cursor_ts is not None:
        sql = _SQL_QUERY_CURSOR
        params["cutoff_ts"] = cursor_ts
        params["cursor_tid"] = cursor_trace_id or ""
    elif since is not None:
        sql = _SQL_QUERY_SINCE
        params["cutoff_ts"] = since
    else:
        sql = _SQL_QUERY_BASE

    try:
        engine = await get_engine(database_url)
        async with engine.connect() as conn:
            result = await conn.execute(text(sql), params)
            rows = result.fetchall()
            return [
                {
                    "timestamp": row[0],
                    "trace_id": row[1],
                    "event_type": row[2],
                    "pair": row[3],
                    "payload": row[4] if isinstance(row[4], dict) else {},
                }
                for row in rows
            ]
    except Exception:
        logger.warning("events_heartbeat query failed", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Internal helpers extracted to keep get_heartbeat_events under C901 limit
# ---------------------------------------------------------------------------


def _parse_types_param(types: str | None) -> list[str]:
    """Map ?types= query param to journal event_type strings."""
    requested = {t.strip() for t in types.split(",") if t.strip()} if types else set(_DEFAULT_TYPES)
    mapped = [_TYPE_MAP[t] for t in requested if t in _TYPE_MAP]
    return mapped if mapped else [_TYPE_MAP[t] for t in _DEFAULT_TYPES]


def _parse_cursor_and_since(
    cursor: str | None,
    since: str | None,
) -> tuple[datetime | None, str | None, datetime | None]:
    """Return (cursor_ts, cursor_trace_id, since_dt).

    cursor wins over since when both provided.
    """
    if cursor:
        try:
            cursor_ts, cursor_trace_id = decode_cursor(cursor)
            return cursor_ts, cursor_trace_id, None
        except ValueError:
            logger.warning("invalid cursor token, ignoring: %s", cursor)

    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=UTC)
        except ValueError:
            logger.warning("invalid since param, ignoring: %s", since)

    return None, None, since_dt


def _rows_to_items(rows: list[dict[str, Any]]) -> list[HeartbeatEvent]:
    """Convert raw DB rows to HeartbeatEvent objects."""
    items: list[HeartbeatEvent] = []
    for row in rows:
        raw_ts = row["timestamp"]
        if isinstance(raw_ts, str):
            raw_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        if raw_ts is not None and raw_ts.tzinfo is None:
            raw_ts = raw_ts.replace(tzinfo=UTC)
        external_type = _TYPE_MAP_INV.get(row["event_type"], row["event_type"])
        items.append(
            HeartbeatEvent(
                timestamp=raw_ts,
                trace_id=row["trace_id"] or "",
                event_type=external_type,
                pair=row["pair"],
                payload=row["payload"],
            )
        )
    return items


def _emit_otel_span(client_id: str) -> None:
    """Emit OTel span for heartbeat poll (non-blocking)."""
    try:
        from opentelemetry import trace as otel_trace

        tracer = otel_trace.get_tracer(__name__)
        span = tracer.start_span("events.heartbeat.poll")
        span.set_attribute("client_identifier", client_id)
        span.end()
    except Exception:
        pass  # OTel not configured — non-blocking


def _record_poll_metrics(client_id: str, lag_seconds: float) -> None:
    """Record heartbeat poll in aggregators (non-blocking)."""
    try:
        from cryptotrader.observability.heartbeat_metrics import (
            get_heartbeat_poll_aggregator,
            get_heartbeat_poll_lag_aggregator,
        )

        get_heartbeat_poll_aggregator().record(client_id)
        get_heartbeat_poll_lag_aggregator().record(lag_seconds)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get("/heartbeat", response_model=HeartbeatResponse, dependencies=[Depends(verify_api_key)])
async def get_heartbeat_events(
    request: Request,
    since: str | None = Query(default=None, description="ISO8601 timestamp; return events after this time"),
    cursor: str | None = Query(default=None, description="Pagination cursor from previous response"),
    types: str | None = Query(
        default=None,
        description="Comma-separated event types: verdict,trade,rejection,phase1_rejected,evolution,oco_state_change",
    ),
    limit: int = Query(default=50, ge=1, le=200),
) -> HeartbeatResponse:
    """Poll the latest trading events from the journal table.

    - `cursor` wins over `since` when both are provided.
    - Default types: verdict, trade, rejection, phase1_rejected.
    - Returns `{items, next_cursor}`.
    """
    from time import time as _time

    poll_start = _time()
    client_id = _client_identifier(request)

    _emit_otel_span(client_id)

    db_event_types = _parse_types_param(types)
    cursor_ts, cursor_trace_id, since_dt = _parse_cursor_and_since(cursor, since)

    try:
        from cryptotrader.config import load_config

        database_url: str | None = load_config().infrastructure.database_url
    except Exception:
        database_url = None

    rows = await _query_heartbeat_events(
        database_url,
        since=since_dt,
        cursor_ts=cursor_ts,
        cursor_trace_id=cursor_trace_id,
        db_event_types=db_event_types,
        limit=limit,
    )

    items = _rows_to_items(rows)
    next_cursor: str | None = encode_cursor(items[-1].timestamp, items[-1].trace_id or "") if items else None
    _record_poll_metrics(client_id, _time() - poll_start)

    return HeartbeatResponse(items=items, next_cursor=next_cursor)
