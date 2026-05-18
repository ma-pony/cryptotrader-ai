"""spec 022 FR-022-15 — Lightweight event journal write helpers.

These helpers write to the `journal` table (separate from `decision_commits`).
The `journal` table is an append-only event log:

    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trace_id VARCHAR(36),
    event_type VARCHAR(64) NOT NULL,
    pair VARCHAR(50),
    payload JSONB NOT NULL DEFAULT '{}'

The SQL view `events_heartbeat` projects a filtered subset of this table,
consumed by GET /api/events/heartbeat.

Callers:
  - nodes/journal.py re-exports record_phase1_rejection / record_evolution_event
  - ops/daemon.py calls record_evolution_event at 3 hooks
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from cryptotrader._compat import UTC

logger = logging.getLogger(__name__)


async def _write_journal_event(
    database_url: str | None,
    *,
    trace_id: str,
    event_type: str,
    pair: str | None,
    payload: dict[str, Any],
) -> None:
    """Write a single event record to the `journal` table.

    Soft-fail: any DB error is logged at WARNING level and swallowed so that
    calling nodes are never blocked by journal write failures.

    Creates the `journal` table automatically if it does not exist (idempotent
    CREATE TABLE IF NOT EXISTS).
    """
    from datetime import datetime

    from sqlalchemy import text

    from cryptotrader.db import get_engine

    if not database_url:
        logger.debug("_write_journal_event: no database_url, skipping write (event_type=%s)", event_type)
        return

    try:
        engine = await get_engine(database_url)
        async with engine.begin() as conn:
            # Ensure the journal table exists (idempotent DDL)
            await conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS journal ("
                    "id BIGSERIAL PRIMARY KEY, "
                    "timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(), "
                    "trace_id VARCHAR(36), "
                    "event_type VARCHAR(64) NOT NULL, "
                    "pair VARCHAR(50), "
                    "payload JSONB NOT NULL DEFAULT '{}'"
                    ")"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO journal (timestamp, trace_id, event_type, pair, payload) "
                    "VALUES (:ts, :trace_id, :event_type, :pair, CAST(:payload AS jsonb))"
                ),
                {
                    "ts": datetime.now(UTC),
                    "trace_id": trace_id,
                    "event_type": event_type,
                    "pair": pair,
                    "payload": json.dumps(payload),
                },
            )
    except Exception:
        logger.warning(
            "journal event write failed (soft-fail) event_type=%s trace_id=%s",
            event_type,
            trace_id,
            exc_info=True,
        )


async def record_phase1_rejection(
    trace_id: str,
    pair: str,
    reason: str,
    payload: dict[str, Any] | None = None,
    *,
    database_url: str | None = None,
) -> None:
    """Write a phase1_rejected event to the journal table.

    Args:
        trace_id: cycle trace_id (UUID string).
        pair: trading pair canonical string (e.g. "SOL/USDT:USDT").
        reason: rejection reason code: low_rr / stop_too_tight / missing_sl_tp /
                direction_inverted_long / direction_inverted_short.
        payload: optional additional context dict.
        database_url: PostgreSQL connection URL; resolved from config if None.
    """
    if database_url is None:
        try:
            from cryptotrader.config import load_config

            database_url = load_config().infrastructure.database_url
        except Exception:
            database_url = None

    full_payload: dict[str, Any] = {"reason": reason}
    if payload:
        full_payload.update(payload)

    await _write_journal_event(
        database_url,
        trace_id=trace_id or str(uuid.uuid4()),
        event_type="phase1_rejected",
        pair=pair,
        payload=full_payload,
    )


async def record_evolution_event(
    event_subtype: str,
    artifact_name: str,
    payload: dict[str, Any] | None = None,
    *,
    database_url: str | None = None,
) -> None:
    """Write an evolution_event to the journal table.

    Args:
        event_subtype: "new_pattern" / "new_skill_draft" / "pareto_rerank".
        artifact_name: slug or name of the artifact (pattern slug, skill name, etc.).
        payload: optional additional context dict.
        database_url: PostgreSQL connection URL; resolved from config if None.
    """
    if database_url is None:
        try:
            from cryptotrader.config import load_config

            database_url = load_config().infrastructure.database_url
        except Exception:
            database_url = None

    full_payload: dict[str, Any] = {
        "event_subtype": event_subtype,
        "artifact_name": artifact_name,
    }
    if payload:
        full_payload.update(payload)

    await _write_journal_event(
        database_url,
        trace_id=str(uuid.uuid4()),
        event_type="evolution_event",
        pair=None,
        payload=full_payload,
    )
