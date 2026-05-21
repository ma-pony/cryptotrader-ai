"""spec 022 T011 — Migration: create `journal` table + `events_heartbeat` SQL view.

Run with:
    uv run python -m api.db_migrations.events_heartbeat_view

The `journal` table is a lightweight append-only event log used by:
  - journal.record_phase1_rejection()
  - journal.record_evolution_event()

The `events_heartbeat` view projects the 6 event types consumed by
GET /api/events/heartbeat.

SQLAlchemy 2.x async style.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_CREATE_JOURNAL_TABLE = """
CREATE TABLE IF NOT EXISTS journal (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trace_id VARCHAR(36),
    event_type VARCHAR(64) NOT NULL,
    pair VARCHAR(50),
    payload JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS ix_journal_timestamp ON journal (timestamp DESC);
CREATE INDEX IF NOT EXISTS ix_journal_event_type ON journal (event_type);
CREATE INDEX IF NOT EXISTS ix_journal_trace_id ON journal (trace_id);
"""

_CREATE_EVENTS_HEARTBEAT_VIEW = """
CREATE OR REPLACE VIEW events_heartbeat AS
SELECT
    timestamp,
    trace_id,
    event_type,
    pair,
    payload
FROM journal
WHERE event_type IN (
    'verdict_decision',
    'journal_trade_committed',
    'risk_gate_rejected',
    'phase1_rejected',
    'evolution_event',
    'oco_state_change'
)
ORDER BY timestamp DESC, trace_id DESC;
"""


async def apply(database_url: str) -> None:
    """Create journal table and events_heartbeat view.

    Idempotent: uses CREATE TABLE IF NOT EXISTS and CREATE OR REPLACE VIEW.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(database_url, echo=False)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(_CREATE_JOURNAL_TABLE))
            await conn.execute(text(_CREATE_EVENTS_HEARTBEAT_VIEW))
        logger.info("events_heartbeat_view migration applied successfully")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    import os

    import dotenv

    dotenv.load_dotenv()
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("CRYPTOTRADER_INFRASTRUCTURE__DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL not set")
    asyncio.run(apply(db_url))
    raise SystemExit(0)  # success
