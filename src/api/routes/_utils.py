"""Shared helpers for API route modules."""

from __future__ import annotations

from datetime import datetime

from cryptotrader._compat import UTC


def coerce_timestamp(ts: object) -> datetime | None:
    """Normalize a ``commit.timestamp`` (str or datetime) into a UTC-aware datetime.

    Returns ``None`` when ``ts`` is not a supported type or cannot be parsed. Any
    naive datetime is promoted to UTC. Accepts ISO 8601 strings including the ``Z``
    suffix.
    """
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
    if isinstance(ts, str):
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None
