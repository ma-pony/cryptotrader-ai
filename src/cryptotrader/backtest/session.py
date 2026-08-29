"""Backtest session storage for cycle records and aggregate results."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cryptotrader._compat import UTC

if TYPE_CHECKING:
    from cryptotrader.backtest.result import BacktestResult
    from cryptotrader.journal.models import MultiVenueCycleRecord

logger = logging.getLogger(__name__)

_SESSIONS_DIR = Path.home() / ".cryptotrader" / "backtest_sessions"


def generate_session_id(pair: str, interval: str, start: str, end: str) -> str:
    """Generate a unique session ID from backtest parameters."""
    pair_clean = pair.replace("/", "_")
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{pair_clean}_{start}_{end}_{interval}_{ts}"


def get_session_dir(session_id: str) -> Path:
    """Get session directory path, creating it if needed."""
    path = _SESSIONS_DIR / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_cycles(session_id: str, records: list[MultiVenueCycleRecord]) -> Path:
    """Serialize immutable multi-venue cycle records to a session JSONL file."""
    session_dir = get_session_dir(session_id)
    path = session_dir / "cycles.jsonl"
    with open(path, "w") as f:
        for cycle in records:
            record = _serialize_cycle(cycle)
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    logger.info("Saved %d cycles to %s", len(records), path)
    return path


def save_result(session_id: str, result: BacktestResult) -> Path:
    """Save backtest result summary to session directory."""
    session_dir = get_session_dir(session_id)
    path = session_dir / "result.json"
    data = asdict(result)
    # Remove large equity_curve from summary (keep only summary stats)
    data.pop("equity_curve", None)
    data.pop("cycle_records", None)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, default=str, indent=2)
    return path


def load_cycles(session_id: str) -> list[dict]:
    """Load cycle records from a session's JSONL file."""
    path = _SESSIONS_DIR / session_id / "cycles.jsonl"
    if not path.exists():
        return []
    cycles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cycles.append(json.loads(line))
    return cycles


def list_sessions() -> list[str]:
    """List all session IDs."""
    if not _SESSIONS_DIR.exists():
        return []
    return sorted(d.name for d in _SESSIONS_DIR.iterdir() if d.is_dir())


def _serialize_cycle(cycle: MultiVenueCycleRecord) -> dict:
    data = asdict(cycle)
    if hasattr(data["created_at"], "isoformat"):
        data["created_at"] = data["created_at"].isoformat()
    return data
