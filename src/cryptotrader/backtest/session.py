"""Backtest session storage — serialize commits and results to session directories."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.backtest.result import BacktestResult
    from cryptotrader.models import DecisionCommit, ExperienceMemory

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


def save_commits(session_id: str, commits: list[DecisionCommit]) -> Path:
    """Serialize backtest commits to JSONL file in session directory."""
    session_dir = get_session_dir(session_id)
    path = session_dir / "commits.jsonl"
    with open(path, "w") as f:
        for dc in commits:
            record = _serialize_commit(dc)
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    logger.info("Saved %d commits to %s", len(commits), path)
    return path


def save_result(session_id: str, result: BacktestResult) -> Path:
    """Save backtest result summary to session directory."""
    session_dir = get_session_dir(session_id)
    path = session_dir / "result.json"
    data = asdict(result)
    # Remove large equity_curve from summary (keep only summary stats)
    data.pop("equity_curve", None)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, default=str, indent=2)
    return path


def save_experience(session_id: str, experience: dict[str, ExperienceMemory]) -> Path:
    """Save distilled experience to session directory."""
    session_dir = get_session_dir(session_id)
    path = session_dir / "experience.json"
    data = {agent_id: asdict(mem) for agent_id, mem in experience.items()}
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, default=str, indent=2)
    return path


def load_commits(session_id: str) -> list[dict]:
    """Load commits from a session's JSONL file."""
    path = _SESSIONS_DIR / session_id / "commits.jsonl"
    if not path.exists():
        return []
    commits = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                commits.append(json.loads(line))
    return commits


def list_sessions() -> list[str]:
    """List all session IDs."""
    if not _SESSIONS_DIR.exists():
        return []
    return sorted(d.name for d in _SESSIONS_DIR.iterdir() if d.is_dir())


def _serialize_commit(dc: DecisionCommit) -> dict:
    """Convert DecisionCommit to a JSON-serializable dict."""
    data = asdict(dc)
    # Convert datetime objects to ISO strings
    if "timestamp" in data and hasattr(data["timestamp"], "isoformat"):
        data["timestamp"] = data["timestamp"].isoformat()
    return data
