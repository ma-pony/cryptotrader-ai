"""Strict local persistence for named backtest sessions."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cryptotrader._compat import UTC
from cryptotrader.cycle_serialization import json_value

if TYPE_CHECKING:
    from cryptotrader.backtest.result import BacktestResult
    from cryptotrader.journal.models import MultiVenueCycleRecord

logger = logging.getLogger(__name__)

_SESSIONS_DIR = Path.home() / ".cryptotrader" / "backtest_sessions"
_SAFE_SESSION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def generate_session_id(pair: str, interval: str, start: str, end: str) -> str:
    pair_clean = pair.replace("/", "_").replace(":", "_")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{pair_clean}_{start}_{end}_{interval}_{timestamp}"


def get_session_dir(session_name: str) -> Path:
    """Return a validated session path rooted beneath the backtest directory."""
    if not _SAFE_SESSION_NAME.fullmatch(session_name):
        raise ValueError("session_name must contain only letters, digits, underscores, and hyphens")
    root = _SESSIONS_DIR.resolve()
    path = (root / session_name).resolve()
    if path.parent != root:
        raise ValueError("session_name must name one direct backtest session")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_session(session_name: str, params: dict[str, Any], result: BacktestResult) -> Path:
    """Persist one complete named result with strict JSON-safe data only."""
    session_dir = get_session_dir(session_name)
    _write_json(session_dir / "params.json", params)
    _write_json(session_dir / "result.json", _result_summary(result))
    save_cycles(session_name, result.cycle_records)
    return session_dir


def save_cycles(session_name: str, records: list[MultiVenueCycleRecord]) -> Path:
    path = get_session_dir(session_name) / "cycles.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for cycle in records:
            handle.write(json.dumps(json_value(_serialize_cycle(cycle)), ensure_ascii=False) + "\n")
    return path


def save_result(session_name: str, result: BacktestResult) -> Path:
    path = get_session_dir(session_name) / "result.json"
    _write_json(path, _result_summary(result))
    return path


def load_session(session_name: str) -> dict[str, Any] | None:
    try:
        session_dir = get_session_dir(session_name)
    except ValueError:
        return None
    result_path = session_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        result = _read_json(result_path)
        params_path = session_dir / "params.json"
        params = _read_json(params_path) if params_path.exists() else {}
    except (OSError, json.JSONDecodeError, TypeError):
        logger.warning("Failed to load backtest session %s", session_name, exc_info=True)
        return None
    return {
        "name": session_name,
        "params": params,
        "result": result,
        "saved_at": datetime.fromtimestamp(result_path.stat().st_mtime, UTC).isoformat(),
    }


def load_cycles(session_name: str) -> list[dict[str, Any]]:
    try:
        path = get_session_dir(session_name) / "cycles.jsonl"
    except ValueError:
        return []
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load backtest cycle records for %s", session_name, exc_info=True)
        return []


def list_sessions() -> list[str]:
    if not _SESSIONS_DIR.exists():
        return []
    return sorted(
        path.name for path in _SESSIONS_DIR.iterdir() if path.is_dir() and _SAFE_SESSION_NAME.fullmatch(path.name)
    )


def _result_summary(result: BacktestResult) -> dict[str, Any]:
    summary = asdict(result)
    summary.pop("equity_curve", None)
    summary.pop("cycle_records", None)
    return json_value(summary)


def _serialize_cycle(cycle: MultiVenueCycleRecord) -> dict[str, Any]:
    data = asdict(cycle)
    if hasattr(data["created_at"], "isoformat"):
        data["created_at"] = data["created_at"].isoformat()
    return data


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_value(value), handle, ensure_ascii=False, separators=(",", ":"))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("backtest session JSON must be an object")
    return value
