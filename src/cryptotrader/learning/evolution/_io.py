"""spec 018 — evolution I/O helpers（模块级文件读写辅助函数）。

拆分自 provider.py 以控制单文件行数（< 400 行）。
公开函数供 provider.py 和 api/routes/memory.py 调用。
"""

from __future__ import annotations

import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.agents.skills.schema import CaseRecord, PatternRecord
    from cryptotrader.learning.evolution.ive import FailureClassification

logger = logging.getLogger(__name__)


def _time_decay(last_accessed: datetime, now: datetime) -> float:
    """计算时间衰减因子：exp(-days/30)，days 为距 last_accessed 的天数。"""
    try:
        delta = now - last_accessed
        days = max(0.0, delta.total_seconds() / 86400)
        return math.exp(-days / 30)
    except Exception:
        return 1.0


def _load_pattern_from_path(path: Path) -> PatternRecord | None:
    """从文件加载 PatternRecord，包含 spec 018 新增字段。"""
    try:
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter
        from cryptotrader.agents.skills.schema import PatternRecord, PnLTrack

        content = path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content, path=path)
        pt = fm.get("pnl_track", {})
        pnl_track = PnLTrack(
            cases=pt.get("cases", 0),
            wins=pt.get("wins", 0),
            win_rate=pt.get("win_rate", 0.0),
            avg_pnl=pt.get("avg_pnl", 0.0),
            last_active=pt.get("last_active", ""),
        )
        # spec 018 新字段
        last_accessed_raw = fm.get("last_accessed_at")
        last_modified_raw = fm.get("last_modified_at")
        now = datetime.now(UTC)

        def _parse_dt(raw: str | None) -> datetime:
            if not raw:
                return now
            try:
                dt = datetime.fromisoformat(str(raw))
                # 确保 timezone-aware
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except (ValueError, TypeError):
                return now

        return PatternRecord(
            name=fm.get("name", path.stem),
            agent=fm.get("agent", ""),
            description=fm.get("description", ""),
            body=body,
            regime_tags=fm.get("regime_tags", []),
            pnl_track=pnl_track,
            maturity=fm.get("maturity", "observed"),
            source_cycles=fm.get("source_cycles", []),
            created=_parse_dt(fm.get("created")),
            file_path=path,
            manually_edited=fm.get("manually_edited", False),
            version=fm.get("version", 1),
            importance=float(fm.get("importance", 0.5)),
            access_count=int(fm.get("access_count", 0)),
            last_accessed_at=_parse_dt(last_accessed_raw),
            last_modified_at=_parse_dt(last_modified_raw),
            fundamental_failure_streak=int(fm.get("fundamental_failure_streak", 0)),
        )
    except Exception:
        logger.warning("_load_pattern_from_path failed for %s", path, exc_info=True)
        return None


def _save_pattern_to_path(pattern: PatternRecord, path: Path) -> None:
    """保存 PatternRecord（含 spec 018 新字段）到文件（原子写）。"""
    from cryptotrader.agents.skills._frontmatter import render_frontmatter
    from cryptotrader.agents.skills._io import atomic_write

    fm: dict = {
        "name": pattern.name,
        "agent": pattern.agent,
        "description": pattern.description,
        "maturity": pattern.maturity,
        "manually_edited": pattern.manually_edited,
        "regime_tags": pattern.regime_tags,
        "pnl_track": {
            "cases": pattern.pnl_track.cases,
            "wins": pattern.pnl_track.wins,
            "win_rate": round(pattern.pnl_track.win_rate, 4),
            "avg_pnl": round(pattern.pnl_track.avg_pnl, 4),
            "last_active": pattern.pnl_track.last_active,
        },
        "source_cycles": pattern.source_cycles,
        "created": pattern.created.isoformat() if isinstance(pattern.created, datetime) else str(pattern.created),
        "version": pattern.version,
        # spec 018 新字段
        "importance": pattern.importance,
        "access_count": pattern.access_count,
        "last_accessed_at": pattern.last_accessed_at.isoformat()
        if isinstance(pattern.last_accessed_at, datetime)
        else str(pattern.last_accessed_at),
        "last_modified_at": pattern.last_modified_at.isoformat()
        if isinstance(pattern.last_modified_at, datetime)
        else str(pattern.last_modified_at),
        "fundamental_failure_streak": pattern.fundamental_failure_streak,
    }
    content = render_frontmatter(fm) + pattern.body
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, content)


def _load_case_from_path(path: Path) -> CaseRecord | None:
    """从文件加载 CaseRecord，包含 spec 018 新增字段。"""
    try:
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter
        from cryptotrader.agents.skills.schema import CaseRecord

        content = path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content, path=path)

        raw_ts = fm.get("timestamp")
        try:
            ts = datetime.fromisoformat(str(raw_ts)) if raw_ts else datetime.now(UTC)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            ts = datetime.now(UTC)

        # Parse agent_analyses from body sections
        agent_analyses = _extract_agent_analyses(body)

        return CaseRecord(
            cycle_id=fm.get("cycle_id", path.stem),
            timestamp=ts,
            pair=fm.get("pair", ""),
            verdict_action=fm.get("verdict_action", "hold"),
            final_pnl=fm.get("final_pnl"),
            agent_analyses=agent_analyses,
            applied_patterns=fm.get("applied_patterns", []),
            risk_gate_passed=fm.get("risk_gate_passed", True),
            file_path=path,
            trade_execution=fm.get("trade_execution"),
            causal_chain=fm.get("causal_chain"),
            ive_classification=fm.get("ive_classification"),
        )
    except Exception:
        logger.warning("_load_case_from_path failed for %s", path, exc_info=True)
        return None


def _extract_agent_analyses(body: str) -> dict[str, str]:
    """从 case markdown body 提取 agent analyses（### AgentId 段落）。"""
    result: dict[str, str] = {}
    current_agent: str | None = None
    current_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("### ") and not line.startswith("#### "):
            if current_agent is not None:
                result[current_agent.lower()] = "\n".join(current_lines).strip()
            current_agent = line[4:].strip().lower()
            current_lines = []
        elif current_agent is not None and not line.startswith("## "):
            current_lines.append(line)
        elif line.startswith("## ") and current_agent is not None:
            result[current_agent.lower()] = "\n".join(current_lines).strip()
            current_agent = None
            current_lines = []
    if current_agent is not None:
        result[current_agent.lower()] = "\n".join(current_lines).strip()
    return result


def _replace_section(body: str, section_title: str, new_section: str) -> str:
    """替换 body 中 ## <section_title> 段落为 new_section。"""
    pattern = re.compile(
        rf"^## {re.escape(section_title)}\n.*?(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    replaced, count = pattern.subn(new_section, body)
    if count == 0:
        return body + "\n" + new_section
    return replaced


def write_ive_classification(case_path: Path, fc: FailureClassification) -> None:
    """将 IVE Classification 写回 case 文件的 ## IVE Classification 段。"""
    try:
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter, render_frontmatter
        from cryptotrader.agents.skills._io import atomic_write

        content = case_path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content, path=case_path)
        fm["ive_classification"] = {
            "failure_type": fc.failure_type,
            "confidence": fc.confidence,
            "reasoning": fc.reasoning,
        }

        # 替换或追加 ## IVE Classification 段
        ive_section = (
            "\n## IVE Classification\n\n"
            f"- **Failure Type**: {fc.failure_type}\n"
            f"- **Confidence**: {fc.confidence:.2f}\n"
            f"- **Reasoning**: {fc.reasoning}\n"
        )
        if "## IVE Classification" in body:
            body = _replace_section(body, "IVE Classification", ive_section)
        else:
            body = body.rstrip() + "\n" + ive_section

        new_content = render_frontmatter(fm) + body
        atomic_write(case_path, new_content)
    except Exception:
        logger.warning("write_ive_classification failed for %s", case_path, exc_info=True)
