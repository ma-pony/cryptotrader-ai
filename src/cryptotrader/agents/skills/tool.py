"""load_skill tool — 普通 Python 函数 + LangChain BaseTool 双接口。

FR-022: 双接口（函数 + LangChain tool），同实现。
FR-023: 仅需 1 参数（skill name），按 directory 解析。
FR-025: 同一 cycle 同一 trace_id 调用 > 10 次返回 rate_limit_per_cycle。
FR-025a: 每次调用通过 metrics_collector 增加 counter。
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from cryptotrader.agents.skills._constants import DEFAULT_AGENT_SKILLS_DIR

logger = logging.getLogger(__name__)

# ── rate-limit 计数器（per trace_id）──
# thread-safe dict: trace_id → call_count
_call_counts: dict[str, int] = defaultdict(int)
_call_counts_lock = threading.Lock()
_RATE_LIMIT_PER_CYCLE = 10


def _reset_call_counter(trace_id: str) -> None:
    """重置指定 trace_id 的调用计数（主要用于测试）。"""
    with _call_counts_lock:
        _call_counts[trace_id] = 0


def _increment_and_check(trace_id: str) -> bool:
    """递增调用计数，返回 True 表示超出限制。"""
    with _call_counts_lock:
        _call_counts[trace_id] += 1
        return _call_counts[trace_id] > _RATE_LIMIT_PER_CYCLE


# ── 核心函数 ──


def load_skill(
    name: str,
    skill_dir: Path | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """加载指定 skill 的 body 内容。

    FR-022: 普通 Python 函数接口。
    FR-023: 按 agent_skills/<name>/SKILL.md directory 解析。
    FR-025: rate-limit（> 10 次/cycle/trace_id）。

    Returns:
        成功: {"name": str, "body": str, "scope": str}
        失败: {"error": "skill_not_found" | "corrupt_file" | "rate_limit_per_cycle", "name": str}
    """
    # FR-025a: metrics
    _record_metric(name, "attempt")

    # FR-025: rate-limit check
    effective_trace_id = trace_id or "default"
    if _increment_and_check(effective_trace_id):
        logger.warning("load_skill rate limit exceeded for trace_id=%s (skill=%s)", effective_trace_id, name)
        _record_metric(name, "rate_limit")
        return {"error": "rate_limit_per_cycle", "name": name}

    base = skill_dir or DEFAULT_AGENT_SKILLS_DIR
    skill_md = base / name / "SKILL.md"

    if not skill_md.exists():
        logger.warning("load_skill: skill not found: %s", name)
        _record_metric(name, "skill_not_found")
        return {"error": "skill_not_found", "name": name}

    from cryptotrader.agents.skills._frontmatter import (
        CorruptFrontmatterError,
        parse_frontmatter,
        validate_skill_frontmatter,
    )

    try:
        content = skill_md.read_text(encoding="utf-8")
        data, body = parse_frontmatter(content, path=skill_md)
        validate_skill_frontmatter(data, path=skill_md)
        _record_metric(name, "ok")
        return {
            "name": str(data["name"]),
            "body": body,
            "scope": str(data.get("scope", "")),
            "description": str(data.get("description", "")),
        }
    except CorruptFrontmatterError as exc:
        logger.warning("load_skill: corrupt frontmatter for '%s': %s", name, exc)
        _record_metric(name, "corrupt_file")
        return {"error": "corrupt_file", "name": name}
    except Exception as exc:
        logger.warning("load_skill: unexpected error for '%s': %s", name, exc)
        _record_metric(name, "corrupt_file")
        return {"error": "corrupt_file", "name": name}


def _record_metric(name: str, result: str) -> None:
    """FR-025a: 通过 metrics_collector 记录 load_skill 调用计数。"""
    try:
        from cryptotrader.metrics import get_metrics_collector

        mc = get_metrics_collector()
        if hasattr(mc, "inc_counter"):
            mc.inc_counter("load_skill_calls", labels={"name": name, "result": result})
    except Exception:
        pass  # metrics 不可用时不阻塞


# ── LangChain BaseTool 接口 ──


def _make_load_skill_tool(skill_dir: Path | None = None):
    """创建 LangChain BaseTool 实例（FR-021 + FR-022）。"""
    try:
        from langchain_core.tools import tool

        @tool
        def load_skill_tool(name: str) -> str:
            """Load the body content of a named skill.

            Args:
                name: The skill directory name (e.g. 'tech-analysis', 'trading-knowledge')

            Returns:
                The skill body as a string, or an error message.
            """
            result = load_skill(name, skill_dir=skill_dir)
            if "error" in result:
                return f"Error: {result['error']} (skill: {name})"
            return result.get("body", "")

        return load_skill_tool
    except ImportError:
        logger.warning("langchain_core not available; load_skill_tool not created")
        return None


# Module-level tool instance (default skill_dir)
load_skill_tool = _make_load_skill_tool()
