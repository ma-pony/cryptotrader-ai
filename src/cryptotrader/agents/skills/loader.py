"""SKILL.md 加载器 — parse + 进程内 LRU 缓存（mtime 失效）。

FR-019a: 扫描 agent_skills/*/SKILL.md，按 frontmatter scope 过滤；
          进程内 LRU 缓存（最大 32 条），每次访问对比磁盘 mtime，不一致则重新加载。
FR-004b: 不硬编码 skill name → agent 映射，通过 frontmatter scope 动态发现。
FR-024: 加载失败 → logger.warning + 跳过，不阻塞 cycle。
"""

from __future__ import annotations

import logging
from pathlib import Path

from cryptotrader.agents.skills._constants import DEFAULT_AGENT_SKILLS_DIR
from cryptotrader.agents.skills._frontmatter import (
    CorruptFrontmatterError,
    parse_frontmatter,
    validate_skill_frontmatter,
)
from cryptotrader.agents.skills.schema import Skill

logger = logging.getLogger(__name__)

# ── 进程内 LRU 缓存（最大 32 个 SKILL.md 解析结果）──
# key: str(path) → (mtime: float, Skill)
_skill_cache: dict[str, tuple[float, Skill]] = {}
_CACHE_MAX = 32


def _clear_cache() -> None:
    """清空 LRU 缓存（主要用于测试）。"""
    _skill_cache.clear()


def parse_skill_md(path: Path) -> Skill | None:
    """解析单个 SKILL.md 文件，返回 Skill 对象。

    失败时 logger.warning + 返回 None（FR-024）。
    使用进程内 mtime 缓存（FR-019a）。
    """
    try:
        current_mtime = path.stat().st_mtime
    except OSError:
        logger.warning("Skill file not found: %s", path)
        return None

    cache_key = str(path)
    cached = _skill_cache.get(cache_key)
    if cached is not None:
        cached_mtime, cached_skill = cached
        if cached_mtime == current_mtime:
            return cached_skill

    try:
        content = path.read_text(encoding="utf-8")
        data, body = parse_frontmatter(content, path=path)
        validate_skill_frontmatter(data, path=path)

        skill = Skill(
            name=str(data["name"]),
            description=str(data["description"]),
            scope=str(data["scope"]),
            body=body,
            file_path=path,
            manually_edited=bool(data.get("manually_edited", False)),
            version=str(data.get("version", "1.0")),
            mtime=current_mtime,
        )

        # 写入缓存（LRU: 超出上限时清除最旧条目）
        if len(_skill_cache) >= _CACHE_MAX:
            oldest_key = next(iter(_skill_cache))
            del _skill_cache[oldest_key]
        _skill_cache[cache_key] = (current_mtime, skill)
        return skill

    except CorruptFrontmatterError as exc:
        logger.warning("Skill frontmatter corrupt: %s — %s", path, exc)
        return None
    except Exception as exc:
        logger.warning("Failed to load skill %s: %s", path, exc)
        return None


def discover_skills_for_agent(
    agent_id: str,
    skill_dir: Path | None = None,
) -> list[Skill]:
    """动态发现并加载与 agent_id 匹配的所有 skills。

    FR-004b: 通过扫描 agent_skills/*/SKILL.md frontmatter scope 字段，
             不硬编码 skill→agent 映射。
    FR-019: 匹配条件：scope == "shared" 或 scope == "agent:<agent_id>"。
    FR-019a: mtime 失效缓存；新增/删除/修改 skill 文件后下个 cycle 自动反映。
    FR-024: 单条 SKILL.md 加载失败跳过 + warning，不阻塞 cycle。
    """
    base = skill_dir or DEFAULT_AGENT_SKILLS_DIR
    if not base.exists():
        logger.warning("agent_skills directory not found: %s", base)
        return []

    result: list[Skill] = []
    for skill_dir_entry in sorted(base.iterdir()):
        if not skill_dir_entry.is_dir() or skill_dir_entry.name.startswith("."):
            continue
        skill_md = skill_dir_entry / "SKILL.md"
        if not skill_md.exists():
            continue
        skill = parse_skill_md(skill_md)
        if skill is None:
            continue  # parse_skill_md already warned

        # 按 scope 过滤
        if skill.is_shared or (skill.scope == f"agent:{agent_id}"):
            result.append(skill)

    return result
