"""spec 019 迁移脚本：把现有 agent_skills/ 数据升级到 spec 019 schema。

用法：
  python scripts/migrate_018_to_019.py [--dry-run] [--skills-root PATH]

功能：
  扫 agent_skills/*/SKILL.md -> 为每个 skill 加 6 个新字段：
    regime_tags / triggers_keywords / importance / access_count /
    last_accessed_at / confidence
  已知 5 个 skill 用 SKILL_MIGRATION_DEFAULTS 中的 mapping；
  未知 skill name -> 默认空字段。

幂等性：重复跑不损坏数据（已有字段不覆盖）。

FR-W3 / FR-W4 / FR-W5 / FR-W6 / FR-W30 / FR-W31 / FR-W32
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

# ── 日志 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── FR-W3: 5 个已知 skill 的硬编码 mapping（基于 brainstorm 阶段 LLM 分析）──
SKILL_MIGRATION_DEFAULTS: dict[str, dict] = {
    "chain-analysis": {
        "regime_tags": [],
        "triggers_keywords": [
            "funding rate",
            "exchange flow",
            "netflow",
            "whale",
            "open interest",
            "OI",
            "liquidation",
            "on-chain",
            "accumulation",
            "distribution",
            "blockchain",
        ],
        "importance": 0.7,
        "confidence": 0.7,
    },
    "macro-analysis": {
        "regime_tags": [],
        "triggers_keywords": [
            "fed",
            "dxy",
            "dollar index",
            "fear greed",
            "etf",
            "vix",
            "s&p",
            "macro",
            "rate cut",
            "rate hike",
            "cpi",
            "risk-on",
            "risk-off",
            "sentiment",
        ],
        "importance": 0.7,
        "confidence": 0.7,
    },
    "news-analysis": {
        "regime_tags": [],
        "triggers_keywords": [
            "news",
            "headline",
            "regulatory",
            "etf approval",
            "ban",
            "hack",
            "exploit",
            "social",
            "sentiment",
            "twitter",
            "catalyst",
        ],
        "importance": 0.6,
        "confidence": 0.6,
    },
    "tech-analysis": {
        "regime_tags": [],
        "triggers_keywords": [
            "rsi",
            "macd",
            "sma",
            "moving average",
            "bollinger",
            "atr",
            "chart",
            "trend",
            "momentum",
            "breakout",
            "support",
            "resistance",
            "indicator",
        ],
        "importance": 0.7,
        "confidence": 0.7,
    },
    "trading-knowledge": {  # shared scope
        "regime_tags": [],
        "triggers_keywords": [
            "funding",
            "regime",
            "spot",
            "perp",
            "perpetual",
            "basis",
            "confidence",
            "calibration",
            "attribution",
            "data sufficiency",
            "microstructure",
        ],
        "importance": 0.8,
        "confidence": 0.8,  # foundational
    },
}

# 默认空字段（未知 skill name 时使用）
_DEFAULT_FIELDS: dict[str, object] = {
    "regime_tags": [],
    "triggers_keywords": [],
    "importance": 0.5,
    "confidence": 0.5,
}

# 6 个新字段名
_NEW_FIELDS = frozenset(
    {"regime_tags", "triggers_keywords", "importance", "access_count", "last_accessed_at", "confidence"}
)


def _parse_frontmatter(content: str) -> tuple[dict, str] | None:
    """解析 YAML frontmatter，失败返回 None。"""
    if not content.startswith("---"):
        return None
    # 找第二个 ---
    second = content.find("\n---", 3)
    if second == -1:
        return None
    yaml_str = content[3:second].strip()
    body = content[second + 4 :].lstrip("\n")
    try:
        data = yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    return data, body


def _render_frontmatter(data: dict) -> str:
    """将 dict 渲染为 YAML frontmatter 块（含 --- 分隔符）。"""
    yaml_str = yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return f"---\n{yaml_str}---\n"


def _get_mtime_iso(path: Path) -> str:
    """获取文件 mtime，返回 ISO8601 字符串。"""
    try:
        mtime = path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime, tz=UTC)
        return dt.isoformat()
    except OSError:
        return datetime.now(UTC).isoformat()


def migrate_skill_file(skill_md: Path, dry_run: bool) -> tuple[bool, str]:
    """迁移单个 SKILL.md 文件。

    Returns:
        (changed: bool, message: str)
    """
    try:
        content = skill_md.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"无法读取 {skill_md}: {exc}"

    parsed = _parse_frontmatter(content)
    if parsed is None:
        return False, f"跳过 {skill_md}：frontmatter 解析失败"

    data, body = parsed

    # 获取 skill name
    skill_name = str(data.get("name", skill_md.parent.name))

    # 获取此 skill 的 mapping（已知用 mapping，未知用默认值）
    mapping = SKILL_MIGRATION_DEFAULTS.get(skill_name)

    # 检查是否需要更新（幂等：已有字段不覆盖）
    fields_added = []
    for field_name in _NEW_FIELDS:
        if field_name in data:
            continue  # 已存在，不覆盖（保留人工编辑）
        if mapping and field_name in mapping:
            data[field_name] = mapping[field_name]
        elif field_name == "access_count":
            data[field_name] = 0
        elif field_name == "last_accessed_at":
            data[field_name] = _get_mtime_iso(skill_md)
        else:
            data[field_name] = _DEFAULT_FIELDS.get(field_name)
        fields_added.append(field_name)

    if not fields_added:
        return False, f"跳过 {skill_md}：所有新字段已存在（幂等）"

    new_content = _render_frontmatter(data) + body
    if dry_run:
        return True, f"[DRY-RUN] 将为 {skill_md} 添加字段: {fields_added}"

    try:
        skill_md.write_text(new_content, encoding="utf-8")
        return True, f"已更新 {skill_md}，添加字段: {fields_added}"
    except OSError as exc:
        return False, f"写入失败 {skill_md}: {exc}"


def run_migration(skills_root: Path, dry_run: bool) -> dict[str, int]:
    """扫 skills_root/*/SKILL.md，执行迁移。

    Returns:
        {"updated": int, "skipped": int, "failed": int}
    """
    # FR-W6: 启动期 print 备份建议
    print("=" * 60)
    print("spec 019 SKILL.md 迁移脚本")
    print("=" * 60)
    print()
    print("⚠  重要：请在运行前备份 agent_skills/ 目录！")
    print("   例如：cp -r agent_skills/ agent_skills.bak/")
    print()
    if dry_run:
        print("运行模式：DRY-RUN（不会修改任何文件）")
    else:
        print("运行模式：实际迁移")
    print()

    if not skills_root.exists():
        logger.warning("agent_skills 目录不存在: %s", skills_root)
        return {"updated": 0, "skipped": 0, "failed": 0}

    stats = {"updated": 0, "skipped": 0, "failed": 0}
    audit_trail: list[str] = []

    for skill_dir in sorted(skills_root.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            logger.warning("跳过 %s：SKILL.md 不存在", skill_dir.name)
            stats["skipped"] += 1
            audit_trail.append(f"SKIP {skill_dir.name}: no SKILL.md")
            continue

        changed, message = migrate_skill_file(skill_md, dry_run)
        logger.info("%s", message)
        audit_trail.append(message)

        if "无法读取" in message or "写入失败" in message or "解析失败" in message:
            stats["failed"] += 1
        elif changed:
            stats["updated"] += 1
        else:
            stats["skipped"] += 1

    # FR-W32: 输出迁移日志 + 失败行的 audit trail
    print()
    print("=" * 60)
    print("迁移完成")
    print(f"  更新: {stats['updated']} 个 skill")
    print(f"  跳过: {stats['skipped']} 个 skill（幂等或无 SKILL.md）")
    print(f"  失败: {stats['failed']} 个 skill")
    print()
    if stats["failed"] > 0:
        print("失败 audit trail：")
        for line in audit_trail:
            if "失败" in line or "无法" in line:
                print(f"  {line}")
    print("=" * 60)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="spec 019 SKILL.md 迁移脚本（加 6 新字段）")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式：显示将要修改的内容，不实际写入文件",
    )
    parser.add_argument(
        "--skills-root",
        type=Path,
        default=Path("agent_skills"),
        help="agent_skills 目录路径（默认：agent_skills/）",
    )
    args = parser.parse_args()

    stats = run_migration(args.skills_root, args.dry_run)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
