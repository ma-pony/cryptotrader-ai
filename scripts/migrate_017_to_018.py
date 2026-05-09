"""spec 018 迁移脚本：把现有 agent_memory/ 数据升级到 spec 018 schema。

用法：
  python scripts/migrate_017_to_018.py [--dry-run] [--memory-root PATH]

功能：
  (a) 扫 agent_memory/cases/*.md → 加 3 个新段（Trade Execution / Causal Chain / IVE Classification）
  (b) 扫 agent_memory/<agent>/patterns/*.md → 加 5 个新字段（importance / access_count /
      last_accessed_at / last_modified_at / fundamental_failure_streak）

幂等性：重复跑不损坏数据（已迁移的 case / pattern 跳过）。

FR-Z3 / FR-Z4 / FR-Z5 / FR-Z31 / FR-Z32 / FR-Z33
"""

from __future__ import annotations

import argparse
import datetime
import logging
import re
import sys
from pathlib import Path

import yaml

# ── 日志 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────────
KNOWN_AGENTS = ["tech", "chain", "news", "macro"]

NEW_CASE_SECTIONS = """\
## Trade Execution
- entry_price: null
- stop_loss: null
- take_profit: null
- actual_exit_price: null
- fill_status: null
- hit_sl: null
- hit_tp: null
- exit_reason: null

## Causal Chain
### Tool Calls (per agent)
_（待填充）_

### Verbal Reinforcement Input
_（待填充）_

### Debate Intermediate
_（待填充）_

## IVE Classification
- failure_type: null
- confidence: null
- reasoning: null
- diagnostic_answers: []
"""

NEW_PATTERN_FIELDS = {
    "importance": 0.5,
    "access_count": 0,
    "fundamental_failure_streak": 0,
    # last_accessed_at / last_modified_at 从 file mtime 填充（见下方逻辑）
}

# ── frontmatter 解析辅助 ───────────────────────────────────────────────────────


def _split_frontmatter(content: str) -> tuple[dict, str] | None:
    """解析 YAML frontmatter。返回 (fm_dict, body_text) 或 None（解析失败）。"""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        logger.warning("YAML 解析失败: %s", exc)
        return None
    if not isinstance(fm, dict):
        return None
    return fm, m.group(2)


def _render_frontmatter(fm: dict, body: str) -> str:
    """把 fm_dict + body 重新序列化为 markdown。"""
    fm_text = yaml.dump(fm, allow_unicode=True, default_flow_style=False, sort_keys=False).rstrip()
    return f"---\n{fm_text}\n---\n{body}"


# ── case 迁移 ─────────────────────────────────────────────────────────────────


def _case_needs_migration(body: str) -> bool:
    """检查 case body 是否缺少新段（幂等性：已有则跳过）。"""
    return "## Trade Execution" not in body


def migrate_case(path: Path, dry_run: bool) -> bool:
    """迁移单个 case 文件。返回是否实际执行了修改。"""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("无法读取 case 文件 %s: %s", path, exc)
        return False

    result = _split_frontmatter(content)
    if result is None:
        logger.warning("case 文件 frontmatter 损坏，跳过: %s", path)
        return False

    _fm, body = result

    if not _case_needs_migration(body):
        logger.debug("case 已迁移，跳过: %s", path.name)
        return False

    new_body = body.rstrip() + "\n\n" + NEW_CASE_SECTIONS
    new_content = _render_frontmatter(_fm, new_body)

    if dry_run:
        logger.info("[DRY-RUN] 将迁移 case: %s (+3 段)", path.name)
        return True

    path.write_text(new_content, encoding="utf-8")
    logger.info("已迁移 case: %s (+3 段)", path.name)
    return True


# ── pattern 迁移 ──────────────────────────────────────────────────────────────


def _pattern_needs_migration(fm: dict) -> bool:
    """检查 pattern frontmatter 是否缺少新字段（幂等性：已有则跳过）。"""
    return "importance" not in fm


def migrate_pattern(path: Path, dry_run: bool) -> bool:
    """迁移单个 pattern 文件。返回是否实际执行了修改。"""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("无法读取 pattern 文件 %s: %s", path, exc)
        return False

    result = _split_frontmatter(content)
    if result is None:
        logger.warning("pattern 文件 frontmatter 损坏，跳过: %s", path)
        return False

    fm, body = result

    if not _pattern_needs_migration(fm):
        logger.debug("pattern 已迁移，跳过: %s", path.name)
        return False

    # 用 file mtime 填充 last_accessed_at / last_modified_at
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = datetime.datetime.now(datetime.UTC).timestamp()

    mtime_iso = datetime.datetime.fromtimestamp(mtime, tz=datetime.UTC).isoformat()

    fm["importance"] = NEW_PATTERN_FIELDS["importance"]
    fm["access_count"] = NEW_PATTERN_FIELDS["access_count"]
    fm["last_accessed_at"] = mtime_iso
    fm["last_modified_at"] = mtime_iso
    fm["fundamental_failure_streak"] = NEW_PATTERN_FIELDS["fundamental_failure_streak"]

    new_content = _render_frontmatter(fm, body)

    if dry_run:
        logger.info(
            "[DRY-RUN] 将迁移 pattern: %s (+5 字段, last_modified_at=%s)",
            path.name,
            mtime_iso,
        )
        return True

    path.write_text(new_content, encoding="utf-8")
    logger.info("已迁移 pattern: %s (+5 字段)", path.name)
    return True


# ── 主流程 ────────────────────────────────────────────────────────────────────


def _migrate_cases(cases_dir: Path, dry_run: bool, stats: dict[str, int]) -> None:
    """迁移 cases 目录下所有 .md 文件。"""
    if not cases_dir.is_dir():
        logger.info("cases 目录不存在，跳过: %s", cases_dir)
        return
    for p in sorted(cases_dir.glob("*.md")):
        if p.name == ".gitkeep":
            continue
        try:
            if migrate_case(p, dry_run):
                stats["cases_migrated"] += 1
            else:
                stats["cases_skipped"] += 1
        except Exception as exc:
            logger.warning("迁移 case 失败 %s: %s", p.name, exc, exc_info=True)
            stats["cases_failed"] += 1


def _migrate_patterns_for_agent(patterns_dir: Path, dry_run: bool, stats: dict[str, int]) -> None:
    """迁移单个 agent patterns 目录。"""
    for p in sorted(patterns_dir.glob("*.md")):
        if p.name.startswith("."):
            continue
        try:
            if migrate_pattern(p, dry_run):
                stats["patterns_migrated"] += 1
            else:
                stats["patterns_skipped"] += 1
        except Exception as exc:
            logger.warning("迁移 pattern 失败 %s: %s", p.name, exc, exc_info=True)
            stats["patterns_failed"] += 1


def run_migration(memory_root: Path, dry_run: bool) -> dict[str, int]:
    """执行全量迁移，返回统计摘要。"""
    stats: dict[str, int] = {
        "cases_migrated": 0,
        "cases_skipped": 0,
        "cases_failed": 0,
        "patterns_migrated": 0,
        "patterns_skipped": 0,
        "patterns_failed": 0,
    }

    _migrate_cases(memory_root / "cases", dry_run, stats)

    for agent_id in KNOWN_AGENTS:
        patterns_dir = memory_root / agent_id / "patterns"
        if not patterns_dir.is_dir():
            logger.debug("patterns 目录不存在，跳过: %s", patterns_dir)
            continue
        _migrate_patterns_for_agent(patterns_dir, dry_run, stats)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="spec 018 迁移脚本：升级 agent_memory/ 到新 schema。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅输出预览，不实际修改文件。",
    )
    parser.add_argument(
        "--memory-root",
        type=Path,
        default=Path("agent_memory"),
        help="agent_memory/ 根目录路径（默认: agent_memory）。",
    )
    args = parser.parse_args()

    memory_root: Path = args.memory_root
    dry_run: bool = args.dry_run

    if not memory_root.is_dir():
        logger.error("memory_root 不存在或不是目录: %s", memory_root)
        sys.exit(1)

    # FR-Z33：启动时输出备份建议
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    print(f"\n建议先备份：cp -r {memory_root} {memory_root}.backup_{ts}\n")

    if dry_run:
        print("[DRY-RUN 模式] 不会修改任何文件，仅预览变更。\n")

    logger.info("开始迁移 memory_root=%s dry_run=%s", memory_root.resolve(), dry_run)

    stats = run_migration(memory_root, dry_run)

    # 输出摘要
    print("\n── 迁移完成 ────────────────────────────────")
    cm, cs, cf = stats["cases_migrated"], stats["cases_skipped"], stats["cases_failed"]
    pm, ps, pf = stats["patterns_migrated"], stats["patterns_skipped"], stats["patterns_failed"]
    print(f"  Cases   : 已迁移 {cm} / 已跳过 {cs} / 失败 {cf}")
    print(f"  Patterns: 已迁移 {pm} / 已跳过 {ps} / 失败 {pf}")
    if dry_run:
        print("  （DRY-RUN：以上均为预览，实际未修改）")
    print("────────────────────────────────────────────\n")

    # 如有失败，退出码 1 方便 CI 感知
    if stats["cases_failed"] > 0 or stats["patterns_failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
