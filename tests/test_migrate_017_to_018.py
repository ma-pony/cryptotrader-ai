"""spec 018 迁移脚本单测 — tests/test_migrate_017_to_018.py

SC-Z3：≥ 8 用例 PASS。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

# 确保能找到 scripts 模块
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from migrate_017_to_018 import (  # noqa: E402
    _case_needs_migration,
    _pattern_needs_migration,
    _split_frontmatter,
    migrate_case,
    migrate_pattern,
    run_migration,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "memory_old_format"

# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _copy_fixture_to_tmp(tmp_path: Path) -> Path:
    """把 fixture 目录复制到 tmp_path，返回新路径（隔离测试）。"""
    dest = tmp_path / "memory_old_format"
    shutil.copytree(FIXTURE_DIR, dest)
    return dest


def _read_frontmatter(path: Path) -> dict:
    content = path.read_text(encoding="utf-8")
    result = _split_frontmatter(content)
    assert result is not None, f"frontmatter 解析失败: {path}"
    return result[0]


def _read_body(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    result = _split_frontmatter(content)
    assert result is not None
    return result[1]


# ── 测试：旧 case 添加 3 个新段 ───────────────────────────────────────────────


def test_case_migration_adds_three_sections(tmp_path: Path):
    """T008(a)：旧 case 迁移后含 Trade Execution / Causal Chain / IVE Classification。"""
    root = _copy_fixture_to_tmp(tmp_path)
    case_path = root / "cases" / "old_case_001.md"

    migrated = migrate_case(case_path, dry_run=False)
    assert migrated is True

    body = _read_body(case_path)
    assert "## Trade Execution" in body
    assert "## Causal Chain" in body
    assert "## IVE Classification" in body


# ── 测试：旧 pattern 添加 5 个新字段 ──────────────────────────────────────────


def test_pattern_migration_adds_five_fields(tmp_path: Path):
    """T008(b)：旧 pattern 迁移后含 5 个新字段。"""
    root = _copy_fixture_to_tmp(tmp_path)
    pattern_path = root / "tech" / "patterns" / "breakout_continuation.md"

    migrated = migrate_pattern(pattern_path, dry_run=False)
    assert migrated is True

    fm = _read_frontmatter(pattern_path)
    assert "importance" in fm
    assert "access_count" in fm
    assert "last_accessed_at" in fm
    assert "last_modified_at" in fm
    assert "fundamental_failure_streak" in fm

    assert fm["importance"] == 0.5
    assert fm["access_count"] == 0
    assert fm["fundamental_failure_streak"] == 0


# ── 测试：幂等性（重跑 2 次结果一致）────────────────────────────────────────


def test_migration_idempotent(tmp_path: Path):
    """T008(c)：重跑迁移脚本不损坏数据（幂等性）。"""
    root = _copy_fixture_to_tmp(tmp_path)

    # 第一次迁移
    stats1 = run_migration(root, dry_run=False)
    assert stats1["cases_migrated"] > 0
    assert stats1["patterns_migrated"] > 0

    # 读取迁移后内容
    case_path = root / "cases" / "old_case_001.md"
    content_after_1 = case_path.read_text(encoding="utf-8")

    # 第二次迁移
    stats2 = run_migration(root, dry_run=False)
    assert stats2["cases_migrated"] == 0  # 全部跳过
    assert stats2["cases_skipped"] >= 1
    assert stats2["patterns_migrated"] == 0
    assert stats2["patterns_skipped"] >= 1

    # 内容不变
    content_after_2 = case_path.read_text(encoding="utf-8")
    assert content_after_1 == content_after_2


# ── 测试：--dry-run 不修改文件 ────────────────────────────────────────────────


def test_dry_run_does_not_modify_files(tmp_path: Path):
    """T008(d)：dry_run=True 时不修改任何文件。"""
    root = _copy_fixture_to_tmp(tmp_path)

    case_path = root / "cases" / "old_case_001.md"
    original_content = case_path.read_text(encoding="utf-8")

    stats = run_migration(root, dry_run=True)
    assert stats["cases_migrated"] > 0  # dry-run 返回"将要迁移"数量

    # 文件未修改
    assert case_path.read_text(encoding="utf-8") == original_content


# ── 测试：损坏 frontmatter 跳过 + warning log ─────────────────────────────────


def test_corrupted_frontmatter_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """T008(e)：frontmatter 损坏时跳过该文件并 warning。"""
    bad_case = tmp_path / "bad_case.md"
    bad_case.write_text("not a valid frontmatter\n\nbody here", encoding="utf-8")

    with caplog.at_level("WARNING"):
        result = migrate_case(bad_case, dry_run=False)

    assert result is False
    assert any(
        "frontmatter 损坏" in r.message or "无法找到" in r.message or "损坏" in r.message for r in caplog.records
    )


# ── 测试：备份提示输出 ────────────────────────────────────────────────────────


def test_backup_suggestion_printed(tmp_path: Path, capsys: pytest.CaptureFixture):
    """T008(f)：迁移启动时打印备份建议。"""
    # 直接调用 main 会因路径参数复杂；改为直接检查 main 输出
    # 使用 subprocess 测试 --dry-run 模式
    import subprocess

    root = _copy_fixture_to_tmp(tmp_path)
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS_DIR / "migrate_017_to_018.py"), "--dry-run", "--memory-root", str(root)],
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    assert "backup" in combined.lower() or "备份" in combined


# ── 测试：已迁移 case 跳过 ────────────────────────────────────────────────────


def test_already_migrated_case_skipped(tmp_path: Path):
    """T008(g)：已含 Trade Execution 段的 case 不重复迁移。"""
    case_path = tmp_path / "already_migrated.md"
    case_path.write_text(
        "---\ncycle_id: abc\ntimestamp: '2026-01-01T00:00:00+00:00'\npair: BTC/USDT\n---\n"
        "## Agent Analyses\nContent\n\n## Trade Execution\n- entry_price: 50000\n",
        encoding="utf-8",
    )

    result = migrate_case(case_path, dry_run=False)
    assert result is False  # 跳过，不修改


# ── 测试：已迁移 pattern 跳过 ────────────────────────────────────────────────


def test_already_migrated_pattern_skipped(tmp_path: Path):
    """T008(h)：已含 importance 字段的 pattern 不重复迁移。"""
    pattern_path = tmp_path / "already_migrated_pattern.md"
    pattern_path.write_text(
        "---\nname: test\nagent: tech\ndescription: desc\nmaturity: active\n"
        "importance: 0.8\naccess_count: 5\nlast_accessed_at: '2026-01-01T00:00:00+00:00'\n"
        "last_modified_at: '2026-01-01T00:00:00+00:00'\nfundamental_failure_streak: 0\n---\n"
        "## Rule\nBody content",
        encoding="utf-8",
    )

    result = migrate_pattern(pattern_path, dry_run=False)
    assert result is False  # 跳过


# ── 测试：完整 run_migration 统计 ────────────────────────────────────────────


def test_run_migration_stats(tmp_path: Path):
    """整体迁移统计正确（1 case + 2 patterns）。"""
    root = _copy_fixture_to_tmp(tmp_path)
    stats = run_migration(root, dry_run=False)

    assert stats["cases_migrated"] == 1
    assert stats["cases_failed"] == 0
    assert stats["patterns_migrated"] == 2
    assert stats["patterns_failed"] == 0


# ── 测试：_case_needs_migration 逻辑 ─────────────────────────────────────────


def test_case_needs_migration_flag():
    """body 无新段时需要迁移，有时不需要。"""
    assert _case_needs_migration("## Agent Analyses\nContent") is True
    assert _case_needs_migration("## Trade Execution\n- entry_price: null") is False


# ── 测试：_pattern_needs_migration 逻辑 ──────────────────────────────────────


def test_pattern_needs_migration_flag():
    """fm 无 importance 时需要迁移，有时不需要。"""
    assert _pattern_needs_migration({"name": "x", "maturity": "active"}) is True
    assert _pattern_needs_migration({"name": "x", "importance": 0.5}) is False
