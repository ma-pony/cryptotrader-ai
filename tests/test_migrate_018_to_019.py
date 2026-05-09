"""spec 019 — 迁移脚本单测（SC-W3：>= 8 用例 PASS）。

tests/test_migrate_018_to_019.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_OLD = REPO_ROOT / "tests" / "fixtures" / "skills_old_format"


# ── helpers ───────────────────────────────────────────────────────────────────


def _write_skill(skills_dir: Path, name: str, scope: str = "shared", extra_fields: dict | None = None) -> Path:
    """在 skills_dir/<name>/SKILL.md 写入最简 SKILL.md。"""
    path = skills_dir / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    fm: dict = {
        "name": name,
        "description": f"Test skill {name} with at least 30 chars.",
        "scope": scope,
        "version": "1.0",
        "manually_edited": False,
    }
    if extra_fields:
        fm.update(extra_fields)
    yaml_str = yaml.dump(fm, allow_unicode=True, default_flow_style=False, sort_keys=False)
    content = f"---\n{yaml_str}---\n\n# Skill Body\n"
    path.write_text(content, encoding="utf-8")
    return path


def _read_frontmatter(path: Path) -> dict:
    """读取 SKILL.md 的 frontmatter dict。"""
    content = path.read_text(encoding="utf-8")
    second = content.find("\n---", 3)
    assert second != -1, "No closing --- found"
    yaml_str = content[3:second].strip()
    return yaml.safe_load(yaml_str)


# ── tests ─────────────────────────────────────────────────────────────────────


class TestMigrateKnownSkills:
    """(a) 5 已知 skill 用 mapping 写入预期值。"""

    def test_tech_analysis_gets_mapping_values(self, tmp_path):
        """tech-analysis 应写入 mapping 中的 importance=0.7 / confidence=0.7。"""
        from scripts.migrate_018_to_019 import SKILL_MIGRATION_DEFAULTS, run_migration

        _write_skill(tmp_path, "tech-analysis", scope="agent:tech")
        stats = run_migration(tmp_path, dry_run=False)
        assert stats["updated"] == 1
        fm = _read_frontmatter(tmp_path / "tech-analysis" / "SKILL.md")
        assert fm["importance"] == SKILL_MIGRATION_DEFAULTS["tech-analysis"]["importance"]
        assert fm["confidence"] == SKILL_MIGRATION_DEFAULTS["tech-analysis"]["confidence"]
        assert fm["triggers_keywords"] == SKILL_MIGRATION_DEFAULTS["tech-analysis"]["triggers_keywords"]
        assert fm["regime_tags"] == []

    def test_trading_knowledge_gets_08_importance(self, tmp_path):
        """trading-knowledge（shared）应写入 importance=0.8 / confidence=0.8。"""
        from scripts.migrate_018_to_019 import run_migration

        _write_skill(tmp_path, "trading-knowledge", scope="shared")
        run_migration(tmp_path, dry_run=False)
        fm = _read_frontmatter(tmp_path / "trading-knowledge" / "SKILL.md")
        assert fm["importance"] == 0.8
        assert fm["confidence"] == 0.8

    def test_chain_analysis_triggers_keywords(self, tmp_path):
        """chain-analysis 应含 FR-W3 列出的 triggers_keywords。"""
        from scripts.migrate_018_to_019 import SKILL_MIGRATION_DEFAULTS, run_migration

        _write_skill(tmp_path, "chain-analysis", scope="agent:chain")
        run_migration(tmp_path, dry_run=False)
        fm = _read_frontmatter(tmp_path / "chain-analysis" / "SKILL.md")
        expected_kws = SKILL_MIGRATION_DEFAULTS["chain-analysis"]["triggers_keywords"]
        assert fm["triggers_keywords"] == expected_kws


class TestMigrateUnknownSkill:
    """(b) 未知 skill 用默认值。"""

    def test_unknown_skill_gets_defaults(self, tmp_path):
        """不在 mapping 中的 skill 应写入默认空字段。"""
        from scripts.migrate_018_to_019 import run_migration

        _write_skill(tmp_path, "unknown-custom-skill", scope="shared")
        stats = run_migration(tmp_path, dry_run=False)
        assert stats["updated"] == 1
        fm = _read_frontmatter(tmp_path / "unknown-custom-skill" / "SKILL.md")
        assert fm["regime_tags"] == []
        assert fm["triggers_keywords"] == []
        assert fm["importance"] == 0.5
        assert fm["confidence"] == 0.5
        assert fm["access_count"] == 0
        assert "last_accessed_at" in fm


class TestIdempotency:
    """(c) 幂等性（重跑 2 次结果一致）。"""

    def test_idempotent_run(self, tmp_path):
        """重复跑 2 次，文件内容一致；第 2 次 updated=0。"""
        from scripts.migrate_018_to_019 import run_migration

        _write_skill(tmp_path, "tech-analysis", scope="agent:tech")
        run_migration(tmp_path, dry_run=False)
        content_after_first = (tmp_path / "tech-analysis" / "SKILL.md").read_text()

        stats2 = run_migration(tmp_path, dry_run=False)
        content_after_second = (tmp_path / "tech-analysis" / "SKILL.md").read_text()

        assert content_after_first == content_after_second, "幂等：2 次运行结果应一致"
        assert stats2["updated"] == 0, "第 2 次运行不应有更新"
        assert stats2["skipped"] >= 1


class TestDryRun:
    """(d) --dry-run 不修改文件。"""

    def test_dry_run_does_not_modify_file(self, tmp_path):
        """--dry-run 模式不写入文件。"""
        from scripts.migrate_018_to_019 import run_migration

        _write_skill(tmp_path, "tech-analysis", scope="agent:tech")
        original = (tmp_path / "tech-analysis" / "SKILL.md").read_text()

        stats = run_migration(tmp_path, dry_run=True)
        after = (tmp_path / "tech-analysis" / "SKILL.md").read_text()

        assert original == after, "dry-run 不应修改文件"
        assert stats["updated"] == 1, "dry-run 应仍然计数 updated"


class TestCorruptFrontmatter:
    """(e) 损坏 frontmatter 跳过 + warning log。"""

    def test_corrupt_frontmatter_skipped(self, tmp_path, caplog):
        """frontmatter 损坏的文件应跳过不崩溃。"""
        import logging

        from scripts.migrate_018_to_019 import run_migration

        bad_path = tmp_path / "bad-skill" / "SKILL.md"
        bad_path.parent.mkdir(parents=True)
        bad_path.write_text("no frontmatter here at all", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            stats = run_migration(tmp_path, dry_run=False)

        assert stats["updated"] == 0
        assert stats["failed"] >= 1 or stats["skipped"] >= 1


class TestBackupMessage:
    """(f) 备份提示输出。"""

    def test_backup_message_printed(self, tmp_path, capsys):
        """运行脚本时应输出备份建议。"""
        from scripts.migrate_018_to_019 import run_migration

        run_migration(tmp_path, dry_run=False)
        captured = capsys.readouterr()
        assert "备份" in captured.out or "backup" in captured.out.lower() or "cp -r" in captured.out


class TestPreserveExistingFields:
    """(g) 已存在的字段保留人工编辑。"""

    def test_existing_importance_not_overwritten(self, tmp_path):
        """已存在的 importance=0.9 不应被 mapping 覆盖。"""
        from scripts.migrate_018_to_019 import run_migration

        _write_skill(tmp_path, "tech-analysis", scope="agent:tech", extra_fields={"importance": 0.9})
        run_migration(tmp_path, dry_run=False)
        fm = _read_frontmatter(tmp_path / "tech-analysis" / "SKILL.md")
        assert fm["importance"] == 0.9, "人工编辑的 importance 不应被覆盖"
        # 但其他新字段应被添加
        assert "confidence" in fm
        assert "triggers_keywords" in fm

    def test_partial_skill_preserves_importance(self, tmp_path):
        """partial-skill（含 importance=0.9）迁移后 importance 仍为 0.9。"""
        from scripts.migrate_018_to_019 import run_migration

        shutil.copytree(FIXTURE_OLD / "partial-skill", tmp_path / "partial-skill")
        run_migration(tmp_path, dry_run=False)
        fm = _read_frontmatter(tmp_path / "partial-skill" / "SKILL.md")
        assert fm["importance"] == 0.9, "保留人工编辑的 importance=0.9"
        # 其他字段应被添加
        assert "regime_tags" in fm
        assert "confidence" in fm


class TestVersionUnchanged:
    """(h) version 字段不变。"""

    def test_version_field_not_changed(self, tmp_path):
        """迁移后 version 字段应保持原值 '1.0'。"""
        from scripts.migrate_018_to_019 import run_migration

        _write_skill(tmp_path, "tech-analysis", scope="agent:tech")
        run_migration(tmp_path, dry_run=False)
        fm = _read_frontmatter(tmp_path / "tech-analysis" / "SKILL.md")
        assert str(fm["version"]) == "1.0", "version 字段应保持不变"
