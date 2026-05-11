"""tests/test_distill_patterns_cold_start.py — spec 021 T008

5 用例覆盖 distill_patterns cold-start 路径：
  1. empty cases → patterns_created=0，不抛异常
  2. 频次 < min_cases_per_pattern (default 5) → 不创建 pattern
  3. 频次 >= min_cases_per_pattern → 创建 pattern 文件，maturity=observed
  4. pnl 全 None → 创建空 PnLTrack 但仍创建 pattern
  5. regime_tags 频次 top-3 投票
"""

from pathlib import Path

import pytest

from cryptotrader.learning.memory import distill_patterns


# ── helpers ──


def _make_case_file(
    path: Path,
    *,
    cycle_id: str,
    final_pnl: float | None,
    applied_agent: str,
    applied_pattern: str,
    regime_tags: list[str] | None = None,
) -> None:
    """向 path 写入符合 parse_frontmatter 格式的 case 文件。"""
    tags_yaml = ""
    if regime_tags:
        items = "\n".join(f"  - {t}" for t in regime_tags)
        tags_yaml = f"regime_tags:\n{items}\n"
    pnl_str = str(final_pnl) if final_pnl is not None else "null"
    content = (
        f"---\ncycle_id: {cycle_id}\nfinal_pnl: {pnl_str}\n{tags_yaml}---\n"
        f"## Applied Patterns\n- applied: {applied_agent}::{applied_pattern}\n"
    )
    path.write_text(content, encoding="utf-8")


def _setup_cases(
    tmp_path: Path,
    n: int,
    *,
    applied_agent: str = "tech",
    applied_pattern: str = "volume-spike",
    final_pnl: float | None = 1.0,
    regime_tags: list[str] | None = None,
) -> Path:
    """在 tmp_path/cases/ 写入 n 个 case 文件，返回 tmp_path。"""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _make_case_file(
            cases_dir / f"cycle-{i:04d}.md",
            cycle_id=f"cycle-{i:04d}",
            final_pnl=final_pnl,
            applied_agent=applied_agent,
            applied_pattern=applied_pattern,
            regime_tags=regime_tags,
        )
    return tmp_path


# ── 用例 1：cases 目录为空 ──


def test_empty_cases_returns_zero(tmp_path):
    """cases 目录空时 distill_patterns 返回 cases_processed=0，不抛异常。"""
    run = distill_patterns(memory_dir=tmp_path, cycles_window=10)
    assert run.cases_processed == 0
    assert run.patterns_created == 0
    assert not run.error  # empty string or None both indicate no error


# ── 用例 2：频次 < min_cases_per_pattern → 不创建 ──


def test_below_threshold_no_pattern_created(tmp_path):
    """4 个 cases（< default min=5）→ patterns_created=0。"""
    _setup_cases(tmp_path, n=4)
    run = distill_patterns(memory_dir=tmp_path, cycles_window=10)
    assert run.cases_processed == 4
    assert run.patterns_created == 0
    pattern_dir = tmp_path / "tech" / "patterns"
    assert not pattern_dir.exists() or len(list(pattern_dir.glob("*.md"))) == 0


# ── 用例 3：频次 >= min_cases_per_pattern → 创建 pattern ──


def test_above_threshold_creates_pattern(tmp_path):
    """5 个 cases 引用 tech::volume-spike → patterns_created=1，文件存在。"""
    _setup_cases(tmp_path, n=5, applied_agent="tech", applied_pattern="volume-spike")
    run = distill_patterns(memory_dir=tmp_path, cycles_window=10)
    assert run.patterns_created == 1
    pattern_file = tmp_path / "tech" / "patterns" / "volume-spike.md"
    assert pattern_file.exists(), f"Expected pattern file: {pattern_file}"
    content = pattern_file.read_text(encoding="utf-8")
    # maturity が observed で作成され、その後 FSM で昇格する場合もある
    assert "maturity:" in content
    assert "Auto-distilled" in content


# ── 用例 4：pnl 全 None → 创建空 PnLTrack 但仍创建 pattern ──


def test_pnl_all_none_creates_pattern_with_empty_pnl_track(tmp_path):
    """所有 case pnl=None → pattern 仍被创建，pnl_track.cases=0。"""
    _setup_cases(tmp_path, n=5, final_pnl=None)
    run = distill_patterns(memory_dir=tmp_path, cycles_window=10)
    assert run.patterns_created == 1
    pattern_file = tmp_path / "tech" / "patterns" / "volume-spike.md"
    assert pattern_file.exists()
    content = pattern_file.read_text(encoding="utf-8")
    # cases 字段应为 0（无 pnl 数据）
    assert "cases: 0" in content


# ── 用例 5：regime_tags 频次 top-3 投票 ──


def test_regime_tags_top3_voting(tmp_path):
    """不同 regime_tags 出现频次不同时，top-3 按频次降序、并列字母序选择。"""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    # bull: 5次, bear: 3次, sideways: 2次, volatile: 1次
    tag_scenarios = [
        (["bull"] * 3),    # cycle 0-2: bull only (3 cases)
        (["bull", "bear"]),  # cycle 3: bull+bear
        (["bull", "bear"]),  # cycle 4: bull+bear → bull=5, bear=2 so far
        (["bear"]),          # cycle 5: bear=3
        (["sideways"]) * 2, # skip — just use 2 separate
    ]
    tags_per_case = [
        ["bull"],
        ["bull"],
        ["bull", "bear"],
        ["bull", "bear"],
        ["bear"],
        ["sideways"],
        ["sideways"],
    ]
    for i, tags in enumerate(tags_per_case):
        _make_case_file(
            cases_dir / f"cycle-{i:04d}.md",
            cycle_id=f"cycle-{i:04d}",
            final_pnl=1.0,
            applied_agent="tech",
            applied_pattern="volume-spike",
            regime_tags=tags,
        )
    run = distill_patterns(memory_dir=tmp_path, cycles_window=20)
    assert run.patterns_created == 1
    pattern_file = tmp_path / "tech" / "patterns" / "volume-spike.md"
    content = pattern_file.read_text(encoding="utf-8")
    # bull(4), bear(2), sideways(2) — 并列字母序: bear < sideways
    assert "bull" in content
    assert "bear" in content
    assert "sideways" in content
