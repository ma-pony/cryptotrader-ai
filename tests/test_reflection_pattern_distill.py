"""Tests for US2: Reflection pattern 蒸馏（FR-008 / FR-010 / FR-011 / FR-012 / FR-013）。

TDD: 先 FAIL，实现 learning/memory.py 后 GREEN。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).parent.parent


def _make_tmp_memory(tmp_path: Path) -> Path:
    """创建临时 agent_memory 目录骨架。"""
    from cryptotrader.agents.skills._io import ensure_memory_dirs

    ensure_memory_dirs(tmp_path)
    return tmp_path


def _make_test_cases(mem_dir: Path, count: int = 10) -> None:
    """写入若干测试 case 文件。"""
    from cryptotrader.learning.memory import write_case

    for i in range(count):
        pnl = 100.0 if i % 2 == 0 else -50.0
        write_case(
            cycle_id=f"2026-05-{i + 1:02d}-cycle-{i:04x}",
            pair="BTC/USDT",
            agent_analyses={
                "tech": f"RSI analysis {i}",
                "chain": f"funding {i}",
                "news": f"news {i}",
                "macro": f"macro {i}",
            },
            verdict_action="long" if i % 2 == 0 else "short",
            verdict_reasoning=f"Reasoning {i}. applied: tech::rsi_oversold_bounce",
            applied_patterns=["tech::rsi_oversold_bounce"],
            risk_gate_passed=True,
            execution_status={"succeeded": True},
            final_pnl=pnl,
            memory_dir=mem_dir,
        )


class TestDistillPatterns:
    """测试 distill_patterns()：从 cases 蒸馏出 patterns（FR-008）。"""

    def test_distill_creates_pattern_file(self, tmp_path):
        """蒸馏后应在 agent_memory/<agent>/patterns/ 下创建 pattern 文件。"""
        from cryptotrader.learning.memory import distill_patterns, write_case

        mem_dir = _make_tmp_memory(tmp_path)
        # 写入足够的 cases
        for i in range(6):
            pnl = 80.0 if i < 4 else -30.0
            write_case(
                cycle_id=f"2026-05-{i + 1:02d}-cycle-{i:04x}",
                pair="BTC/USDT",
                agent_analyses={"tech": f"analysis {i}", "chain": "ok", "news": "ok", "macro": "ok"},
                verdict_action="long",
                verdict_reasoning="applied: tech::test_pattern",
                applied_patterns=["tech::test_pattern"],
                risk_gate_passed=True,
                execution_status=None,
                final_pnl=pnl,
                memory_dir=mem_dir,
            )
        # 手动创建一个 pattern 文件（模拟已有 pattern）
        pattern_file = mem_dir / "tech" / "patterns" / "test_pattern.md"
        _write_initial_pattern(pattern_file, "test_pattern", "tech", cases=6, wins=4)

        result = distill_patterns(memory_dir=mem_dir, cycles_window=10)
        assert result is not None
        # 验证 tech 目录有 pattern 文件
        tech_patterns = list((mem_dir / "tech" / "patterns").glob("*.md"))
        assert len(tech_patterns) >= 1

    def test_distill_updates_existing_pattern_pnl(self, tmp_path):
        """已有 pattern 文件，蒸馏后 PnL track 应增量更新（FR-009）。"""
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter
        from cryptotrader.learning.memory import update_pattern_pnl

        mem_dir = _make_tmp_memory(tmp_path)
        pattern_file = mem_dir / "tech" / "patterns" / "rsi_bounce.md"
        _write_initial_pattern(pattern_file, "rsi_bounce", "tech", cases=5, wins=3)

        # 更新 PnL
        update_pattern_pnl(
            applied_patterns={"tech": ["rsi_bounce"]},
            pnl=150.0,
            memory_dir=mem_dir,
        )

        content = pattern_file.read_text(encoding="utf-8")
        data, _ = parse_frontmatter(content, path=pattern_file)
        pnl_track = data.get("pnl_track", {})
        assert pnl_track.get("cases", 0) == 6, "cases 应增加 1"

    def test_maturity_fsm_observed_to_probationary(self, tmp_path):
        """maturity FSM: observed → probationary（cases≥5 + win_rate≥0.60 时晋升）（FR-011）。"""
        from cryptotrader.learning.evolution.fsm import evaluate_transitions
        from cryptotrader.learning.memory import _load_pattern

        mem_dir = _make_tmp_memory(tmp_path)
        pattern_file = mem_dir / "tech" / "patterns" / "test_fsm.md"
        _write_initial_pattern(pattern_file, "test_fsm", "tech", cases=5, wins=4, maturity="observed")

        pattern = _load_pattern(pattern_file)
        result = evaluate_transitions(pattern)
        assert result is not None, "应晋升，实际无转移"
        assert result.maturity == "probationary", f"应到 probationary，实际: {result.maturity}"

    def test_reflection_failure_does_not_block_cycle(self, tmp_path):
        """reflection 失败（内部异常）不应抛出异常阻塞 cycle（FR-012）。"""
        from cryptotrader.learning.memory import distill_patterns

        mem_dir = _make_tmp_memory(tmp_path)
        # 不写任何 case 文件 — 应安静退出，不抛异常
        with patch("cryptotrader.learning.memory._read_cases", side_effect=RuntimeError("db error")):
            result = distill_patterns(memory_dir=mem_dir, cycles_window=10)
            # 不抛异常，返回 ReflectionRun（可能是空的）
            assert result is not None

    def test_applied_pattern_prefix_routing(self, tmp_path):
        """解析 <agent>::<pattern> 前缀，应分发到正确的 agent patterns 目录（FR-008）。"""
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter
        from cryptotrader.learning.memory import update_pattern_pnl

        mem_dir = _make_tmp_memory(tmp_path)
        # 创建 tech 和 chain 的 pattern 文件
        tech_file = mem_dir / "tech" / "patterns" / "rsi_bounce.md"
        chain_file = mem_dir / "chain" / "patterns" / "funding_squeeze.md"
        _write_initial_pattern(tech_file, "rsi_bounce", "tech", cases=3, wins=2)
        _write_initial_pattern(chain_file, "funding_squeeze", "chain", cases=3, wins=2)

        # 同时更新两个 agent 的 patterns
        update_pattern_pnl(
            applied_patterns={"tech": ["rsi_bounce"], "chain": ["funding_squeeze"]},
            pnl=200.0,
            memory_dir=mem_dir,
        )

        tech_data, _ = parse_frontmatter(tech_file.read_text(), path=tech_file)
        chain_data, _ = parse_frontmatter(chain_file.read_text(), path=chain_file)
        assert tech_data["pnl_track"]["cases"] == 4
        assert chain_data["pnl_track"]["cases"] == 4

    def test_manually_edited_pattern_body_preserved(self, tmp_path):
        """manually_edited: true 的 pattern 文件，body 不被重写（仅更新 pnl_track，FR-013 + spec edge case）。"""
        from cryptotrader.learning.memory import update_pattern_pnl

        mem_dir = _make_tmp_memory(tmp_path)
        pattern_file = mem_dir / "tech" / "patterns" / "manual_pattern.md"
        _write_initial_pattern(
            pattern_file,
            "manual_pattern",
            "tech",
            cases=5,
            wins=4,
            maturity="active",
            manually_edited=True,
            body="Custom body content that must not be overwritten.",
        )

        update_pattern_pnl(
            applied_patterns={"tech": ["manual_pattern"]},
            pnl=100.0,
            memory_dir=mem_dir,
        )

        content = pattern_file.read_text(encoding="utf-8")
        assert "Custom body content that must not be overwritten." in content, (
            "manually_edited pattern 的 body 不应被覆盖"
        )


def _write_initial_pattern(
    path: Path,
    name: str,
    agent: str,
    cases: int,
    wins: int,
    maturity: str = "observed",
    manually_edited: bool = False,
    body: str = "",
) -> None:
    """写入一个初始 pattern 文件（测试辅助）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    win_rate = wins / cases if cases > 0 else 0.0
    avg_pnl = 80.0 if wins > cases / 2 else -30.0
    fm = {
        "name": name,
        "agent": agent,
        "description": f"Test pattern {name}",
        "maturity": maturity,
        "manually_edited": manually_edited,
        "regime_tags": ["high_funding"],
        "pnl_track": {
            "cases": cases,
            "wins": wins,
            "win_rate": round(win_rate, 3),
            "avg_pnl": avg_pnl,
            "last_active": "2026-05-06",
        },
        "source_cycles": [],
        "created": "2026-05-01T00:00:00",
        "version": 1,
    }
    from cryptotrader.agents.skills._frontmatter import render_frontmatter

    content = render_frontmatter(fm)
    content += body if body else f"\n## Pattern: {name}\n\nConditions and notes.\n"
    path.write_text(content, encoding="utf-8")
