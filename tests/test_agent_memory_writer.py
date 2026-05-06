"""Tests for US2: Memory 层 case 写入（FR-006 / FR-007 / FR-013）。

TDD: 先 FAIL，实现 learning/memory.py 后 GREEN。
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


def _make_tmp_memory(tmp_path: Path) -> Path:
    """创建临时 agent_memory 目录骨架。"""
    from cryptotrader.agents.skills._io import ensure_memory_dirs

    ensure_memory_dirs(tmp_path)
    return tmp_path


class TestWriteCase:
    """测试 write_case()：per-cycle 单文件写入（FR-006）。"""

    def test_write_case_creates_file(self, tmp_path):
        """写入 cycle 后 agent_memory/cases/<cycle_id>.md 应存在。"""
        from cryptotrader.learning.memory import write_case

        mem_dir = _make_tmp_memory(tmp_path)
        cycle_id = "2026-05-06-cycle-abcd1234"
        path = write_case(
            cycle_id=cycle_id,
            pair="BTC/USDT",
            agent_analyses={
                "tech": "bullish analysis",
                "chain": "neutral chain",
                "news": "positive news",
                "macro": "macro ok",
            },
            verdict_action="long",
            verdict_reasoning="Strong bullish convergence.",
            applied_patterns=["tech::rsi_oversold_bounce"],
            risk_gate_passed=True,
            execution_status={"succeeded": True},
            final_pnl=None,
            memory_dir=mem_dir,
        )
        assert path.exists(), "case 文件应被创建"
        assert path.name.endswith(".md"), "case 文件应为 .md 格式"

    def test_write_case_frontmatter_fields(self, tmp_path):
        """写入的 case 文件 frontmatter 应含必填字段（FR-006）。"""
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter
        from cryptotrader.learning.memory import write_case

        mem_dir = _make_tmp_memory(tmp_path)
        cycle_id = "2026-05-06-cycle-abcd1234"
        path = write_case(
            cycle_id=cycle_id,
            pair="BTC/USDT",
            agent_analyses={"tech": "bullish"},
            verdict_action="long",
            verdict_reasoning="Bullish.",
            applied_patterns=[],
            risk_gate_passed=True,
            execution_status=None,
            final_pnl=None,
            memory_dir=mem_dir,
        )
        content = path.read_text(encoding="utf-8")
        data, _ = parse_frontmatter(content, path=path)
        assert data["cycle_id"] == cycle_id
        assert data["pair"] == "BTC/USDT"
        assert data["verdict_action"] == "long"
        assert "final_pnl" in data

    def test_write_case_body_contains_agent_analyses(self, tmp_path):
        """case 文件 body 应包含 4 个 agent 各自的 analysis 段落（FR-006）。"""
        from cryptotrader.learning.memory import write_case

        mem_dir = _make_tmp_memory(tmp_path)
        path = write_case(
            cycle_id="2026-05-06-cycle-test0001",
            pair="ETH/USDT",
            agent_analyses={
                "tech": "RSI oversold at 28",
                "chain": "Funding negative -0.02%",
                "news": "Positive ETF news",
                "macro": "DXY declining",
            },
            verdict_action="long",
            verdict_reasoning="All signals converge bullish.",
            applied_patterns=["tech::rsi_oversold", "chain::negative_funding_squeeze"],
            risk_gate_passed=True,
            execution_status=None,
            final_pnl=None,
            memory_dir=mem_dir,
        )
        body = path.read_text(encoding="utf-8")
        assert "RSI oversold at 28" in body
        assert "Funding negative -0.02%" in body
        assert "Positive ETF news" in body
        assert "DXY declining" in body
        assert "tech::rsi_oversold" in body

    def test_update_final_pnl_backfills(self, tmp_path):
        """update_final_pnl() 应回填 frontmatter final_pnl 字段（FR-006 平仓回填）。"""
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter
        from cryptotrader.learning.memory import update_final_pnl, write_case

        mem_dir = _make_tmp_memory(tmp_path)
        cycle_id = "2026-05-06-cycle-pnltest"
        path = write_case(
            cycle_id=cycle_id,
            pair="BTC/USDT",
            agent_analyses={"tech": "bullish"},
            verdict_action="long",
            verdict_reasoning="Bullish.",
            applied_patterns=[],
            risk_gate_passed=True,
            execution_status=None,
            final_pnl=None,
            memory_dir=mem_dir,
        )
        # 初始 final_pnl 为 None
        content = path.read_text(encoding="utf-8")
        data, _ = parse_frontmatter(content, path=path)
        assert data.get("final_pnl") is None

        # 回填 PnL
        update_final_pnl(cycle_id=cycle_id, pnl=120.5, memory_dir=mem_dir)
        content2 = path.read_text(encoding="utf-8")
        data2, _ = parse_frontmatter(content2, path=path)
        assert data2["final_pnl"] == pytest.approx(120.5)

    def test_write_case_failure_does_not_raise(self, tmp_path):
        """write_case 写失败（如目录不可写）应 logger.warning 后不抛异常（FR-007）。"""
        # 传入一个不存在且无法创建的路径（实际上会尝试创建目录，测试异常路径）
        # 用 mock 模拟原子写入失败
        from unittest.mock import patch

        from cryptotrader.learning.memory import write_case

        mem_dir = _make_tmp_memory(tmp_path)

        with patch("cryptotrader.agents.skills._io.atomic_write", side_effect=OSError("disk full")):
            # 不应抛出异常
            result = write_case(
                cycle_id="2026-05-06-cycle-errtest",
                pair="BTC/USDT",
                agent_analyses={"tech": "neutral"},
                verdict_action="hold",
                verdict_reasoning="No signal.",
                applied_patterns=[],
                risk_gate_passed=True,
                execution_status=None,
                final_pnl=None,
                memory_dir=mem_dir,
            )
            # 写失败返回 None 或 dummy path，不抛
            assert result is None or isinstance(result, Path)

    def test_write_case_atomic(self, tmp_path):
        """case 文件应通过原子写入（不存在半成品文件）（FR-013）。"""
        from cryptotrader.learning.memory import write_case

        mem_dir = _make_tmp_memory(tmp_path)
        cycle_id = "2026-05-06-cycle-atomictest"
        path = write_case(
            cycle_id=cycle_id,
            pair="BTC/USDT",
            agent_analyses={"tech": "bullish"},
            verdict_action="long",
            verdict_reasoning="Bullish signal.",
            applied_patterns=[],
            risk_gate_passed=True,
            execution_status=None,
            final_pnl=None,
            memory_dir=mem_dir,
        )
        # 文件存在且完整（不为空）
        assert path is not None
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert len(content) > 50, "case 文件不应为空或过短"

    def test_write_case_correct_path(self, tmp_path):
        """case 文件路径应在 agent_memory/cases/ 下（per-cycle 单文件，非 per-agent，FR-006）。"""
        from cryptotrader.learning.memory import write_case

        mem_dir = _make_tmp_memory(tmp_path)
        cycle_id = "2026-05-06-cycle-pathtest"
        path = write_case(
            cycle_id=cycle_id,
            pair="BTC/USDT",
            agent_analyses={"tech": "neutral"},
            verdict_action="hold",
            verdict_reasoning="Neutral.",
            applied_patterns=[],
            risk_gate_passed=False,
            execution_status=None,
            final_pnl=None,
            memory_dir=mem_dir,
        )
        # 路径应在 <mem_dir>/cases/ 下，而不是 <mem_dir>/tech/cases/ 等
        assert path is not None
        assert "cases" in str(path)
        # 不应包含 agent 子目录
        relative = path.relative_to(mem_dir)
        parts = relative.parts
        assert parts[0] == "cases", f"case 文件应在 cases/ 下，实际路径: {relative}"
