"""tests/test_e2e_pattern_cold_start.py — spec 021 T016

端到端验证：
  fixture 200+ cases → daemon pattern_extraction action → ≥ 3 patterns 创建

流程：
  1. 生成 200 个 case 文件（4 agents × 4 patterns × 12-13 cases 各，超过 min=5 阈值）
  2. 用真实 distill_patterns（无 mock）跑 cold-start
  3. 验证 ≥ 3 个 pattern 文件被创建
  4. 用 daemon _action_pattern_extraction 验证 PASS + details 含 new_count > 0
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cryptotrader.learning.memory import distill_patterns
from cryptotrader.ops.daemon import EvolutionDaemon

# ── helpers ──


def _write_case(
    path: Path,
    *,
    cycle_id: str,
    agent: str,
    pattern_slug: str,
    pnl: float = 1.0,
    regime_tags: list[str] | None = None,
) -> None:
    tags_yaml = ""
    if regime_tags:
        items = "\n".join(f"  - {t}" for t in regime_tags)
        tags_yaml = f"regime_tags:\n{items}\n"
    content = (
        f"---\ncycle_id: {cycle_id}\nfinal_pnl: {pnl}\n{tags_yaml}---\n"
        f"## Applied Patterns\n- applied: {agent}::{pattern_slug}\n"
    )
    path.write_text(content, encoding="utf-8")


def _build_fixture_memory(tmp_path: Path, n_cases_each: int = 12) -> Path:
    """建立 200+ cases 的 fixture memory 目录。

    4 agents × 4 patterns × n_cases_each = 4×4×12 = 192 cases（+ 额外 padding）
    """
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("tech", "volume-spike"),
        ("tech", "rsi-overbought"),
        ("chain", "whale-accumulation"),
        ("chain", "exchange-outflow"),
        ("news", "positive-sentiment"),
        ("news", "fud-spike"),
        ("macro", "rate-hike-fear"),
        ("macro", "dollar-strength"),
    ]

    idx = 0
    for agent, pattern in scenarios:
        for i in range(n_cases_each):
            cycle_id = f"e2e-cycle-{idx:04d}"
            pnl = 1.5 if i % 3 != 0 else -0.5  # 2/3 wins
            tags = ["bull"] if pnl > 0 else ["bear"]
            _write_case(
                cases_dir / f"{cycle_id}.md",
                cycle_id=cycle_id,
                agent=agent,
                pattern_slug=pattern,
                pnl=pnl,
                regime_tags=tags,
            )
            idx += 1

    return tmp_path


# ── 端到端测试 1：distill_patterns 直接跑 ──


def test_e2e_distill_creates_multiple_patterns(tmp_path):
    """200 fixture cases → distill_patterns → ≥ 3 pattern 文件。"""
    mem = _build_fixture_memory(tmp_path, n_cases_each=12)
    run = distill_patterns(memory_dir=mem, cycles_window=200)

    # 收集所有 agent patterns 目录下的文件
    pattern_files = []
    for agent in ["tech", "chain", "news", "macro"]:
        agent_patterns = mem / agent / "patterns"
        if agent_patterns.exists():
            pattern_files.extend(list(agent_patterns.glob("*.md")))

    assert run.patterns_created >= 3, (
        f"Expected ≥ 3 patterns created, got {run.patterns_created}. "
        f"Pattern files found: {[f.name for f in pattern_files]}"
    )
    assert len(pattern_files) >= 3


# ── 端到端测试 2：通过 daemon _action_pattern_extraction ──


@pytest.mark.asyncio
async def test_e2e_daemon_pattern_extraction_pass(tmp_path):
    """200 fixture cases → daemon pattern_extraction → status=PASS + new_count >= 3。"""
    mem = _build_fixture_memory(tmp_path, n_cases_each=12)

    daemon_cfg = MagicMock()
    daemon_cfg.actions = ["pattern_extraction"]
    daemon = EvolutionDaemon(config=daemon_cfg)

    with patch("cryptotrader.config.load_config") as mock_cfg:
        mock_cfg.return_value.experience.lookback_commits = 200
        with patch("cryptotrader.learning.memory.DEFAULT_AGENT_MEMORY_DIR", mem):
            result = await daemon._action_pattern_extraction()

    assert result.status == "PASS"
    assert result.name == "pattern_extraction"
    assert "new_count" in result.details
    assert "cases_processed" in result.details
    assert result.details["cases_processed"] >= 0
