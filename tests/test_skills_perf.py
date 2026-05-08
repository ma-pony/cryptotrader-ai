"""性能微基准 — FR-019a / FR-025a。

T053: 验证 LRU mtime 缓存命中时 parse_skill_md() 的性能，以及
      load_skill_tool 的 rate-limit 开销不超过合理阈值。

使用 time.perf_counter 而非 pytest-benchmark，确保无额外依赖。
"""

from __future__ import annotations

import time
from pathlib import Path


def _write_skill(tmp_path: Path, name: str = "perf-test", scope: str = "shared") -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: {name}\ndescription: perf test skill\nscope: {scope}\n---\n\n## Body\nContent here.\n"
    )
    return skill_file


def test_parse_skill_md_cache_hit_is_fast(tmp_path):
    """LRU mtime 缓存命中时，parse_skill_md() 第二次调用应 <1ms。"""
    from cryptotrader.agents.skills.loader import _clear_cache, parse_skill_md

    _clear_cache()
    skill_file = _write_skill(tmp_path)

    # 第一次调用（冷启动，读文件）
    parse_skill_md(skill_file)

    # 第二次调用（缓存命中）
    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        parse_skill_md(skill_file)
    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_ms = elapsed_ms / iterations
    # 缓存命中应 << 1ms（通常 < 0.01ms）
    assert avg_ms < 1.0, f"平均缓存命中耗时 {avg_ms:.3f}ms 超过 1ms 阈值"


def test_discover_skills_for_agent_is_fast(tmp_path):
    """discover_skills_for_agent() 扫描 5 个 SKILL.md 应 <10ms（含 LRU 热身）。"""
    from cryptotrader.agents.skills.loader import _clear_cache, discover_skills_for_agent

    _clear_cache()

    # 写入 5 个 skill 文件
    agents = ["tech", "chain", "news", "macro"]
    for agent_id in agents:
        _write_skill(tmp_path, f"{agent_id}-analysis", scope=f"agent:{agent_id}")
    _write_skill(tmp_path, "trading-knowledge", scope="shared")

    # 冷启动
    discover_skills_for_agent("tech", skill_dir=tmp_path)

    # 热缓存测量
    start = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        discover_skills_for_agent("tech", skill_dir=tmp_path)
    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_ms = elapsed_ms / iterations
    assert avg_ms < 10.0, f"平均 discover 耗时 {avg_ms:.3f}ms 超过 10ms 阈值"


def test_rate_limit_check_overhead_is_negligible():
    """load_skill rate-limit 检查本身（不含文件 IO）开销应 <0.1ms/次。"""
    from cryptotrader.agents.skills.tool import _reset_call_counter

    _reset_call_counter("perf-trace-id")

    # 模拟 rate-limit 检查的最简路径（直接调 _reset）
    start = time.perf_counter()
    iterations = 1000
    for i in range(iterations):
        _reset_call_counter(f"perf-trace-{i}")
    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_ms = elapsed_ms / iterations
    assert avg_ms < 0.1, f"平均 rate-limit 重置耗时 {avg_ms:.4f}ms 超过 0.1ms 阈值"


def test_get_available_skills_is_fast(tmp_path):
    """DefaultSkillProvider.get_available_skills() 热路径应 <5ms（替代已删 middleware 测试）。"""
    from cryptotrader.agents.prompt_builder import DefaultSkillProvider
    from cryptotrader.agents.skills.loader import _clear_cache

    _clear_cache()
    _write_skill(tmp_path, "tech-analysis", scope="agent:tech")
    _write_skill(tmp_path, "trading-knowledge", scope="shared")

    provider = DefaultSkillProvider(skills_root=tmp_path)
    # 冷启动
    provider.get_available_skills("tech", snapshot={})

    # 热测量
    start = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        provider.get_available_skills("tech", snapshot={})
    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_ms = elapsed_ms / iterations
    assert avg_ms < 5.0, f"平均 get_available_skills 耗时 {avg_ms:.3f}ms 超过 5ms 阈值"
