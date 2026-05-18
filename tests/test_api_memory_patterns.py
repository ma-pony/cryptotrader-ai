"""GET /api/memory/patterns endpoint 单测 — spec 024 SC-P3 补完。

5 用例：
  (a) total >= 3（spec 021 已在磁盘产出 3 个 patterns）
  (b) ?agent=tech 过滤仅返回 tech patterns
  (c) ?maturity=observed 过滤
  (d) ?limit=2 截断结果
  (e) ?agent=news 空目录返回 items=[] total=0（不 500）
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AUTH_MODE", "disabled")
os.environ.setdefault("API_KEY", "test-key-024")

# ── Pattern MD 模板 ────────────────────────────────────────────────────────────

_PATTERN_TEMPLATE = """\
---
name: {name}
agent: {agent}
description: "Auto-distilled pattern: {name} (from {n} cases)"
regime_tags: {regime_tags}
maturity: {maturity}
pnl_track:
  pnls: {pnls}
source_cycles: {source_cycles}
created: "2026-05-11T03:00:00Z"
version: 1
manually_edited: false
---

# {name}

Auto-distilled from {n} cases.
"""


def _write_pattern(
    memory_root: Path,
    agent: str,
    name: str,
    maturity: str = "observed",
    regime_tags: list[str] | None = None,
    pnls: list[float] | None = None,
    n: int = 6,
) -> Path:
    """在 memory_root/<agent>/patterns/<name>.md 写入 PatternRecord 文件。"""
    patterns_dir = memory_root / agent / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)
    path = patterns_dir / f"{name}.md"
    path.write_text(
        _PATTERN_TEMPLATE.format(
            name=name,
            agent=agent,
            maturity=maturity,
            regime_tags=str(regime_tags or ["high_vol"]),
            pnls=str(pnls or [12.5, -3.2, 8.0]),
            source_cycles=str([f"cycle-{i}" for i in range(min(n, 5))]),
            n=n,
        ),
        encoding="utf-8",
    )
    return path


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def memory_root(tmp_path: Path) -> Path:
    """创建含 3 个 patterns 的临时 memory 目录（模拟 spec 021 产出）。

    tech:  volume-spike-rsi-overbought (observed)
    chain: on-chain-whale-accumulation (probationary)
    macro: high-funding-fade           (observed)
    """
    _write_pattern(tmp_path, "tech", "volume-spike-rsi-overbought", maturity="observed")
    _write_pattern(tmp_path, "chain", "on-chain-whale-accumulation", maturity="probationary")
    _write_pattern(tmp_path, "macro", "high-funding-fade", maturity="observed")
    return tmp_path


@pytest.fixture
def client(memory_root: Path) -> TestClient:
    """TestClient，memory router 的 _MEMORY_ROOT 指向 memory_root。"""
    from unittest.mock import patch

    import api.routes.memory as mem_module
    from api.main import app

    original = mem_module._MEMORY_ROOT
    mem_module._MEMORY_ROOT = memory_root

    with patch("api.routes.memory._MEMORY_ROOT", memory_root):
        yield TestClient(app, raise_server_exceptions=False)

    mem_module._MEMORY_ROOT = original


# ── (a) total >= 3 ────────────────────────────────────────────────────────────


class TestPatternsTotal:
    def test_total_gte_3(self, client: TestClient, memory_root: Path) -> None:
        """(a) spec 021 已产出 ≥ 3 patterns → total >= 3。"""
        from unittest.mock import patch

        with patch("api.routes.memory._MEMORY_ROOT", memory_root):
            resp = client.get("/api/memory/patterns")

        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 3
        assert len(data["items"]) >= 3

    def test_response_schema_fields(self, client: TestClient, memory_root: Path) -> None:
        """响应包含 PatternRecordResponse 全部字段。"""
        from unittest.mock import patch

        with patch("api.routes.memory._MEMORY_ROOT", memory_root):
            resp = client.get("/api/memory/patterns")

        assert resp.status_code == 200
        item = resp.json()["items"][0]
        for field in ("name", "agent", "description", "maturity", "regime_tags", "pnl_track", "source_cycles", "body"):
            assert field in item, f"缺少字段: {field}"


# ── (b) ?agent=tech 过滤 ───────────────────────────────────────────────────────


class TestAgentFilter:
    def test_agent_tech_filter(self, client: TestClient, memory_root: Path) -> None:
        """(b) ?agent=tech 仅返回 tech patterns。"""
        from unittest.mock import patch

        with patch("api.routes.memory._MEMORY_ROOT", memory_root):
            resp = client.get("/api/memory/patterns?agent=tech")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for item in data["items"]:
            assert item["agent"] == "tech"

    def test_invalid_agent_returns_400(self, client: TestClient, memory_root: Path) -> None:
        """无效 agent 参数返回 400。"""
        from unittest.mock import patch

        with patch("api.routes.memory._MEMORY_ROOT", memory_root):
            resp = client.get("/api/memory/patterns?agent=invalid")

        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_query"


# ── (c) ?maturity=observed 过滤 ───────────────────────────────────────────────


class TestMaturityFilter:
    def test_maturity_observed_filter(self, client: TestClient, memory_root: Path) -> None:
        """(c) ?maturity=observed 过滤仅返回 maturity=observed 的 patterns。"""
        from unittest.mock import patch

        with patch("api.routes.memory._MEMORY_ROOT", memory_root):
            resp = client.get("/api/memory/patterns?maturity=observed")

        assert resp.status_code == 200
        data = resp.json()
        # fixture 中有 tech(observed) + macro(observed) = 2 条
        assert data["total"] >= 1
        for item in data["items"]:
            assert item["maturity"] == "observed"

    def test_maturity_probationary_filter(self, client: TestClient, memory_root: Path) -> None:
        """maturity=probationary 仅返回 probationary patterns。"""
        from unittest.mock import patch

        with patch("api.routes.memory._MEMORY_ROOT", memory_root):
            resp = client.get("/api/memory/patterns?maturity=probationary")

        assert resp.status_code == 200
        data = resp.json()
        for item in data["items"]:
            assert item["maturity"] == "probationary"

    def test_invalid_maturity_returns_400(self, client: TestClient, memory_root: Path) -> None:
        """无效 maturity 参数返回 400。"""
        from unittest.mock import patch

        with patch("api.routes.memory._MEMORY_ROOT", memory_root):
            resp = client.get("/api/memory/patterns?maturity=unknown_state")

        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_query"


# ── (d) ?limit=2 截断 ─────────────────────────────────────────────────────────


class TestLimitParam:
    def test_limit_truncates_results(self, client: TestClient, memory_root: Path) -> None:
        """(d) ?limit=2 截断 items 为 2 条（总数 ≥ 3 时有效）。"""
        from unittest.mock import patch

        with patch("api.routes.memory._MEMORY_ROOT", memory_root):
            resp = client.get("/api/memory/patterns?limit=2")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        # total 反映实际返回数（截断后）
        assert data["total"] == 2

    def test_limit_0_returns_empty(self, client: TestClient, memory_root: Path) -> None:
        """?limit=0 返回 items=[], total=0。"""
        from unittest.mock import patch

        with patch("api.routes.memory._MEMORY_ROOT", memory_root):
            resp = client.get("/api/memory/patterns?limit=0")

        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0


# ── (e) ?agent=news 空目录返回 items=[] total=0（不 500）─────────────────────


class TestEmptyDir:
    def test_empty_agent_dir_returns_empty(self, client: TestClient, memory_root: Path) -> None:
        """(e) ?agent=news — news patterns 目录不存在 → items=[], total=0，不 500。"""
        # fixture 中没有创建 news patterns 目录
        from unittest.mock import patch

        with patch("api.routes.memory._MEMORY_ROOT", memory_root):
            resp = client.get("/api/memory/patterns?agent=news")

        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_no_memory_root_returns_empty(self, tmp_path: Path) -> None:
        """memory_root 完全空（无任何子目录）→ items=[], total=0，不 500。"""
        from unittest.mock import patch

        import api.routes.memory as mem_module
        from api.main import app

        empty_root = tmp_path / "nonexistent_memory"
        # 故意不创建 empty_root

        original = mem_module._MEMORY_ROOT
        mem_module._MEMORY_ROOT = empty_root

        with patch("api.routes.memory._MEMORY_ROOT", empty_root):
            tc = TestClient(app, raise_server_exceptions=False)
            resp = tc.get("/api/memory/patterns")

        mem_module._MEMORY_ROOT = original

        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0
