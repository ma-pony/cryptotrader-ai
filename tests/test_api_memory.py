"""spec 018 Memory API route tests — tests/test_api_memory.py

SC-Z12: >= 6 use cases PASS.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """TestClient with memory root patched to tmp_path."""
    import api.routes.memory as mem_module

    original = mem_module._MEMORY_ROOT
    mem_module._MEMORY_ROOT = tmp_path
    from api.main import app

    with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
        yield TestClient(app, raise_server_exceptions=False)
    mem_module._MEMORY_ROOT = original


def _make_pattern(
    tmp_path: Path,
    agent: str = "tech",
    name: str = "test_rule",
    maturity: str = "active",
    wins: int = 5,
    cases: int = 10,
) -> None:
    """Write a minimal pattern markdown file."""
    pattern_dir = tmp_path / agent / "patterns"
    pattern_dir.mkdir(parents=True, exist_ok=True)
    path = pattern_dir / f"{name}.md"
    content = f"""---
name: {name}
agent: {agent}
description: test description
maturity: {maturity}
manually_edited: false
regime_tags: []
pnl_track:
  cases: {cases}
  wins: {wins}
  win_rate: {wins / max(cases, 1):.4f}
  avg_pnl: 10.5
  last_active: "2026-05-01"
source_cycles: []
created: "2026-04-01T00:00:00+00:00"
version: 1
importance: 0.7
access_count: 3
last_accessed_at: "2026-05-07T12:00:00+00:00"
last_modified_at: "2026-05-06T10:00:00+00:00"
fundamental_failure_streak: 0
---
## Rule Body
Test rule body content.
"""
    path.write_text(content, encoding="utf-8")


def _make_case(
    tmp_path: Path,
    cycle_id: str = "cycle_001",
    pair: str = "BTC/USDT",
    verdict_action: str = "long",
    final_pnl: float = 50.0,
    ive_classification: dict | None = None,
    applied_patterns: list[str] | None = None,
    timestamp: str | None = None,
) -> None:
    """Write a minimal case markdown file."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    path = cases_dir / f"{cycle_id}.md"
    ts = timestamp or datetime.now(UTC).isoformat()
    ive_str = json.dumps(ive_classification) if ive_classification else "null"
    applied_str = json.dumps(applied_patterns or [])
    content = f"""---
cycle_id: {cycle_id}
timestamp: {ts}
pair: {pair}
verdict_action: {verdict_action}
final_pnl: {final_pnl}
risk_gate_passed: true
applied_patterns: {applied_str}
ive_classification: {ive_str}
---
# Cycle {cycle_id}
"""
    path.write_text(content, encoding="utf-8")


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestGetMemoryRules:
    def test_rules_returns_200_with_tech_agent(self, client: TestClient, tmp_path: Path) -> None:
        """T042(a): GET /api/memory/rules?agent=tech 返回 200 + JSON。"""
        _make_pattern(tmp_path, agent="tech", name="rule_one")

        with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
            resp = client.get("/api/memory/rules?agent=tech")

        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert "total" in body
        assert body["total"] >= 1
        item = body["items"][0]
        assert item["agent"] == "tech"
        assert item["name"] == "rule_one"
        assert "pnl_track" in item

    def test_rules_invalid_status_returns_400(self, client: TestClient, tmp_path: Path) -> None:
        """T042(e): 错误参数 status=invalid 返回 400。"""
        with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
            resp = client.get("/api/memory/rules?status=invalid")

        assert resp.status_code == 400
        body = resp.json()
        assert body.get("error") == "invalid_query"

    def test_rules_unknown_agent_returns_404(self, client: TestClient, tmp_path: Path) -> None:
        """T042(f): agent 不存在返回 404。"""
        with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
            resp = client.get("/api/memory/rules?agent=unknown_agent")

        assert resp.status_code == 404
        body = resp.json()
        assert body.get("error") == "agent_not_found"

    def test_rules_filters_by_status(self, client: TestClient, tmp_path: Path) -> None:
        """rules endpoint 按 status 过滤正确。"""
        _make_pattern(tmp_path, agent="tech", name="active_rule", maturity="active")
        _make_pattern(tmp_path, agent="tech", name="observed_rule", maturity="observed")

        with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
            resp = client.get("/api/memory/rules?agent=tech&status=active")

        assert resp.status_code == 200
        items = resp.json()["items"]
        assert all(i["maturity"] == "active" for i in items)
        names = [i["name"] for i in items]
        assert "active_rule" in names
        assert "observed_rule" not in names

    def test_rules_cache_control_header(self, client: TestClient, tmp_path: Path) -> None:
        """rules 响应含 Cache-Control: max-age=30。"""
        with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
            resp = client.get("/api/memory/rules")

        assert resp.status_code == 200
        assert "max-age=30" in resp.headers.get("cache-control", "")


class TestGetMemoryCases:
    def test_cases_returns_200(self, client: TestClient, tmp_path: Path) -> None:
        """T042(b): GET /api/memory/cases?agent=macro 返回近期 case。"""
        _make_case(
            tmp_path,
            cycle_id="case_001",
            applied_patterns=["macro::high_funding_fade"],
            ive_classification={"failure_type": "noise", "confidence": 0.3, "reasoning": "市场噪声"},
        )

        with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
            resp = client.get("/api/memory/cases?agent=macro")

        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert body["total"] >= 1
        item = body["items"][0]
        assert item["cycle_id"] == "case_001"
        assert item["ive_classification"]["failure_type"] == "noise"

    def test_cases_invalid_from_returns_400(self, client: TestClient, tmp_path: Path) -> None:
        """cases endpoint 错误 from 参数返回 400。"""
        with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
            resp = client.get("/api/memory/cases?from=not-a-date")

        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_query"

    def test_cases_returns_in_descending_order(self, client: TestClient, tmp_path: Path) -> None:
        """cases 按 timestamp 倒序返回。"""
        _make_case(tmp_path, cycle_id="old_case", timestamp="2026-05-06T00:00:00Z")
        _make_case(tmp_path, cycle_id="new_case", timestamp="2026-05-08T12:00:00Z")

        with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
            resp = client.get("/api/memory/cases")

        assert resp.status_code == 200
        ids = [i["cycle_id"] for i in resp.json()["items"]]
        assert ids.index("new_case") < ids.index("old_case")


class TestGetMemoryTransitions:
    def test_transitions_returns_200(self, client: TestClient, tmp_path: Path) -> None:
        """T042(c): GET /api/memory/transitions?since=... 返回 events。"""
        # mock provider with no-op transitions (empty dir)
        mock_provider = MagicMock()
        mock_provider.evaluate_all_rules.return_value = []
        mock_load = MagicMock(return_value=mock_provider)

        with (
            patch("api.routes.memory._MEMORY_ROOT", tmp_path),
            patch("api.routes.memory._load_provider", mock_load),
        ):
            resp = client.get("/api/memory/transitions?since=2026-05-08T00:00:00")

        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert "total" in body

    def test_transitions_invalid_since_returns_400(self, client: TestClient, tmp_path: Path) -> None:
        """transitions endpoint 错误 since 参数返回 400。"""
        with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
            resp = client.get("/api/memory/transitions?since=bad-date")

        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_query"

    def test_transitions_cache_control_header(self, client: TestClient, tmp_path: Path) -> None:
        """transitions 响应含 Cache-Control: max-age=30。"""
        mock_provider = MagicMock()
        mock_provider.evaluate_all_rules.return_value = []
        mock_load = MagicMock(return_value=mock_provider)

        with (
            patch("api.routes.memory._MEMORY_ROOT", tmp_path),
            patch("api.routes.memory._load_provider", mock_load),
        ):
            resp = client.get("/api/memory/transitions")

        assert resp.status_code == 200
        assert "max-age=30" in resp.headers.get("cache-control", "")


class TestGetMemoryArchived:
    def test_archived_returns_200(self, client: TestClient, tmp_path: Path) -> None:
        """T042(d): GET /api/memory/archived 返回 archived list。"""
        archived_dir = tmp_path / "tech" / "patterns" / ".archived"
        archived_dir.mkdir(parents=True, exist_ok=True)
        path = archived_dir / "old_rule.md"
        content = """---
name: old_rule
agent: tech
description: old rule
maturity: archived
manually_edited: false
regime_tags: []
pnl_track:
  cases: 10
  wins: 2
  win_rate: 0.2
  avg_pnl: -30.0
  last_active: "2026-04-01"
source_cycles: []
created: "2026-03-01T00:00:00+00:00"
version: 3
importance: 0.4
access_count: 10
last_accessed_at: "2026-04-30T00:00:00+00:00"
last_modified_at: "2026-04-30T00:00:00+00:00"
fundamental_failure_streak: 3
---
## Rule Body
Archived rule.
"""
        path.write_text(content, encoding="utf-8")

        with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
            resp = client.get("/api/memory/archived")

        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert body["total"] >= 1
        item = body["items"][0]
        assert item["name"] == "old_rule"
        assert item["fundamental_failure_streak"] == 3
        assert "final_pnl_track" in item

    def test_archived_cache_control_header(self, client: TestClient, tmp_path: Path) -> None:
        """archived 响应含 Cache-Control: max-age=300。"""
        with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
            resp = client.get("/api/memory/archived")

        assert resp.status_code == 200
        assert "max-age=300" in resp.headers.get("cache-control", "")

    def test_archived_empty_returns_empty_list(self, client: TestClient, tmp_path: Path) -> None:
        """archived 目录为空时返回 empty items。"""
        with patch("api.routes.memory._MEMORY_ROOT", tmp_path):
            resp = client.get("/api/memory/archived")

        assert resp.status_code == 200
        assert resp.json()["total"] == 0
        assert resp.json()["items"] == []
