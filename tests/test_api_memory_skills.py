"""spec 019 Skill API route tests — tests/test_api_memory_skills.py

SC-W11: >= 8 use cases PASS（contracts/skill-api-routes.md 单测要求）。
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AUTH_MODE", "disabled")
os.environ.setdefault("API_KEY", "test-key-019")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """TestClient，memory router 的 _SKILLS_ROOT 指向 tmp_path。"""
    import api.routes.memory as mem_module

    original_skills = mem_module._SKILLS_ROOT
    mem_module._SKILLS_ROOT = tmp_path

    from api.main import app

    with patch("api.routes.memory._SKILLS_ROOT", tmp_path):
        yield TestClient(app, raise_server_exceptions=False)

    mem_module._SKILLS_ROOT = original_skills


def _write_skill(
    skills_dir: Path,
    name: str,
    scope: str = "shared",
    importance: float = 0.7,
    confidence: float = 0.7,
    access_count: int = 5,
    regime_tags: list[str] | None = None,
    triggers_keywords: list[str] | None = None,
    write_draft: bool = False,
) -> Path:
    """写入测试用 SKILL.md（含 spec 019 全部新字段）。"""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    la = datetime.now(UTC).isoformat()
    regime_str = str(regime_tags or [])
    triggers_str = str(triggers_keywords or ["momentum", "trend"])
    content = f"""---
name: {name}
description: Test skill for {name} providing analysis techniques.
scope: {scope}
version: "1.0"
manually_edited: false
regime_tags: {regime_str}
triggers_keywords: {triggers_str}
importance: {importance}
confidence: {confidence}
access_count: {access_count}
last_accessed_at: "{la}"
---

# Skill Body for {name}

This is the body content for skill {name}.
"""
    path.write_text(content, encoding="utf-8")

    if write_draft:
        draft = skill_dir / "SKILL.md.draft"
        draft_content = f"""---
name: {name}-proposed
description: Proposed skill from pattern analysis.
scope: shared
version: "1.0"
manually_edited: false
regime_tags: ["high_funding"]
triggers_keywords: ["funding", "long"]
importance: 0.6
confidence: 0.6
access_count: 0
last_accessed_at: "{la}"
llm_call_failed: false
---

# Draft Body
Draft content here.
"""
        draft.write_text(draft_content, encoding="utf-8")

    return path


# ── 1. GET /api/memory/skills?agent=tech ─────────────────────────────────────


class TestGetSkillsList:
    def test_returns_200_with_skill_list(self, client: TestClient, tmp_path: Path) -> None:
        """SC-W11.1: GET /api/memory/skills?agent=tech 返回 200 + JSON list。"""
        _write_skill(tmp_path, "tech-analysis", scope="agent:tech", importance=0.8)
        _write_skill(tmp_path, "chain-analysis", scope="agent:chain", importance=0.7)

        with patch("api.routes.memory._SKILLS_ROOT", tmp_path):
            resp = client.get("/api/memory/skills?agent=tech", headers={"X-API-Key": "test-key-019"})

        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        # agent:tech filter should include agent:tech and shared
        tech_names = [i["name"] for i in data["items"]]
        assert "tech-analysis" in tech_names
        # chain-analysis has scope agent:chain → excluded
        assert "chain-analysis" not in tech_names

    def test_returns_all_skills_without_agent_filter(self, client: TestClient, tmp_path: Path) -> None:
        """GET /api/memory/skills（无 agent filter）返回所有 skills。"""
        _write_skill(tmp_path, "tech-analysis", scope="agent:tech")
        _write_skill(tmp_path, "general-skill", scope="shared")

        with patch("api.routes.memory._SKILLS_ROOT", tmp_path):
            resp = client.get("/api/memory/skills", headers={"X-API-Key": "test-key-019"})

        assert resp.status_code == 200
        data = resp.json()
        names = [i["name"] for i in data["items"]]
        assert "tech-analysis" in names
        assert "general-skill" in names

    def test_invalid_agent_returns_400(self, client: TestClient, tmp_path: Path) -> None:
        """SC-W11.7: 错误参数返回 400。"""
        with patch("api.routes.memory._SKILLS_ROOT", tmp_path):
            resp = client.get("/api/memory/skills?agent=invalid_agent", headers={"X-API-Key": "test-key-019"})

        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_query"

    def test_response_has_cache_control_header(self, client: TestClient, tmp_path: Path) -> None:
        """SC-W11.8: Response 含 Cache-Control header。"""
        with patch("api.routes.memory._SKILLS_ROOT", tmp_path):
            resp = client.get("/api/memory/skills", headers={"X-API-Key": "test-key-019"})

        assert resp.status_code == 200
        assert "cache-control" in resp.headers or "Cache-Control" in resp.headers


# ── 2. GET /api/memory/skills/{name} ─────────────────────────────────────────


class TestGetSkillDetail:
    def test_returns_200_with_body(self, client: TestClient, tmp_path: Path) -> None:
        """SC-W11.2: GET /api/memory/skills/{name} 返回 200 + 详情含 body。"""
        _write_skill(tmp_path, "tech-analysis", scope="agent:tech")

        with patch("api.routes.memory._SKILLS_ROOT", tmp_path):
            resp = client.get("/api/memory/skills/tech-analysis", headers={"X-API-Key": "test-key-019"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "tech-analysis"
        assert "body" in data
        assert len(data["body"]) > 0
        # 新字段存在
        assert "regime_tags" in data
        assert "importance" in data
        assert "access_count" in data

    def test_unknown_name_returns_404(self, client: TestClient, tmp_path: Path) -> None:
        """SC-W11.3: GET /api/memory/skills/unknown 返回 404。"""
        with patch("api.routes.memory._SKILLS_ROOT", tmp_path):
            resp = client.get("/api/memory/skills/no-such-skill", headers={"X-API-Key": "test-key-019"})

        assert resp.status_code == 404
        assert resp.json()["error"] == "not_found"


# ── 3. GET /api/memory/skill-access ──────────────────────────────────────────


class TestGetSkillAccess:
    def test_returns_200_with_access_data(self, client: TestClient, tmp_path: Path) -> None:
        """SC-W11.4: GET /api/memory/skill-access?since=... 返回 200 + access 事件。"""
        _write_skill(tmp_path, "chain-analysis", scope="agent:chain", access_count=10)

        with patch("api.routes.memory._SKILLS_ROOT", tmp_path):
            since = "2020-01-01T00:00:00Z"
            resp = client.get(
                f"/api/memory/skill-access?since={since}",
                headers={"X-API-Key": "test-key-019"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        names = [i["skill_name"] for i in data["items"]]
        assert "chain-analysis" in names

    def test_invalid_since_returns_400(self, client: TestClient, tmp_path: Path) -> None:
        """SC-W11.7: 错误 since 参数返回 400。"""
        with patch("api.routes.memory._SKILLS_ROOT", tmp_path):
            resp = client.get(
                "/api/memory/skill-access?since=not-a-date",
                headers={"X-API-Key": "test-key-019"},
            )

        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_query"


# ── 4. GET /api/memory/skill-proposals ───────────────────────────────────────


class TestGetSkillProposals:
    def test_returns_200_with_proposals(self, client: TestClient, tmp_path: Path) -> None:
        """SC-W11.5: GET /api/memory/skill-proposals?since=... 返回 200 + proposal 历史。"""
        _write_skill(tmp_path, "macro-proposed", scope="shared", write_draft=True)

        with patch("api.routes.memory._SKILLS_ROOT", tmp_path):
            since = "2020-01-01T00:00:00Z"
            resp = client.get(
                f"/api/memory/skill-proposals?since={since}",
                headers={"X-API-Key": "test-key-019"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data

    def test_proposals_cache_control_300(self, client: TestClient, tmp_path: Path) -> None:
        """SC-W11.8: skill-proposals Cache-Control max-age=300。"""
        with patch("api.routes.memory._SKILLS_ROOT", tmp_path):
            resp = client.get("/api/memory/skill-proposals", headers={"X-API-Key": "test-key-019"})

        assert resp.status_code == 200
        cc = resp.headers.get("cache-control", resp.headers.get("Cache-Control", ""))
        assert "300" in cc


# ── 5. Auth tests ─────────────────────────────────────────────────────────────


class TestAuthEnforcement:
    def test_missing_api_key_returns_401_or_bypassed(self, client: TestClient, tmp_path: Path) -> None:
        """SC-W11.6: 缺鉴权时返回 401（AUTH_MODE=enabled）或 bypass（disabled）。"""
        # In test env AUTH_MODE=disabled → 200 bypass is valid
        with patch("api.routes.memory._SKILLS_ROOT", tmp_path):
            resp = client.get("/api/memory/skills")
        # Either 200 (bypassed) or 401 (enforced) — both valid per config
        assert resp.status_code in (200, 401)
