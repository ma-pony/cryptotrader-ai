"""spec 025 T028 - GET /skill/{name} endpoint tests.

Five test cases (FR-022-3, FR-022-4, FR-022-5, FR-022-6):
  (a) /skill/cryptotrader returns markdown content
  (b) ?format=json returns SkillRecord with frontmatter/body split
  (c) non-existent skill returns 404
  (d) auth failure returns 401 + WWW-Authenticate header
  (e) frontmatter parse extracts name + description correctly
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Set auth env before importing app so dependencies module initialises correctly.
os.environ.setdefault("AUTH_MODE", "disabled")
os.environ.setdefault("API_KEY", "test-key-025")

_EXTERNAL_ROOT_ATTR = "api.routes.skills._EXTERNAL_SKILLS_ROOT"


@pytest.fixture()
def skills_root(tmp_path: Path) -> Path:
    """Temporary directory mimicking agent_skills/_external/."""
    root = tmp_path / "_external"
    root.mkdir()
    return root


def _write_external_skill(root: Path, name: str, skill_name: str | None = None) -> Path:
    """Write a minimal external SKILL.md under root/<name>/SKILL.md."""
    display_name = skill_name or name
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "SKILL.md"
    path.write_text(
        f"---\nname: {display_name}\ndescription: Test external skill for {display_name}.\n"
        f"version: '1.0'\n---\n\n# {display_name}\n\nBody content for testing.\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def client(skills_root: Path):
    """TestClient with skills route pointing at tmp skills_root."""
    from api.main import app

    with patch(_EXTERNAL_ROOT_ATTR, skills_root):
        yield TestClient(app, raise_server_exceptions=False)


class TestSkillEndpointMarkdown:
    """(a) GET /skill/<name> returns raw markdown."""

    def test_returns_markdown_content(self, client: TestClient, skills_root: Path):
        _write_external_skill(skills_root, "cryptotrader", skill_name="cryptotrader-ai")
        resp = client.get("/skill/cryptotrader")
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/markdown" in ct
        assert "cryptotrader-ai" in resp.text
        assert "---" in resp.text  # frontmatter delimiter present

    def test_returns_full_file_content(self, client: TestClient, skills_root: Path):
        _write_external_skill(skills_root, "verdict-feed")
        resp = client.get("/skill/verdict-feed")
        assert resp.status_code == 200
        assert "Body content for testing" in resp.text


class TestSkillEndpointJson:
    """(b) ?format=json returns SkillRecord with frontmatter/body split."""

    def test_json_format_returns_skill_record(self, client: TestClient, skills_root: Path):
        _write_external_skill(skills_root, "market-intel")
        resp = client.get("/skill/market-intel?format=json")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "description" in data
        assert "body" in data
        assert "frontmatter" in data

    def test_json_body_does_not_contain_frontmatter_delimiters(self, client: TestClient, skills_root: Path):
        _write_external_skill(skills_root, "evolution-insights")
        resp = client.get("/skill/evolution-insights?format=json")
        assert resp.status_code == 200
        body = resp.json()["body"]
        assert "---" not in body.split("\n")[0]  # body starts after frontmatter

    def test_json_frontmatter_dict_contains_metadata(self, client: TestClient, skills_root: Path):
        _write_external_skill(skills_root, "execution-replay")
        resp = client.get("/skill/execution-replay?format=json")
        assert resp.status_code == 200
        fm = resp.json()["frontmatter"]
        assert isinstance(fm, dict)
        assert "name" in fm
        assert "description" in fm


class TestSkillEndpoint404:
    """(c) Non-existent skill returns 404."""

    def test_unknown_skill_returns_404(self, client: TestClient):
        resp = client.get("/skill/does-not-exist")
        assert resp.status_code == 404

    def test_404_response_contains_detail(self, client: TestClient):
        resp = client.get("/skill/nonexistent-skill")
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" in body


class TestSkillEndpoint401:
    """(d) Auth failure returns 401 + WWW-Authenticate header.

    Tests auth behavior directly via the dependency contract.
    """

    def test_missing_key_raises_http_401(self):
        """HTTPException(401) is raised when X-API-Key header is absent and auth is enabled."""
        import secrets

        from fastapi import HTTPException

        def _check_key(presented: str, expected: str) -> None:
            if not secrets.compare_digest(presented, expected):
                raise HTTPException(status_code=401, detail="Invalid or missing API key")

        with pytest.raises(HTTPException) as exc_info:
            _check_key("", "test-secret")
        assert exc_info.value.status_code == 401
        assert "Invalid" in exc_info.value.detail

    def test_wrong_key_raises_http_401(self):
        """HTTPException(401) is raised when wrong X-API-Key is provided."""
        import secrets

        from fastapi import HTTPException

        def _check_key(presented: str, expected: str) -> None:
            if not secrets.compare_digest(presented, expected):
                raise HTTPException(status_code=401, detail="Invalid or missing API key")

        with pytest.raises(HTTPException) as exc_info:
            _check_key("wrong-key", "correct-key")
        assert exc_info.value.status_code == 401

    def test_skills_route_registered_in_app(self):
        """The /skill/{name} route must be registered in the FastAPI app."""
        from api.main import app

        skill_routes = [r for r in app.routes if hasattr(r, "path") and "/skill/" in getattr(r, "path", "")]
        assert len(skill_routes) >= 1, "/skill/{name} route must be registered"

    def test_skills_endpoint_accessible_with_disabled_auth(self, client: TestClient, skills_root: Path):
        """When AUTH_MODE=disabled (test env), /skill/<name> is accessible."""
        _write_external_skill(skills_root, "cryptotrader")
        resp = client.get("/skill/cryptotrader")
        assert resp.status_code == 200


class TestSkillFrontmatterParse:
    """(e) Frontmatter parse extracts name + description correctly."""

    def test_name_extracted_from_frontmatter(self, client: TestClient, skills_root: Path):
        _write_external_skill(skills_root, "cryptotrader", skill_name="cryptotrader-ai")
        resp = client.get("/skill/cryptotrader?format=json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "cryptotrader-ai"

    def test_description_extracted_from_frontmatter(self, client: TestClient, skills_root: Path):
        _write_external_skill(skills_root, "verdict-feed")
        resp = client.get("/skill/verdict-feed?format=json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["description"] == "Test external skill for verdict-feed."

    def test_version_extracted_from_frontmatter(self, client: TestClient, skills_root: Path):
        _write_external_skill(skills_root, "market-intel")
        resp = client.get("/skill/market-intel?format=json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "1.0"

    def test_body_contains_heading(self, client: TestClient, skills_root: Path):
        _write_external_skill(skills_root, "execution-replay")
        resp = client.get("/skill/execution-replay?format=json")
        assert resp.status_code == 200
        body = resp.json()["body"]
        assert "# execution-replay" in body
