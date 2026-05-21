"""External Skill Protocol endpoint — GET /skill/{name}.

spec 025 FR-022-1 through FR-022-6:
- Serves agent_skills/_external/<name>/SKILL.md
- format=markdown  -> raw markdown, Content-Type: text/markdown
- format=json      -> SkillRecord JSON (frontmatter split from body)
- 404 for unknown skill names
- 401 + WWW-Authenticate when API key is invalid
- ExternalSkillFetchAggregator metrics (graceful degrade if import fails)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["skills"])

# Root for external skill docs (relative to repo root at runtime).
# Resolved against CWD; production starts from repo root via uvicorn/docker.
_EXTERNAL_SKILLS_ROOT = Path("agent_skills/_external")


# ── Pydantic schema ──────────────────────────────────────────────────────────


class SkillFrontmatter(BaseModel):
    """Parsed YAML frontmatter from a SKILL.md file."""

    name: str
    description: str
    version: str = "1.0"


class SkillRecord(BaseModel):
    """External skill record returned by GET /skill/{name}?format=json."""

    name: str
    description: str
    version: str
    body: str
    frontmatter: dict


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_skill_md(path: Path) -> tuple[dict, str]:
    """Split YAML frontmatter and markdown body from a SKILL.md file.

    Returns (frontmatter_dict, body_str).
    Raises ValueError if frontmatter delimiters are missing.
    """
    import yaml

    content = path.read_text(encoding="utf-8")
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Missing YAML frontmatter delimiters in {path}")
    fm = yaml.safe_load(parts[1]) or {}
    body = parts[2].lstrip("\n")
    return fm, body


def _record_fetch(skill_name: str, request: Request, status: int) -> None:
    """Record skill fetch metric via ExternalSkillFetchAggregator.

    Gracefully degrades if the aggregator is not yet available (other team
    branch not merged). ImportError and AttributeError are both suppressed.
    """
    try:
        from cryptotrader.observability.heartbeat_metrics import (  # type: ignore[import-not-found]
            get_external_skill_fetch_aggregator,
        )

        client_id = request.headers.get("X-Agent-ID", "unknown")
        get_external_skill_fetch_aggregator().record(skill_name, client_id, status)
    except (ImportError, AttributeError):
        pass
    except Exception:
        logger.debug("ExternalSkillFetchAggregator.record failed (non-fatal)", exc_info=True)


# ── Route ────────────────────────────────────────────────────────────────────


@router.get("/skill/{name}")
async def get_skill(
    name: str,
    request: Request,
    format: Annotated[Literal["markdown", "json"], Query()] = "markdown",
) -> Response:
    """Serve an external SKILL.md by name.

    - format=markdown (default): raw Markdown with Content-Type: text/markdown
    - format=json: SkillRecord JSON with frontmatter/body split
    - 404 if skill name does not exist under agent_skills/_external/
    """
    # Path traversal guard: reject names containing path separators or dotdot components
    # before touching the filesystem.
    if not name or "/" in name or "\\" in name or ".." in name:
        _record_fetch(name, request, 400)
        raise HTTPException(status_code=400, detail="Invalid skill name")

    skill_path = _EXTERNAL_SKILLS_ROOT / name / "SKILL.md"

    # Secondary guard: resolve() confirms the path stays inside _EXTERNAL_SKILLS_ROOT.
    try:
        skill_path.resolve().relative_to(_EXTERNAL_SKILLS_ROOT.resolve())  # noqa: ASYNC240
    except ValueError:
        _record_fetch(name, request, 400)
        raise HTTPException(status_code=400, detail="Invalid skill name") from None

    if not skill_path.exists():
        _record_fetch(name, request, 404)
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

    try:
        fm, body = _parse_skill_md(skill_path)
    except Exception as exc:
        logger.warning("Failed to parse skill '%s': %s", name, exc)
        _record_fetch(name, request, 500)
        raise HTTPException(status_code=500, detail="Failed to parse skill file") from exc

    if format == "markdown":
        raw = skill_path.read_text(encoding="utf-8")
        _record_fetch(name, request, 200)
        return Response(content=raw, media_type="text/markdown; charset=utf-8")

    # format == "json"
    record = SkillRecord(
        name=str(fm.get("name", name)),
        description=str(fm.get("description", "")),
        version=str(fm.get("version", "1.0")),
        body=body,
        frontmatter=fm,
    )
    _record_fetch(name, request, 200)
    from fastapi.responses import JSONResponse

    return JSONResponse(content=record.model_dump())
