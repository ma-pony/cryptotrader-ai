"""Memory / Skill API Routes — skill side only after 2026-05-13 evolution-memory removal.

GET /api/memory/skills          — skill 列表
GET /api/memory/skills/{name}   — skill 详情
GET /api/memory/skill-access    — skill access 统计
GET /api/memory/skill-proposals — draft proposals
GET /api/memory/patterns        — PatternRecord 列表（spec 024 SC-P3 补完）
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cryptotrader._compat import UTC

logger = logging.getLogger(__name__)

router = APIRouter(tags=["memory"])

_VALID_AGENTS = {"tech", "chain", "news", "macro"}
_SKILLS_ROOT = Path("agent_skills")
_MEMORY_ROOT = Path("agent_memory")


# ── PatternRecord API schema (spec 024 SC-P3) ─────────────────────────────────


class PnLTrackResponse(BaseModel):
    pnls: list[float]


class PatternRecordResponse(BaseModel):
    """PatternRecord API wrapper (spec 024 / FR-022-17~20).

    直接映射 cryptotrader.learning.memory.PatternRecord 字段。
    """

    name: str
    agent: str
    description: str
    maturity: str
    regime_tags: list[str]
    pnl_track: PnLTrackResponse
    source_cycles: list[str]
    body: str
    version: int
    manually_edited: bool
    created: str | None


class PatternsList(BaseModel):
    items: list[PatternRecordResponse]
    total: int


# ── Skill API schema ───────────────────────────────────────────────────────────


class SkillItem(BaseModel):
    name: str
    scope: str
    version: str
    regime_tags: list[str]
    triggers_keywords: list[str]
    importance: float
    confidence: float
    access_count: int
    last_accessed_at: str | None
    manually_edited: bool
    description: str


class SkillDetail(SkillItem):
    body: str


class SkillsList(BaseModel):
    items: list[SkillItem]
    total: int


class SkillAccessItem(BaseModel):
    skill_name: str
    scope: str
    access_count: int
    last_accessed_at: str | None


class SkillAccessList(BaseModel):
    items: list[SkillAccessItem]
    total: int


class SkillProposalMetadata(BaseModel):
    regime_tags: list[str]
    triggers_keywords: list[str]
    importance: float
    confidence: float


class SkillProposalItem(BaseModel):
    name: str
    draft_path: str
    created_at: str
    llm_inferred_metadata: SkillProposalMetadata
    llm_call_failed: bool
    user_saved: bool


class SkillProposalsList(BaseModel):
    items: list[SkillProposalItem]
    total: int


def _load_all_skills(agent: str | None = None) -> list[Any]:
    """扫 agent_skills/ 目录，加载所有 SKILL.md。agent 过滤时按 scope 匹配。"""
    from cryptotrader.learning.evolution.skill_provider import _load_skill_from_path

    skills = []
    if not _SKILLS_ROOT.exists():
        return []
    for skill_dir in sorted(_SKILLS_ROOT.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        skill = _load_skill_from_path(skill_md)
        if skill is None:
            continue
        if agent is not None:
            expected_scope = f"agent:{agent}"
            if skill.scope != expected_scope and skill.scope != "shared":
                continue
        skills.append(skill)
    return skills


def _skill_to_item(skill: Any) -> SkillItem:
    la = skill.last_accessed_at.isoformat() if skill.last_accessed_at else None
    return SkillItem(
        name=skill.name,
        scope=skill.scope,
        version=skill.version,
        regime_tags=skill.regime_tags,
        triggers_keywords=skill.triggers_keywords,
        importance=skill.importance,
        confidence=skill.confidence,
        access_count=skill.access_count,
        last_accessed_at=la,
        manually_edited=skill.manually_edited,
        description=skill.description,
    )


@router.get("/skills", response_model=SkillsList)
async def get_memory_skills(
    agent: str | None = Query(default=None, description="tech / chain / news / macro"),
) -> JSONResponse:
    if agent is not None and agent not in _VALID_AGENTS:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": f"agent must be one of {sorted(_VALID_AGENTS)}"},
        )
    try:
        skills = _load_all_skills(agent)
        items = [_skill_to_item(s) for s in skills]
        response = SkillsList(items=items, total=len(items))
        return JSONResponse(
            content=response.model_dump(),
            headers={"Cache-Control": "max-age=30"},
        )
    except Exception as exc:
        logger.warning("GET /api/memory/skills failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )


@router.get("/skills/{name}", response_model=SkillDetail)
async def get_memory_skill_detail(name: str) -> JSONResponse:
    if not name or "/" in name or "\\" in name or ".." in name:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": "invalid skill name"},
        )
    try:
        from cryptotrader.learning.evolution.skill_provider import _load_skill_from_path

        skill_md = _SKILLS_ROOT / name / "SKILL.md"
        try:
            skill_md.resolve().relative_to(_SKILLS_ROOT.resolve())  # noqa: ASYNC240
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_query", "detail": "invalid skill name"},
            )
        if not skill_md.exists():
            return JSONResponse(
                status_code=404,
                content={"error": "not_found", "detail": f"skill '{name}' not found"},
            )
        skill = _load_skill_from_path(skill_md)
        if skill is None:
            return JSONResponse(
                status_code=404,
                content={"error": "not_found", "detail": f"skill '{name}' could not be loaded"},
            )
        la = skill.last_accessed_at.isoformat() if skill.last_accessed_at else None
        detail = SkillDetail(
            name=skill.name,
            scope=skill.scope,
            version=skill.version,
            regime_tags=skill.regime_tags,
            triggers_keywords=skill.triggers_keywords,
            importance=skill.importance,
            confidence=skill.confidence,
            access_count=skill.access_count,
            last_accessed_at=la,
            manually_edited=skill.manually_edited,
            description=skill.description,
            body=skill.body,
        )
        return JSONResponse(
            content=detail.model_dump(),
            headers={"Cache-Control": "max-age=30"},
        )
    except Exception as exc:
        logger.warning("GET /api/memory/skills/%s failed", name, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )


@router.get("/skill-access", response_model=SkillAccessList)
async def get_skill_access(
    since: str | None = Query(default=None, description="ISO8601 起始时间；缺省 24h 前"),
    agent: str | None = Query(default=None, description="过滤特定 agent"),
) -> JSONResponse:
    if agent is not None and agent not in _VALID_AGENTS:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": f"agent must be one of {sorted(_VALID_AGENTS)}"},
        )
    try:
        since_dt: datetime
        if since is not None:
            try:
                since_dt = datetime.fromisoformat(since)
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=UTC)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "invalid_query", "detail": f"since must be ISO8601, got: {since!r}"},
                )
        else:
            since_dt = datetime.now(UTC) - timedelta(hours=24)

        skills = _load_all_skills(agent)
        items = []
        for s in skills:
            la = s.last_accessed_at
            if la is not None and la.tzinfo is None:
                la = la.replace(tzinfo=UTC)
            if la is None or la >= since_dt:
                items.append(
                    SkillAccessItem(
                        skill_name=s.name,
                        scope=s.scope,
                        access_count=s.access_count,
                        last_accessed_at=la.isoformat() if la else None,
                    )
                )
        response = SkillAccessList(items=items, total=len(items))
        return JSONResponse(
            content=response.model_dump(),
            headers={"Cache-Control": "max-age=30"},
        )
    except Exception as exc:
        logger.warning("GET /api/memory/skill-access failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )


@router.get("/skill-proposals", response_model=SkillProposalsList)
async def get_skill_proposals(
    since: str | None = Query(default=None, description="ISO8601 起始时间；缺省 7 天前"),
) -> JSONResponse:
    try:
        since_dt: datetime
        if since is not None:
            try:
                since_dt = datetime.fromisoformat(since)
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=UTC)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "invalid_query", "detail": f"since must be ISO8601, got: {since!r}"},
                )
        else:
            since_dt = datetime.now(UTC) - timedelta(days=7)

        from cryptotrader.agents.skills._frontmatter import parse_frontmatter

        items = []
        if _SKILLS_ROOT.exists():  # noqa: ASYNC240
            for skill_dir in sorted(_SKILLS_ROOT.iterdir()):  # noqa: ASYNC240
                if not skill_dir.is_dir() or skill_dir.name.startswith("."):
                    continue
                draft_path = skill_dir / "SKILL.md.draft"
                if not draft_path.exists():
                    continue
                try:
                    mtime = draft_path.stat().st_mtime
                    created_at_dt = datetime.fromtimestamp(mtime, tz=UTC)
                    if created_at_dt < since_dt:
                        continue
                    content = draft_path.read_text(encoding="utf-8")
                    fm, _ = parse_frontmatter(content, path=draft_path)
                    llm_failed = bool(fm.get("llm_call_failed", False))
                    proposal_meta = SkillProposalMetadata(
                        regime_tags=list(fm.get("regime_tags") or []),
                        triggers_keywords=list(fm.get("triggers_keywords") or []),
                        importance=float(fm.get("importance", 0.5)),
                        confidence=float(fm.get("confidence", 0.5)),
                    )
                    user_saved = (skill_dir / "SKILL.md").exists()
                    items.append(
                        SkillProposalItem(
                            name=skill_dir.name,
                            draft_path=str(draft_path),
                            created_at=created_at_dt.isoformat(),
                            llm_inferred_metadata=proposal_meta,
                            llm_call_failed=llm_failed,
                            user_saved=user_saved,
                        )
                    )
                except Exception:
                    logger.warning("Failed to load draft %s", draft_path, exc_info=True)

        response = SkillProposalsList(items=items, total=len(items))
        return JSONResponse(
            content=response.model_dump(),
            headers={"Cache-Control": "max-age=300"},
        )
    except Exception as exc:
        logger.warning("GET /api/memory/skill-proposals failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )


# ── GET /api/memory/patterns ───────────────────────────────────────────────────


def _load_all_patterns(
    memory_root: Path,
    agent: str | None = None,
    maturity: str | None = None,
) -> list[PatternRecordResponse]:
    """扫 agent_memory/{tech,chain,news,macro}/patterns/*.md，返回 PatternRecordResponse 列表。

    跳过 .gitkeep / .lock 文件；单文件解析失败 log warning + 继续。
    agent / maturity 过滤在加载后应用。
    """
    from cryptotrader.learning.memory import _load_pattern_record

    agents_to_scan = [agent] if agent else sorted(_VALID_AGENTS)
    results: list[PatternRecordResponse] = []

    for ag in agents_to_scan:
        patterns_dir = memory_root / ag / "patterns"
        if not patterns_dir.exists():
            continue
        for md_path in sorted(patterns_dir.iterdir()):
            if md_path.name in {".gitkeep", ".lock"} or md_path.suffix != ".md":
                continue
            record = _load_pattern_record(md_path)
            if record is None:
                continue
            if maturity is not None and record.maturity != maturity:
                continue
            results.append(
                PatternRecordResponse(
                    name=record.name,
                    agent=record.agent,
                    description=record.description,
                    maturity=record.maturity,
                    regime_tags=record.regime_tags,
                    pnl_track=PnLTrackResponse(pnls=record.pnl_track.pnls),
                    source_cycles=record.source_cycles,
                    body=record.body,
                    version=record.version,
                    manually_edited=record.manually_edited,
                    created=record.created,
                )
            )
    return results


@router.get("/patterns", response_model=PatternsList)
async def get_memory_patterns(
    agent: str | None = Query(default=None, description="tech / chain / news / macro"),
    maturity: str | None = Query(default=None, description="observed / probationary / active / deprecated / archived"),
    limit: int | None = Query(default=None, description="最多返回条数；缺省不限制"),
) -> JSONResponse:
    """GET /api/memory/patterns — 返回 trilogy 进化系统产出的 PatternRecord 列表。

    - 扫 agent_memory/{tech,chain,news,macro}/patterns/*.md
    - 支持 ?agent=&maturity=&limit= query filter
    - 目录为空或不存在时返回 items=[], total=0（不 500）

    spec 024 SC-P3 补完；FR-022-17/18/19/20。
    """
    if agent is not None and agent not in _VALID_AGENTS:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": f"agent must be one of {sorted(_VALID_AGENTS)}"},
        )

    valid_maturity = {"observed", "probationary", "active", "deprecated", "archived"}
    if maturity is not None and maturity not in valid_maturity:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": f"maturity must be one of {sorted(valid_maturity)}"},
        )

    if limit is not None and limit < 0:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": "limit must be >= 0"},
        )

    try:
        all_items = _load_all_patterns(_MEMORY_ROOT, agent=agent, maturity=maturity)
        total_count = len(all_items)
        items = all_items[:limit] if limit is not None else all_items
        response = PatternsList(items=items, total=total_count)
        return JSONResponse(
            content=response.model_dump(),
            headers={"Cache-Control": "max-age=30"},
        )
    except Exception as exc:
        logger.warning("GET /api/memory/patterns failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )
