"""spec 018/019 — Memory API Routes（US-Z5 / US6）。

FR-Z40: GET /api/memory/rules — 返回 pattern rules 摘要列表
FR-Z41: GET /api/memory/cases — 返回 cases 摘要列表（含 IVE）
FR-Z42: GET /api/memory/transitions — 返回 FSM 状态转换事件
FR-Z43: GET /api/memory/archived — 返回 archived patterns 列表
spec 019 FR-W17: GET /api/memory/skills — 返回 skill 列表
spec 019 FR-W18: GET /api/memory/skills/{name} — 返回 skill 详情
spec 019 FR-W19: GET /api/memory/skill-access — 返回 skill access 统计
spec 019 FR-W20: GET /api/memory/skill-proposals — 返回 draft proposals

Cache-Control: max-age=30 for rules/cases/transitions/skills/skill-access;
               max-age=300 for archived/skill-proposals.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["memory"])

# 有效 maturity 值
_VALID_MATURITIES = {"observed", "probationary", "active", "deprecated", "archived"}
# 有效 agent IDs
_VALID_AGENTS = {"tech", "chain", "news", "macro"}

# memory root（从 EvolvingMemoryProvider 默认值一致）
_MEMORY_ROOT = Path("agent_memory")


# ── Response models ───────────────────────────────────────────────────────────


class PnLTrackOut(BaseModel):
    successes: int
    losses: int
    total_pnl: float


class RuleItem(BaseModel):
    name: str
    agent: str
    description: str
    maturity: str
    importance: float
    access_count: int
    last_accessed_at: str | None
    pnl_track: PnLTrackOut
    regime_tags: list[str]
    fundamental_failure_streak: int
    version: int
    manually_edited: bool


class RulesList(BaseModel):
    items: list[RuleItem]
    total: int


class TradeExecutionOut(BaseModel):
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    actual_exit_price: float | None = None
    fill_status: str | None = None
    hit_sl: bool | None = None


class IVEClassificationOut(BaseModel):
    failure_type: str
    confidence: float
    reasoning: str


class CaseItem(BaseModel):
    cycle_id: str
    timestamp: str
    pair: str
    verdict_action: str
    final_pnl: float | None
    trade_execution: TradeExecutionOut | None
    ive_classification: IVEClassificationOut | None


class CasesList(BaseModel):
    items: list[CaseItem]
    total: int


class TransitionItem(BaseModel):
    rule_id: str
    agent_id: str
    old_state: str
    new_state: str
    triggered_by: str
    timestamp: str


class TransitionsList(BaseModel):
    items: list[TransitionItem]
    total: int


class ArchivedRuleItem(BaseModel):
    name: str
    agent: str
    archived_at: str | None
    fundamental_failure_streak: int
    final_pnl_track: PnLTrackOut


class ArchivedList(BaseModel):
    items: list[ArchivedRuleItem]
    total: int


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_provider() -> Any:
    """加载 EvolvingMemoryProvider 实例（新建只读实例）。"""
    from cryptotrader.learning.evolution.provider import EvolvingMemoryProvider

    return EvolvingMemoryProvider(memory_root=_MEMORY_ROOT)


def _pnl_track_out(pnl_track: Any) -> PnLTrackOut:
    """把 PnLTrack dataclass 转为 response model。"""
    wins = getattr(pnl_track, "wins", 0)
    cases = getattr(pnl_track, "cases", 0)
    losses = max(0, cases - wins)
    avg_pnl = getattr(pnl_track, "avg_pnl", 0.0)
    total_pnl = avg_pnl * cases
    return PnLTrackOut(successes=wins, losses=losses, total_pnl=round(total_pnl, 4))


def _dt_str(dt: Any) -> str | None:
    """把 datetime 转为 ISO 字符串，None-safe。"""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()
    return str(dt)


def _build_trade_execution_out(trade_ex: dict | None) -> TradeExecutionOut | None:
    """把 trade_execution dict 转为 response model。"""
    if not trade_ex:
        return None
    return TradeExecutionOut(
        entry_price=trade_ex.get("entry_price"),
        stop_loss=trade_ex.get("stop_loss"),
        take_profit=trade_ex.get("take_profit"),
        actual_exit_price=trade_ex.get("actual_exit_price"),
        fill_status=trade_ex.get("fill_status"),
        hit_sl=trade_ex.get("hit_sl"),
    )


def _build_ive_out(ive: dict | None) -> IVEClassificationOut | None:
    """把 ive_classification dict 转为 response model。"""
    if not ive:
        return None
    return IVEClassificationOut(
        failure_type=ive.get("failure_type", "noise"),
        confidence=float(ive.get("confidence", 0.0)),
        reasoning=ive.get("reasoning", ""),
    )


def _parse_aware_dt(raw: str | None, default: datetime) -> datetime | None:
    """解析 ISO 字符串为 timezone-aware datetime，失败返回 None（由调用方返回 400）。"""
    if raw is None:
        return default
    dt = datetime.fromisoformat(raw)  # raises ValueError if invalid
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _case_passes_filters(case: Any, from_dt: datetime, to_dt: datetime, agent: str | None) -> bool:
    """检查 case 是否在时间窗口内且符合 agent 过滤。"""
    ts = case.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    if not (from_dt <= ts <= to_dt):
        return False
    if agent is not None:
        refs = list(case.applied_patterns or [])
        if not any(ref.startswith(f"{agent}::") for ref in refs):
            return False
    return True


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/rules", response_model=RulesList)
async def get_memory_rules(
    agent: str | None = Query(default=None, description="tech/chain/news/macro"),
    status: str | None = Query(default=None, description="observed/probationary/active/deprecated/archived"),
) -> JSONResponse:
    """FR-Z40: 返回 pattern rules 摘要列表。"""
    # 参数验证
    if status is not None and status not in _VALID_MATURITIES:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": f"status must be one of {sorted(_VALID_MATURITIES)}"},
        )
    if agent is not None and agent not in _VALID_AGENTS:
        return JSONResponse(
            status_code=404,
            content={"error": "agent_not_found"},
        )

    try:
        from cryptotrader.learning.evolution.provider import _load_pattern_from_path

        agents_to_scan = [agent] if agent else list(_VALID_AGENTS)
        items: list[RuleItem] = []

        for agent_id in agents_to_scan:
            pattern_dir = _MEMORY_ROOT / agent_id / "patterns"
            if not pattern_dir.exists():
                continue
            for path in sorted(pattern_dir.glob("*.md")):
                rule = _load_pattern_from_path(path)
                if rule is None:
                    continue
                if status is not None and rule.maturity != status:
                    continue
                items.append(
                    RuleItem(
                        name=rule.name,
                        agent=rule.agent,
                        description=rule.description,
                        maturity=rule.maturity,
                        importance=rule.importance,
                        access_count=rule.access_count,
                        last_accessed_at=_dt_str(rule.last_accessed_at),
                        pnl_track=_pnl_track_out(rule.pnl_track),
                        regime_tags=list(rule.regime_tags or []),
                        fundamental_failure_streak=rule.fundamental_failure_streak,
                        version=rule.version,
                        manually_edited=rule.manually_edited,
                    )
                )

        response = RulesList(items=items, total=len(items))
        return JSONResponse(
            content=response.model_dump(),
            headers={"Cache-Control": "max-age=30"},
        )

    except Exception as exc:
        logger.warning("GET /api/memory/rules failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )


@router.get("/cases", response_model=CasesList)
async def get_memory_cases(
    from_: str | None = Query(default=None, alias="from", description="ISO8601 start time"),
    to: str | None = Query(default=None, description="ISO8601 end time"),
    agent: str | None = Query(default=None, description="filter by agent in analyses"),
) -> JSONResponse:
    """FR-Z41: 返回 cases 摘要列表（按 timestamp 倒序，含 IVE）。"""
    now = datetime.now(UTC)
    try:
        from_dt = _parse_aware_dt(from_, now - timedelta(days=7))
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": "Invalid 'from' datetime format"},
        )
    try:
        to_dt = _parse_aware_dt(to, now)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": "Invalid 'to' datetime format"},
        )

    try:
        from cryptotrader.learning.evolution.provider import _load_case_from_path

        cases_dir = _MEMORY_ROOT / "cases"
        items: list[CaseItem] = []

        if cases_dir.exists():
            for path in sorted(cases_dir.glob("*.md")):
                case = _load_case_from_path(path)
                if case is None:
                    continue
                if not _case_passes_filters(case, from_dt, to_dt, agent):
                    continue

                items.append(
                    CaseItem(
                        cycle_id=case.cycle_id,
                        timestamp=_dt_str(case.timestamp) or "",
                        pair=case.pair,
                        verdict_action=case.verdict_action,
                        final_pnl=case.final_pnl,
                        trade_execution=_build_trade_execution_out(case.trade_execution),
                        ive_classification=_build_ive_out(case.ive_classification),
                    )
                )

        # 按 timestamp 倒序
        items.sort(key=lambda c: c.timestamp, reverse=True)
        response = CasesList(items=items, total=len(items))
        return JSONResponse(
            content=response.model_dump(),
            headers={"Cache-Control": "max-age=30"},
        )

    except Exception as exc:
        logger.warning("GET /api/memory/cases failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )


@router.get("/transitions", response_model=TransitionsList)
async def get_memory_transitions(
    since: str | None = Query(default=None, description="ISO8601 start time; defaults to 24h ago"),
) -> JSONResponse:
    """FR-Z42: 返回 FSM 状态转换事件（从 provider 触发一次 evaluate_all_rules）。"""
    now = datetime.now(UTC)
    try:
        since_dt = datetime.fromisoformat(since) if since else now - timedelta(hours=24)
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=UTC)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": "Invalid 'since' datetime format"},
        )

    try:
        provider = _load_provider()
        transitions = provider.evaluate_all_rules()

        items: list[TransitionItem] = []
        for t in transitions:
            ts_raw = getattr(t, "timestamp", now)
            ts_str = _dt_str(ts_raw) or now.isoformat()
            # filter by since
            try:
                ts_dt = datetime.fromisoformat(ts_str)
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=UTC)
                if ts_dt < since_dt:
                    continue
            except (ValueError, TypeError):
                pass

            items.append(
                TransitionItem(
                    rule_id=f"{getattr(t, 'agent_id', '')}::{getattr(t, 'rule_id', '')}",
                    agent_id=getattr(t, "agent_id", ""),
                    old_state=getattr(t, "old_state", ""),
                    new_state=getattr(t, "new_state", ""),
                    triggered_by=getattr(t, "triggered_by", ""),
                    timestamp=ts_str,
                )
            )

        response = TransitionsList(items=items, total=len(items))
        return JSONResponse(
            content=response.model_dump(),
            headers={"Cache-Control": "max-age=30"},
        )

    except Exception as exc:
        logger.warning("GET /api/memory/transitions failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )


@router.get("/archived", response_model=ArchivedList)
async def get_memory_archived() -> JSONResponse:
    """FR-Z43: 返回 archived patterns 列表（从 .archived/ 子目录读取）。"""
    try:
        from cryptotrader.learning.evolution.provider import _load_pattern_from_path

        items: list[ArchivedRuleItem] = []

        for agent_id in _VALID_AGENTS:
            archived_dir = _MEMORY_ROOT / agent_id / "patterns" / ".archived"
            if not archived_dir.exists():
                continue
            for path in sorted(archived_dir.glob("*.md")):
                rule = _load_pattern_from_path(path)
                if rule is None:
                    continue
                archived_at = _dt_str(rule.last_modified_at)
                items.append(
                    ArchivedRuleItem(
                        name=rule.name,
                        agent=rule.agent,
                        archived_at=archived_at,
                        fundamental_failure_streak=rule.fundamental_failure_streak,
                        final_pnl_track=_pnl_track_out(rule.pnl_track),
                    )
                )

        response = ArchivedList(items=items, total=len(items))
        return JSONResponse(
            content=response.model_dump(),
            headers={"Cache-Control": "max-age=300"},
        )

    except Exception as exc:
        logger.warning("GET /api/memory/archived failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )


# ── spec 019 Skill endpoints ──────────────────────────────────────────────────

_SKILLS_ROOT = Path("agent_skills")


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
        # scope filter: "agent:tech" → agent="tech"
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
    """spec 019 FR-W17: 返回 skill 列表（按 scope 过滤）。"""
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
    """spec 019 FR-W18: 返回 skill 详情（含 body）。"""
    # 路径穿越防护：skill name 不允许包含路径分隔符或 ".."
    if not name or "/" in name or "\\" in name or ".." in name:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": "invalid skill name"},
        )
    try:
        from cryptotrader.learning.evolution.skill_provider import _load_skill_from_path

        skill_md = _SKILLS_ROOT / name / "SKILL.md"
        # 二次校验：确保解析后路径仍在 skills root 内（防止符号链接绕过）
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
    """spec 019 FR-W19: 返回 skill access 统计（从 SKILL.md frontmatter 读 access_count + last_accessed_at）。"""
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
    """spec 019 FR-W20: 返回 skill proposals（.draft 文件列表）。"""
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
