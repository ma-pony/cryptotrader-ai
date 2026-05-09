"""EvolvingSkillProvider — spec 019 D-RT-01 两层检索算法（FR-W7..W10）。

实现 spec 017a SkillProvider Protocol。
与 EvolvingMemoryProvider 并存于 nodes/agents.py module-level singleton。
"""

from __future__ import annotations

import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── regime 推断阈值（从 snapshot 中的关键字段）──────────────────────────────────

_FUNDING_HIGH_THRESHOLD = 0.0003  # > 0.03%
_FUNDING_NEG_THRESHOLD = -0.0001  # < -0.01%
_FEAR_GREED_EXTREME_FEAR = 25
_FEAR_GREED_EXTREME_GREED = 75
_VOL_HIGH_THRESHOLD = 0.04  # 4% daily vol (rough threshold)


def _extract_regime(snapshot: dict) -> str | None:
    """从 snapshot 推断当前 regime 标签（最高优先级规则）。

    返回 spec 014 taxonomy 中的一个 regime 标签，或 None（无法推断）。
    snapshot 结构较松散，尽力推断。
    """
    # funding rate
    fr = None
    if isinstance(snapshot, dict):
        fr = snapshot.get("funding_rate")
        if fr is None:
            # 尝试嵌套结构
            market = snapshot.get("market") or {}
            if isinstance(market, dict):
                fr = market.get("funding_rate")

    if fr is not None:
        try:
            fr_f = float(fr)
            if fr_f > _FUNDING_HIGH_THRESHOLD:
                return "high_funding"
            if fr_f < _FUNDING_NEG_THRESHOLD:
                return "negative_funding"
        except (TypeError, ValueError):
            pass

    # fear & greed
    fg = None
    if isinstance(snapshot, dict):
        fg = snapshot.get("fear_greed_index") or snapshot.get("fear_greed")
        if fg is None:
            market = snapshot.get("market") or {}
            if isinstance(market, dict):
                fg = market.get("fear_greed_index") or market.get("fear_greed")

    if fg is not None:
        try:
            fg_f = float(fg)
            if fg_f <= _FEAR_GREED_EXTREME_FEAR:
                return "extreme_fear"
            if fg_f >= _FEAR_GREED_EXTREME_GREED:
                return "extreme_greed"
        except (TypeError, ValueError):
            pass

    return None


def _load_skill_from_path(path: Path) -> Any | None:
    """从 SKILL.md 文件加载 Skill 对象（含 spec 019 新字段）。"""
    try:
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter
        from cryptotrader.agents.skills.schema import Skill

        content = path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content, path=path)

        # 解析 last_accessed_at
        raw_la = fm.get("last_accessed_at")
        if raw_la is not None:
            try:
                la = datetime.fromisoformat(str(raw_la))
                if la.tzinfo is None:
                    la = la.replace(tzinfo=UTC)
            except (ValueError, TypeError):
                la = datetime.now(UTC)
        else:
            # 用文件 mtime 作为默认值
            try:
                la = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            except OSError:
                la = datetime.now(UTC)

        return Skill(
            name=str(fm.get("name", path.parent.name)),
            description=str(fm.get("description", "")),
            scope=str(fm.get("scope", "shared")),
            body=body,
            file_path=path,
            manually_edited=bool(fm.get("manually_edited", False)),
            version=str(fm.get("version", "1.0")),
            mtime=path.stat().st_mtime,
            # spec 019 新字段
            regime_tags=list(fm.get("regime_tags") or []),
            triggers_keywords=list(fm.get("triggers_keywords") or []),
            importance=float(fm.get("importance", 0.5)),
            access_count=int(fm.get("access_count", 0)),
            last_accessed_at=la,
            confidence=float(fm.get("confidence", 0.5)),
        )
    except Exception:
        logger.warning("_load_skill_from_path failed for %s", path, exc_info=True)
        return None


def _write_back_access(skill: Any, now: datetime) -> None:
    """回写 access_count + last_accessed_at 到 SKILL.md frontmatter（原子写）。"""
    try:
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter, render_frontmatter
        from cryptotrader.agents.skills._io import atomic_write

        path = skill.file_path
        content = path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content, path=path)

        fm["access_count"] = int(fm.get("access_count", 0)) + 1
        fm["last_accessed_at"] = now.isoformat()

        new_content = render_frontmatter(fm) + body
        atomic_write(path, new_content)

        # 更新内存中的 skill 对象
        skill.access_count += 1
        skill.last_accessed_at = now
    except Exception:
        logger.warning("_write_back_access failed for skill '%s'", getattr(skill, "name", "?"), exc_info=True)


def _emit_telemetry(
    candidates: list[Any],
    filtered_out: list[dict],
    top_k_with_scores: list[dict],
    duration_ms: float,
) -> None:
    """写 4 个 OpenTelemetry span attributes（FR-W28）。"""
    attrs = {
        "skill.retrieval.candidates_after_regime_filter": str([s.name for s in candidates]),
        "skill.retrieval.filtered_out": str(filtered_out),
        "skill.retrieval.top_k_with_scores": str(top_k_with_scores),
        "skill.retrieval.duration_ms": round(duration_ms, 2),
    }
    span_attached = False
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is not None and span.is_recording():
            for key, val in attrs.items():
                if isinstance(val, (list, dict)):
                    span.set_attribute(key, str(val))
                else:
                    span.set_attribute(key, val)
            span_attached = True
    except Exception:
        pass

    if not span_attached:
        logger.info(
            "skill_retrieval candidates=%s filtered_out=%s top_k=%s duration_ms=%.2f",
            [s.name for s in candidates],
            filtered_out,
            top_k_with_scores,
            duration_ms,
        )


class EvolvingSkillProvider:
    """EvolvingSkillProvider — 实现 spec 017a SkillProvider Protocol。

    D-RT-01 两层检索算法：
      第一层：scope + regime_tags 预过滤
      第二层：score = (idf_score + importance + recency_bonus) x confidence
    """

    def __init__(
        self,
        skill_root: Path = Path("agent_skills"),
        top_k: int = 5,
        # spec 019 backward-compat alias: old DefaultSkillProvider used skills_root=
        skills_root: Path | None = None,
    ) -> None:
        self._skill_root = skills_root if skills_root is not None else skill_root
        self._top_k = top_k

    def get_available_skills(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> list:
        """D-RT-01 两层检索，返回 top-k Skill list（FR-W7/W8/W9）。

        任一步骤异常 -> catch + log warning + 返回 []（FR-W9）。
        """
        try:
            return self._get_available_skills_impl(agent_id, snapshot, k)
        except Exception:
            logger.warning(
                "EvolvingSkillProvider.get_available_skills failed for agent '%s'",
                agent_id,
                exc_info=True,
            )
            return []

    def _get_available_skills_impl(
        self,
        agent_id: str,
        snapshot: dict,
        k: int,
    ) -> list:
        """内部实现，异常由外层 catch。"""
        from cryptotrader.agents.skills.loader import discover_skills_for_agent
        from cryptotrader.learning.evolution.idf import compute_idf, extract_query_keywords, score_skill

        t0 = time.monotonic()
        effective_k = k if k > 0 else self._top_k

        # 第一层：scope filter（reuse spec 014）
        scope_candidates = discover_skills_for_agent(agent_id, skill_dir=self._skill_root)

        # 用新字段重新加载（discover_skills_for_agent 返回的 Skill 可能缺 spec 019 字段）
        candidates = []
        filtered_out = []
        for s in scope_candidates:
            full = _load_skill_from_path(s.file_path)
            if full is None:
                continue
            candidates.append(full)

        # regime_tags 预过滤
        current_regime = _extract_regime(snapshot)
        regime_filtered = []
        for s in candidates:
            if not s.regime_tags:
                # 空 list -> match all（向后兼容）
                regime_filtered.append(s)
            elif current_regime and current_regime in s.regime_tags:
                regime_filtered.append(s)
            else:
                filtered_out.append({"name": s.name, "reason": "regime_tags mismatch"})

        # 第二层：IDF + importance + recency 加权排序
        corpus = [s.triggers_keywords for s in regime_filtered]
        idf_table = compute_idf(corpus)
        query_keywords = extract_query_keywords(snapshot)
        now = datetime.now(UTC)

        scored = []
        for s in regime_filtered:
            idf_s = score_skill(s.triggers_keywords, query_keywords, idf_table)
            recency_bonus = math.exp(-max(0.0, (now - s.last_accessed_at).total_seconds()) / (7 * 86400))
            total = (idf_s + s.importance + recency_bonus) * s.confidence
            scored.append((s, total, idf_s, s.importance, recency_bonus))

        # 按 score 倒序，取 top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k_items = scored[:effective_k]

        top_k_with_scores = [
            {
                "name": item[0].name,
                "score": round(item[1], 4),
                "idf_component": round(item[2], 4),
                "importance_component": round(item[3], 4),
                "recency_component": round(item[4], 4),
            }
            for item in top_k_items
        ]

        result_skills = [item[0] for item in top_k_items]

        # 回写 access_count + last_accessed_at
        for s in result_skills:
            _write_back_access(s, now)

        duration_ms = (time.monotonic() - t0) * 1000
        _emit_telemetry(regime_filtered, filtered_out, top_k_with_scores, duration_ms)

        return result_skills

    def get_skill_by_name(self, name: str) -> Any | None:
        """扫所有 SKILL.md 找 name 匹配，回写 access_count（FR-W10）。

        任一步骤异常 -> catch + log warning + 返回 None。
        """
        try:
            return self._get_skill_by_name_impl(name)
        except Exception:
            logger.warning(
                "EvolvingSkillProvider.get_skill_by_name failed for '%s'",
                name,
                exc_info=True,
            )
            return None

    def _get_skill_by_name_impl(self, name: str) -> Any | None:
        """内部实现，异常由外层 catch。"""
        if not self._skill_root.exists():
            return None

        for skill_dir in sorted(self._skill_root.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith("."):
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            skill = _load_skill_from_path(skill_md)
            if skill is None:
                continue

            if skill.name == name:
                now = datetime.now(UTC)
                _write_back_access(skill, now)
                return skill

        return None
