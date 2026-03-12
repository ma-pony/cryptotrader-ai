"""Agent self-reflection — structured experience memory generation.

Each agent periodically reviews its own historical analyses + actual PnL results
to generate structured ExperienceMemory (success patterns, forbidden zones,
strategic insights). Supports incremental evolution and anti-overfitting verification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.config import ExperienceConfig
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.models import DecisionCommit

from cryptotrader.models import ExperienceMemory, ExperienceRule

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path.home() / ".cryptotrader" / "agent_reflections.db"

_AGENT_IDS = ("tech_agent", "chain_agent", "news_agent", "macro_agent")

_AGENT_DOMAIN_CONTEXT = {
    "tech_agent": (
        "你的主要信号：RSI（超卖<30, 超买>70）、MACD 交叉、SMA20/60 交叉、"
        "布林带宽度（挤压 vs 扩张）、ATR。回顾哪些导致了正确/错误判断。"
    ),
    "chain_agent": (
        "你的主要信号：funding rate 极端值（>0.03%拥挤多头, <-0.01%拥挤空头）、"
        "交易所净流量方向、鲸鱼转账聚集、OI 变化、清算接近度。"
    ),
    "news_agent": (
        "你的主要信号：新闻情绪分数极端值（>0.5或<-0.5作为反向指标）、"
        "重大事件识别（监管、ETF、交易所事故）、噪音 vs 信号分类。"
    ),
    "macro_agent": (
        "你的主要信号：联储利率方向变化、DXY 趋势与加密的反向关系、"
        "恐贪指数极端值（<25或>75）、ETF 日流入>$200M、VIX 飙升。"
    ),
}

_AGENT_LABELS = {
    "tech_agent": "Technical Analysis",
    "chain_agent": "On-Chain Analysis",
    "news_agent": "News & Sentiment Analysis",
    "macro_agent": "Macro Analysis",
}

_MIN_RECORDS_PER_AGENT = 3

_EXPERIENCE_OUTPUT_SCHEMA = (
    "{\n"
    '  "success_patterns": [\n'
    '    {"pattern": "描述", "conditions": {"regime_tags": ["high_funding"]}, '
    '"rate": 0.70, "sample_count": 12, "reason": "为什么有效"}\n'
    "  ],\n"
    '  "forbidden_zones": [\n'
    '    {"pattern": "描述", "conditions": {"regime_tags": ["high_vol"]}, '
    '"rate": 0.65, "sample_count": 8, "reason": "为什么危险"}\n'
    "  ],\n"
    '  "strategic_insights": ["洞察1", "洞察2"]\n'
    "}"
)

# ── DB Layer ──


def _ensure_db(db_path: Path) -> None:
    """Create the reflections table if it doesn't exist, with experience_json column."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS agent_reflections (
                agent_id       TEXT PRIMARY KEY,
                memo           TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            )"""
        )
        # Migrate: add experience_json column if not present
        cursor = conn.execute("PRAGMA table_info(agent_reflections)")
        columns = {row[1] for row in cursor.fetchall()}
        if "experience_json" not in columns:
            conn.execute("ALTER TABLE agent_reflections ADD COLUMN experience_json TEXT")


def _load_reflections_sync(db_path: Path) -> dict[str, ExperienceMemory]:
    """Load all agent experience memories from SQLite."""
    if not db_path.exists():
        return {}
    with sqlite3.connect(str(db_path)) as conn:
        # Check if experience_json column exists
        cursor = conn.execute("PRAGMA table_info(agent_reflections)")
        columns = {row[1] for row in cursor.fetchall()}
        has_json = "experience_json" in columns

        if has_json:
            rows = conn.execute("SELECT agent_id, memo, experience_json FROM agent_reflections").fetchall()
        else:
            rows = conn.execute("SELECT agent_id, memo FROM agent_reflections").fetchall()

    result: dict[str, ExperienceMemory] = {}
    for row in rows:
        agent_id = row[0]
        memo = row[1]
        experience_json = row[2] if has_json and len(row) > 2 else None

        if experience_json:
            result[agent_id] = _deserialize_memory(experience_json)
        elif memo:
            # Fallback: wrap legacy memo as strategic insight
            result[agent_id] = ExperienceMemory(strategic_insights=[memo])
        else:
            result[agent_id] = ExperienceMemory()
    return result


def _save_reflection_sync(db_path: Path, agent_id: str, memory: ExperienceMemory) -> None:
    """Persist a single agent's experience memory."""
    _ensure_db(db_path)
    now = datetime.now(UTC).isoformat()
    memory.updated_at = now
    json_str = _serialize_memory(memory)
    # Keep memo column populated for backward compat (plain text summary)
    memo = "; ".join(memory.strategic_insights) if memory.strategic_insights else ""
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO agent_reflections (agent_id, memo, updated_at, experience_json) "
            "VALUES (?, ?, ?, ?)",
            (agent_id, memo, now, json_str),
        )


def _serialize_memory(memory: ExperienceMemory) -> str:
    """Serialize ExperienceMemory to JSON string."""
    return json.dumps(asdict(memory), ensure_ascii=False)


def _deserialize_memory(json_str: str) -> ExperienceMemory:
    """Deserialize JSON string to ExperienceMemory."""
    data = json.loads(json_str)
    return ExperienceMemory(
        success_patterns=[ExperienceRule(**r) for r in data.get("success_patterns", [])],
        forbidden_zones=[ExperienceRule(**r) for r in data.get("forbidden_zones", [])],
        strategic_insights=data.get("strategic_insights", []),
        updated_at=data.get("updated_at", ""),
    )


# ── Async wrappers ──


async def load_reflections(db_path: Path | None = None) -> dict[str, ExperienceMemory]:
    """Load all agent experience memories from SQLite."""
    path = db_path or _DEFAULT_DB_PATH
    return await asyncio.to_thread(_load_reflections_sync, path)


async def save_reflection(db_path: Path, agent_id: str, memory: ExperienceMemory | str) -> None:
    """Persist a single agent's reflection (accepts ExperienceMemory or legacy str)."""
    if isinstance(memory, str):
        memory = ExperienceMemory(strategic_insights=[memory])
    await asyncio.to_thread(_save_reflection_sync, db_path, agent_id, memory)


# ── Commit formatting ──


def _format_commit_for_agent(agent_id: str, dc: DecisionCommit) -> dict | None:
    """Extract a single agent's analysis from a DecisionCommit for reflection."""
    analysis = dc.analyses.get(agent_id)
    if analysis is None:
        return None

    summary = dc.snapshot_summary or {}
    verdict_action = dc.verdict.action if dc.verdict else "hold"

    return {
        "date": dc.timestamp.strftime("%Y-%m-%d %H:%M"),
        "direction": analysis.direction,
        "confidence": analysis.confidence,
        "reasoning": analysis.reasoning,
        "key_factors": analysis.key_factors,
        "pnl": dc.pnl,
        "verdict_action": verdict_action,
        "price": summary.get("price", 0),
        "volatility": summary.get("volatility", 0),
        "funding_rate": summary.get("funding_rate", 0),
    }


# ── Prompt construction ──


def _build_evolution_system_prompt(agent_id: str, existing_memory: ExperienceMemory | None) -> str:
    """Build system prompt for structured experience evolution."""
    label = _AGENT_LABELS.get(agent_id, agent_id)
    base = f"你是 {label} agent。你正在回顾自己过去的分析记录和实际结果，以生成结构化的经验记忆来改进未来的判断。"
    constraints = (
        "\n\n约束：\n"
        "1. 规律只能基于以下维度归纳：funding_rate、volatility、价格趋势、共识度、恐贪指数。"
        "禁止基于日期、具体价格、具体交易对归纳规律。\n"
        "2. 每条规律最多使用 3 个条件维度。\n"
        "3. 对每条规律，需提供至少一个反例或失败场景描述（falsification）。\n"
        "4. rate 字段是你估计的成功率/失败率，必须基于历史数据中的实际比例。"
    )
    evolution = ""
    if existing_memory and (existing_memory.success_patterns or existing_memory.forbidden_zones):
        evolution = (
            "\n\n增量进化指令：你将收到现有规则库。对每条现有规则，你可以：\n"
            "- retain：保留不变\n"
            "- update：根据新数据更新 rate/sample_count/reason\n"
            "- retire：如果新数据显示规则不再有效，移除\n"
            "- add：发现新规律时添加\n"
            "合并时保留现有 conditions 结构。"
        )
    return base + constraints + evolution


def _build_evolution_user_prompt(agent_id: str, records: list[dict], existing_memory: ExperienceMemory | None) -> str:
    """Build user prompt with history records and output schema."""
    label = _AGENT_LABELS.get(agent_id, agent_id)
    domain_ctx = _AGENT_DOMAIN_CONTEXT.get(agent_id, "")

    history = _format_history_records(records)
    existing = _format_existing_memory(existing_memory)

    return (
        f"你是 {label} agent。回顾你最近 {len(records)} 次有结果的分析。\n\n"
        f"【领域专属信号】\n{domain_ctx}\n\n"
        f"你的历史分析：\n{history}\n---\n\n"
        f"{existing}"
        f"任务：基于历史数据，输出结构化经验记忆 JSON。\n\n"
        f"输出格式：\n{_EXPERIENCE_OUTPUT_SCHEMA}\n\n"
        "注意：\n"
        "- success_patterns: 成功率高的模式（win_rate 在 rate 字段中）\n"
        "- forbidden_zones: 失败率高的模式（loss_rate 在 rate 字段中）\n"
        "- conditions.regime_tags 可选值: high_funding, negative_funding, high_vol, low_vol, "
        "trending_up, trending_down, extreme_fear, extreme_greed\n"
        "- 只输出 JSON，不要额外文字。"
    )


def _format_history_records(records: list[dict]) -> str:
    """Format historical records for prompt injection."""
    lines: list[str] = []
    for r in records:
        pnl_str = f"pnl={r['pnl']:+.2f}" if r["pnl"] is not None else "pnl=pending"
        kf = ", ".join(r["key_factors"]) if r.get("key_factors") else "none"
        lines.append(
            f"---\n"
            f"[{r['date']}] | direction={r['direction']} confidence={r['confidence']:.2f} "
            f"| 结果: {pnl_str} | verdict={r['verdict_action']}\n"
            f"  Reasoning: {r['reasoning'][:300]}\n"
            f"  Key factors: [{kf}]\n"
            f"  市场环境: price={r['price']} volatility={r['volatility']:.4f} "
            f"funding={r['funding_rate']:.5f}"
        )
    return "".join(lines)


def _format_existing_memory(memory: ExperienceMemory | None) -> str:
    """Format existing memory for incremental evolution prompt."""
    if not memory or (not memory.success_patterns and not memory.forbidden_zones):
        return ""
    parts = ["【现有规则库】\n"]
    parts.extend(
        f"  [SUCCESS] {r.pattern} | rate={r.rate:.2f} sample={r.sample_count} conditions={r.conditions}\n"
        for r in memory.success_patterns
    )
    parts.extend(
        f"  [FORBIDDEN] {r.pattern} | rate={r.rate:.2f} sample={r.sample_count} conditions={r.conditions}\n"
        for r in memory.forbidden_zones
    )
    parts.append("\n")
    return "".join(parts)


# ── Legacy prompt (kept for backward compat in tests) ──


def _build_reflection_prompt(agent_id: str, records: list[dict]) -> tuple[str, str]:
    """Build system + user prompt for a single agent's reflection (legacy interface)."""
    system = _build_evolution_system_prompt(agent_id, None)
    user = _build_evolution_user_prompt(agent_id, records, None)
    return system, user


# ── Verification layer ──


def _filter_records_by_regime(records: list[dict], rule: ExperienceRule) -> list[dict]:
    """Filter records to those matching the rule's regime conditions."""
    from cryptotrader.config import RegimeThresholdsConfig
    from cryptotrader.learning.regime import regime_overlap, tag_regime

    regime_tags = rule.conditions.get("regime_tags", [])
    if not regime_tags:
        return records  # No regime constraint → use all records

    thresholds = RegimeThresholdsConfig()
    filtered = []
    for r in records:
        summary = {
            "funding_rate": r.get("funding_rate", 0),
            "volatility": r.get("volatility", 0),
        }
        r_tags = tag_regime(summary, thresholds)
        if regime_overlap(regime_tags, r_tags) > 0:
            filtered.append(r)
    return filtered


def _compute_empirical_rate(records: list[dict]) -> float:
    """Compute actual win rate from records with PnL."""
    with_pnl = [r for r in records if r["pnl"] is not None]
    if not with_pnl:
        return 0.0
    wins = sum(1 for r in with_pnl if r["pnl"] > 0)
    return wins / len(with_pnl)


def _verify_rules(rules: list[ExperienceRule], records: list[dict], tolerance: float) -> list[ExperienceRule]:
    """Filter rules whose claimed rate deviates too much from empirical data.

    Filters records by regime conditions before computing empirical rate,
    so verification is regime-aware rather than using global statistics.
    """
    verified: list[ExperienceRule] = []
    for rule in rules:
        if rule.sample_count < 3:
            # Too few samples to verify — keep as observation
            rule.maturity = "observation"
            verified.append(rule)
            continue
        # Filter records to matching regime conditions
        regime_records = _filter_records_by_regime(records, rule)
        if len(regime_records) < 2:
            # Not enough regime-matched data to verify — keep rule
            verified.append(rule)
            continue
        # Compute empirical rate: win_rate for success, loss_rate for forbidden
        win_rate = _compute_empirical_rate(regime_records)
        empirical = win_rate if rule.category == "success_pattern" else 1.0 - win_rate
        if abs(rule.rate - empirical) <= tolerance:
            verified.append(rule)
        else:
            logger.info(
                "Rule rejected (rate drift): claimed=%.2f empirical=%.2f pattern=%s",
                rule.rate,
                empirical,
                rule.pattern[:80],
            )
    return verified


def _assign_maturity(rule: ExperienceRule) -> ExperienceRule:
    """Assign maturity level based on sample_count and regime_count."""
    if rule.sample_count >= 30 and rule.regime_count >= 2:
        rule.maturity = "rule"
    elif rule.sample_count >= 10:
        rule.maturity = "hypothesis"
    else:
        rule.maturity = "observation"
    return rule


# ── LLM reflection ──


async def run_agent_reflection(
    agent_id: str,
    records: list[dict],
    model: str,
    existing_memory: ExperienceMemory | None = None,
) -> ExperienceMemory:
    """Execute a single LLM reflection call for one agent.

    Returns structured ExperienceMemory.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from cryptotrader.agents.base import create_llm, extract_content

    system_prompt = _build_evolution_system_prompt(agent_id, existing_memory)
    user_prompt = _build_evolution_user_prompt(agent_id, records, existing_memory)

    llm = create_llm(model=model, temperature=0.3, json_mode=True)

    response = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    text = extract_content(response)
    return _parse_experience_response(text)


def _parse_experience_response(text: str) -> ExperienceMemory:
    """Parse LLM response into ExperienceMemory with fallback."""
    try:
        from cryptotrader.debate.verdict import _extract_json

        data = _extract_json(text)
        return _build_memory_from_dict(data)
    except (ValueError, json.JSONDecodeError, KeyError):
        logger.warning("Failed to parse structured experience, falling back to text")
        return ExperienceMemory(strategic_insights=[text])


def _build_memory_from_dict(data: dict) -> ExperienceMemory:
    """Build ExperienceMemory from parsed JSON dict."""
    success = [
        _assign_maturity(
            ExperienceRule(
                pattern=r.get("pattern", ""),
                category="success_pattern",
                conditions=r.get("conditions", {}),
                rate=r.get("rate", 0.0),
                sample_count=r.get("sample_count", 0),
                reason=r.get("reason", ""),
            )
        )
        for r in data.get("success_patterns", [])
    ]
    forbidden = [
        _assign_maturity(
            ExperienceRule(
                pattern=r.get("pattern", ""),
                category="forbidden_zone",
                conditions=r.get("conditions", {}),
                rate=r.get("rate", 0.0),
                sample_count=r.get("sample_count", 0),
                reason=r.get("reason", ""),
            )
        )
        for r in data.get("forbidden_zones", [])
    ]
    insights = data.get("strategic_insights", [])
    return ExperienceMemory(
        success_patterns=success,
        forbidden_zones=forbidden,
        strategic_insights=insights,
    )


# ── Main entry point ──


async def maybe_reflect(
    store: JournalStore,
    cycle_count: int,
    config: ExperienceConfig,
    db_path: Path | None = None,
) -> dict[str, ExperienceMemory]:
    """Check if reflection is due; if so, run LLM reflections for all agents.

    Returns dict of agent_id → ExperienceMemory (empty if skipped).
    """
    if not config.enabled:
        return {}

    if cycle_count % config.every_n_cycles != 0:
        return {}

    path = db_path or _DEFAULT_DB_PATH

    try:
        # Load existing memories for incremental evolution
        existing_memories = await load_reflections(path)
        return await _run_all_reflections(store, config, path, existing_memories)
    except Exception:
        logger.warning("Reflection cycle failed", exc_info=True)
        return {}


async def _run_all_reflections(
    store: JournalStore,
    config: ExperienceConfig,
    path: Path,
    existing_memories: dict[str, ExperienceMemory],
) -> dict[str, ExperienceMemory]:
    """Run reflections for all agents (extracted to satisfy C901)."""
    commits = await store.log(limit=config.lookback_commits)
    commits_with_pnl = [dc for dc in commits if dc.pnl is not None]

    if len(commits_with_pnl) < config.min_commits_required:
        logger.debug(
            "Reflection skipped: only %d commits with PnL (need %d)",
            len(commits_with_pnl),
            config.min_commits_required,
        )
        return {}

    logger.info("Running agent reflections (%d commits with PnL)", len(commits_with_pnl))
    results: dict[str, ExperienceMemory] = {}

    for agent_id in _AGENT_IDS:
        memory = await _reflect_single_agent(
            agent_id,
            commits_with_pnl,
            config,
            path,
            existing_memories.get(agent_id),
        )
        if memory is not None:
            results[agent_id] = memory

    return results


async def _reflect_single_agent(
    agent_id: str,
    commits_with_pnl: list,
    config: ExperienceConfig,
    path: Path,
    existing_memory: ExperienceMemory | None,
) -> ExperienceMemory | None:
    """Run reflection for a single agent."""
    records = []
    for dc in commits_with_pnl:
        rec = _format_commit_for_agent(agent_id, dc)
        if rec is not None:
            records.append(rec)

    if len(records) < _MIN_RECORDS_PER_AGENT:
        logger.debug("Reflection skipped for %s: only %d records", agent_id, len(records))
        return None

    try:
        memory = await run_agent_reflection(agent_id, records, config.model, existing_memory)
        # Verify rules against empirical data
        memory.success_patterns = _verify_rules(memory.success_patterns, records, config.verify_win_rate_tolerance)
        memory.forbidden_zones = _verify_rules(memory.forbidden_zones, records, config.verify_win_rate_tolerance)
        await save_reflection(path, agent_id, memory)
        logger.info(
            "Reflection saved for %s (%d patterns, %d zones)",
            agent_id,
            len(memory.success_patterns),
            len(memory.forbidden_zones),
        )
        return memory
    except Exception:
        logger.warning("Reflection failed for %s", agent_id, exc_info=True)
        return None
