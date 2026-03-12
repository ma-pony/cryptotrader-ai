"""Agent self-reflection — LLM-driven strategy memo generation.

Each agent periodically reviews its own historical analyses + actual PnL results
to generate a strategy memo (3-5 insights). The memo is injected into the agent's
prompt on subsequent analyses, enabling self-correction over time.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.config import ReflectionConfig
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.models import DecisionCommit

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


def _ensure_db(db_path: Path) -> None:
    """Create the reflections table if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS agent_reflections (
                agent_id   TEXT PRIMARY KEY,
                memo       TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )


def _load_reflections_sync(db_path: Path) -> dict[str, str]:
    """Synchronous SQLite read — all agent memos."""
    if not db_path.exists():
        return {}
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT agent_id, memo FROM agent_reflections").fetchall()
    return {row[0]: row[1] for row in rows}


def _save_reflection_sync(db_path: Path, agent_id: str, memo: str) -> None:
    """Synchronous SQLite upsert — single agent memo."""
    _ensure_db(db_path)
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO agent_reflections (agent_id, memo, updated_at) VALUES (?, ?, ?)",
            (agent_id, memo, now),
        )


async def load_reflections(db_path: Path | None = None) -> dict[str, str]:
    """Load all agent strategy memos from SQLite.

    Returns:
        Dict mapping agent_id → memo text. Empty dict if no reflections yet.
    """
    path = db_path or _DEFAULT_DB_PATH
    return await asyncio.to_thread(_load_reflections_sync, path)


async def save_reflection(db_path: Path, agent_id: str, memo: str) -> None:
    """Persist a single agent's reflection memo (upsert)."""
    await asyncio.to_thread(_save_reflection_sync, db_path, agent_id, memo)


def _format_commit_for_agent(agent_id: str, dc: DecisionCommit) -> dict | None:
    """Extract a single agent's analysis from a DecisionCommit for reflection.

    Returns None if the agent is not present in this commit.
    """
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


def _build_reflection_prompt(agent_id: str, records: list[dict]) -> tuple[str, str]:
    """Build system + user prompt for a single agent's reflection.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    label = _AGENT_LABELS.get(agent_id, agent_id)
    domain_ctx = _AGENT_DOMAIN_CONTEXT.get(agent_id, "")

    system_prompt = f"你是 {label} agent。你正在回顾自己过去的分析记录和实际结果，以生成策略备忘录来改进未来的判断。"

    # Format historical records
    history_lines: list[str] = []
    for r in records:
        pnl_str = f"pnl={r['pnl']:+.2f}" if r["pnl"] is not None else "pnl=pending"
        kf = ", ".join(r["key_factors"]) if r["key_factors"] else "none"
        history_lines.append(
            f"---\n"
            f"[{r['date']}] | direction={r['direction']} confidence={r['confidence']:.2f} "
            f"| 结果: {pnl_str} | verdict={r['verdict_action']}\n"
            f"  Reasoning: {r['reasoning'][:300]}\n"
            f"  Key factors: [{kf}]\n"
            f"  市场环境: price={r['price']} volatility={r['volatility']:.4f} "
            f"funding={r['funding_rate']:.5f}"
        )

    user_prompt = (
        f"你是 {label} agent。回顾你最近 {len(records)} 次有结果的分析。\n\n"
        f"【领域专属信号】\n{domain_ctx}\n\n"
        f"你的历史分析：\n{''.join(history_lines)}\n---\n\n"
        "任务：写一份简洁的策略备忘录（3-5 条要点），你的未来自己会在每次分析前阅读。\n"
        "1. 你领域中哪些信号在最近的分析中最有预测力？\n"
        "2. 哪些信号具有误导性或导致了错误判断？举具体例子。\n"
        "3. 你有什么系统性偏差需要纠正？\n"
        "4. 一条具体的规则或阈值调整，应用于下次分析。\n\n"
        "基于上方历史数据给出具体建议，不要泛泛而谈。\n"
        "输出纯文本，不需要 JSON。"
    )

    return system_prompt, user_prompt


async def run_agent_reflection(
    agent_id: str,
    records: list[dict],
    model: str,
) -> str:
    """Execute a single LLM reflection call for one agent.

    Args:
        agent_id: e.g. "tech_agent"
        records: List of formatted commit records (from _format_commit_for_agent)
        model: LLM model to use (empty string = use config default)

    Returns:
        Strategy memo text.
    """
    from cryptotrader.agents.base import create_llm

    system_prompt, user_prompt = _build_reflection_prompt(agent_id, records)

    llm = create_llm(model=model, temperature=0.3)

    from langchain_core.messages import HumanMessage, SystemMessage

    response = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    from cryptotrader.agents.base import extract_content

    return extract_content(response)


async def maybe_reflect(
    store: JournalStore,
    cycle_count: int,
    config: ReflectionConfig,
    db_path: Path | None = None,
) -> dict[str, str]:
    """Check if reflection is due; if so, run LLM reflections for all agents.

    This function is designed to be called every cycle. It only triggers
    actual LLM calls every `config.every_n_cycles` cycles.

    Args:
        store: JournalStore to fetch historical commits from.
        cycle_count: Current trading cycle number.
        config: Reflection configuration.
        db_path: Path to reflections SQLite DB.

    Returns:
        Updated agent reflections dict (empty if reflection was skipped).
    """
    if not config.enabled:
        return {}

    if cycle_count % config.every_n_cycles != 0:
        return {}

    path = db_path or _DEFAULT_DB_PATH

    try:
        commits = await store.log(limit=config.lookback_commits)
        # Filter to commits with PnL results
        commits_with_pnl = [dc for dc in commits if dc.pnl is not None]

        if len(commits_with_pnl) < config.min_commits_required:
            logger.debug(
                "Reflection skipped: only %d commits with PnL (need %d)",
                len(commits_with_pnl),
                config.min_commits_required,
            )
            return {}

        logger.info("Running agent reflections (cycle %d, %d commits with PnL)", cycle_count, len(commits_with_pnl))

        results: dict[str, str] = {}

        for agent_id in _AGENT_IDS:
            records = []
            for dc in commits_with_pnl:
                rec = _format_commit_for_agent(agent_id, dc)
                if rec is not None:
                    records.append(rec)

            if len(records) < _MIN_RECORDS_PER_AGENT:
                logger.debug("Reflection skipped for %s: only %d records", agent_id, len(records))
                continue

            try:
                memo = await run_agent_reflection(agent_id, records, config.model)
                await save_reflection(path, agent_id, memo)
                results[agent_id] = memo
                logger.info("Reflection saved for %s (%d chars)", agent_id, len(memo))
            except Exception:
                logger.warning("Reflection failed for %s", agent_id, exc_info=True)

        return results

    except Exception:
        logger.warning("Reflection cycle failed", exc_info=True)
        return {}
