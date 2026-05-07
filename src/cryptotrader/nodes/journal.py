"""Journal commit nodes — record trades and rejections."""

from __future__ import annotations

import logging
from typing import Any

from cryptotrader.state import ArenaState, get_pair
from cryptotrader.tracing import node_logger

logger = logging.getLogger(__name__)


_LATENCY_STAGE_MAP: dict[str, str] = {
    # Node name → stage bucket. Unknown nodes fall into "other".
    "collect_data": "data",
    "update_pnl": "data",
    "stop_loss_check": "data",
    "verbal_reinforcement": "data",
    "inject_experience": "data",
    "init_decision": "data",
    "tech_agent": "agents",
    "chain_agent": "agents",
    "news_agent": "agents",
    "macro_agent": "agents",
    "debate_gate": "debate",
    "debate": "debate",
    "debate_round": "debate",
    "debate_round_1": "debate",
    "debate_round_2": "debate",
    "check_stability": "debate",
    "enrich_context": "verdict",
    "enrich_verdict_context": "verdict",
    "verdict": "verdict",
    "judge_verdict": "verdict",
    "bull_bear_debate": "debate",
    "risk_gate": "risk",
    "hitl_gate": "risk",
    "execute_trade": "execute",
    "execute": "execute",
    "record_trade": "execute",
    "record_rejection": "execute",
    "journal_trade": "execute",
    "journal_rejection": "execute",
}


def _snapshot_token_usage() -> dict[str, Any]:
    """Export the current decision's token ledger, or empty dict if unbound.

    Shape matches :class:`cryptotrader.models.TokenUsage`.
    """
    from cryptotrader.llm.token_tracker import current_ledger

    ledger = current_ledger()
    if ledger is None:
        return {}
    return ledger.to_dict()


def _resolve_node_trace(state: ArenaState) -> list:
    """Resolve the node_trace for this run.

    Preference order:
      1) ``state["data"]["node_trace"]`` — set by an external runner if it ever does.
      2) ``tracing.trace_get(metadata.trace_id)`` — populated by run_graph_traced
         and by analysis_runner via the trace registry.
      3) Empty list.

    The registry path is the production source: nodes can't share an updated
    list via state-deltas (LangGraph would only see the last-write per chunk),
    so the runner accumulates externally and the journal node reads it here.
    """
    explicit = state.get("data", {}).get("node_trace")
    if explicit:
        return explicit
    from cryptotrader.tracing import trace_get

    trace_id = state.get("metadata", {}).get("trace_id")
    return trace_get(trace_id)


def _aggregate_latency(node_trace: list) -> dict[str, Any]:
    """Collapse per-node durations into {data,agents,debate,verdict,risk,execute,total}.

    Shape matches :class:`cryptotrader.models.LatencyBreakdown`.
    """
    buckets: dict[str, float] = {
        "data": 0.0,
        "agents": 0.0,
        "debate": 0.0,
        "verdict": 0.0,
        "risk": 0.0,
        "execute": 0.0,
        "other": 0.0,
    }
    for entry in node_trace or []:
        if isinstance(entry, dict):
            node = entry.get("node", "")
            ms = float(entry.get("duration_ms", 0.0) or 0.0)
        else:
            node = getattr(entry, "node", "")
            ms = float(getattr(entry, "duration_ms", 0.0) or 0.0)
        bucket = _LATENCY_STAGE_MAP.get(node, "other")
        buckets[bucket] = buckets.get(bucket, 0.0) + ms
    buckets["total"] = sum(buckets.values())
    return buckets


def _to_agent_analyses(raw_analyses: dict, pair: str) -> dict:
    """Convert raw analysis dicts from state to AgentAnalysis objects."""
    from cryptotrader.models import AgentAnalysis

    analyses = {}
    for k, v in raw_analyses.items():
        if isinstance(v, dict):
            analyses[k] = AgentAnalysis(
                agent_id=k,
                pair=pair,
                direction=v.get("direction", "neutral"),
                confidence=v.get("confidence", 0.5),
                reasoning=v.get("reasoning", ""),
                key_factors=v.get("key_factors", []),
                risk_flags=v.get("risk_flags", []),
                data_points=v.get("data_points", {}),
                data_sufficiency=v.get("data_sufficiency", "medium"),
                is_mock=v.get("is_mock", False),
            )
        else:
            analyses[k] = v
    return analyses


async def _get_portfolio_snapshot(state: ArenaState) -> dict:
    """Get current portfolio state for journal commit."""
    try:
        from cryptotrader.portfolio.manager import PortfolioManager

        db_url = state["metadata"].get("database_url")
        pm = PortfolioManager(db_url)
        portfolio = await pm.get_portfolio()
        return {
            "total_value": portfolio.get("total_value", 0),
            "positions": portfolio.get("positions", {}),
        }
    except Exception:
        logger.info("Portfolio snapshot for journal failed", exc_info=True)
        return {}


def _write_memory_case(state: ArenaState, commit: Any, pair_str: str, raw_verdict: dict | None) -> None:
    """FR-006: 写入 agent_memory/cases/<cycle_id>.md（per-cycle 单文件）。

    FR-007: 失败时 logger.warning，不阻塞 cycle。
    调用 learning.memory.write_case()（原子写，进程内锁）。
    """
    try:
        from cryptotrader.learning.memory import write_case

        cycle_id = commit.hash[:16]  # 使用 commit hash 前 16 位作为 cycle_id

        # 从 analyses 中提取各 agent 的 reasoning 文本
        agent_analyses: dict[str, str] = {}
        for agent_id, analysis in (commit.analyses or {}).items():
            if hasattr(analysis, "reasoning"):
                agent_analyses[agent_id] = analysis.reasoning or ""
            elif isinstance(analysis, dict):
                agent_analyses[agent_id] = analysis.get("reasoning", "")

        verdict_action = raw_verdict.get("action", "hold") if raw_verdict else "hold"
        verdict_reasoning = raw_verdict.get("reasoning", "") if raw_verdict else ""
        # FR-026: 用 parse_applied() 解析 applied: 引用（支持 prefix + bare 两种形式）
        from cryptotrader.learning.memory import parse_applied

        applied_map = parse_applied(verdict_reasoning)  # {agent: [pattern_name]}
        # 展平为 list[str]，格式为 "agent::pattern_name"
        applied_patterns: list[str] = [f"{agent}::{name}" for agent, names in applied_map.items() for name in names]

        risk_gate_passed = commit.risk_gate.passed if commit.risk_gate else True
        exec_status = commit.execution_status

        write_case(
            cycle_id=cycle_id,
            pair=pair_str,
            agent_analyses=agent_analyses,
            verdict_action=verdict_action,
            verdict_reasoning=verdict_reasoning,
            applied_patterns=applied_patterns,
            risk_gate_passed=risk_gate_passed,
            execution_status=exec_status,
            final_pnl=commit.pnl,
        )

        # T041 / FR-026: 有 final_pnl 时立即更新 pattern PnL track
        if commit.pnl is not None and applied_map:
            from cryptotrader.learning.memory import update_pattern_pnl

            update_pattern_pnl(applied_map, pnl=commit.pnl)

    except Exception:
        logger.warning("Memory case write failed (non-blocking)", exc_info=True)


@node_logger()
async def journal_trade(state: ArenaState) -> dict:
    """Journal a successful trade."""
    try:
        from cryptotrader.journal.commit import build_commit
        from cryptotrader.journal.store import JournalStore
        from cryptotrader.models import GateResult, Order, TradeVerdict
        from cryptotrader.tracing import get_trace_id

        db_url = state["metadata"].get("database_url")
        store = JournalStore(db_url)

        pair_str = get_pair(state).canonical()
        analyses = _to_agent_analyses(state["data"].get("analyses", {}), pair_str)

        raw_verdict = state["data"].get("verdict")
        verdict = None
        if raw_verdict and isinstance(raw_verdict, dict):
            verdict = TradeVerdict(**{k: v for k, v in raw_verdict.items() if k in TradeVerdict.__dataclass_fields__})

        raw_gate = state["data"].get("risk_gate")
        risk_gate = None
        if raw_gate and isinstance(raw_gate, dict):
            risk_gate = GateResult(**{k: v for k, v in raw_gate.items() if k in GateResult.__dataclass_fields__})

        raw_order = state["data"].get("order")
        order = None
        if raw_order and isinstance(raw_order, dict):
            order = Order(**{k: v for k, v in raw_order.items() if k in Order.__dataclass_fields__})

        fill_price = state["data"].get("fill_price")
        slippage = state["data"].get("slippage")

        # Build portfolio_after snapshot
        portfolio_after = await _get_portfolio_snapshot(state)

        parent_hash = state["data"].get("journal_hash")
        node_trace = _resolve_node_trace(state)
        # Capture realized PnL for close-action commits at write time.
        # Open-action commits stay pnl=None; their fake "unrealized snapshot at
        # next cycle" pollution from update_past_pnl is also disabled for opens
        # (see nodes/data.py:_update_one_commit_pnl). This keeps commit.pnl
        # semantically clean: realized round-trip P&L when present, None for
        # opens / skipped trades. Frontend stats (avg_trade_pnl / win_rate /
        # realized_pnl_30d) all rely on this contract since 2026-05-07.
        action = (raw_verdict or {}).get("action", "").lower()
        commit_pnl = state["data"].get("realized_pnl") if action == "close" else None
        commit = build_commit(
            pair=pair_str,
            snapshot_summary=state["data"].get("snapshot_summary", {}),
            analyses=analyses,
            debate_rounds=state.get("debate_round", 0),
            divergence=(state.get("divergence_scores") or [0.0])[-1],
            verdict=verdict,
            risk_gate=risk_gate,
            order=order,
            parent_hash=parent_hash,
            fill_price=fill_price,
            slippage=slippage,
            portfolio_after=portfolio_after,
            pnl=commit_pnl,
            trace_id=get_trace_id(),
            consensus_metrics=state["data"].get("consensus_metrics"),
            verdict_source=state["data"].get("verdict", {}).get("verdict_source", "ai"),
            node_trace=node_trace,
            debate_skip_reason=state["data"].get("debate_skip_reason", ""),
            challenges=state["data"].get("debate_turns") or [],
            latency_breakdown=_aggregate_latency(node_trace),
            token_usage=_snapshot_token_usage(),
            execution_status=state["data"].get("execution_status"),
        )
        await store.commit(commit)
        logger.info(
            "Journal trade committed: hash=%s pair=%s action=%s",
            commit.hash,
            pair_str,
            raw_verdict.get("action") if raw_verdict else "unknown",
        )

        # FR-006: 写入 agent_memory/cases/<cycle_id>.md（per-cycle 单文件）
        # FR-007: 写失败不阻塞 cycle（write_case 内部捕获异常）
        _write_memory_case(state, commit, pair_str, raw_verdict)

        return {"data": {"journal_hash": commit.hash}}
    except Exception:
        logger.error(
            "CRITICAL: journal commit failed after live trade for %s",
            state["metadata"].get("pair"),
            exc_info=True,
        )
        return {"data": {"journal_hash": None}}


@node_logger()
async def journal_rejection(state: ArenaState) -> dict:
    """Journal a risk-gate rejection."""
    try:
        from cryptotrader.journal.commit import build_commit
        from cryptotrader.journal.store import JournalStore
        from cryptotrader.models import GateResult, TradeVerdict
        from cryptotrader.nodes.verdict import _get_notifier
        from cryptotrader.tracing import get_trace_id

        db_url = state["metadata"].get("database_url")
        store = JournalStore(db_url)

        pair_str = get_pair(state).canonical()
        analyses = _to_agent_analyses(state["data"].get("analyses", {}), pair_str)

        raw_verdict = state["data"].get("verdict")
        verdict = None
        if raw_verdict and isinstance(raw_verdict, dict):
            verdict = TradeVerdict(**{k: v for k, v in raw_verdict.items() if k in TradeVerdict.__dataclass_fields__})

        raw_gate = state["data"].get("risk_gate")
        risk_gate = None
        if raw_gate and isinstance(raw_gate, dict):
            risk_gate = GateResult(**{k: v for k, v in raw_gate.items() if k in GateResult.__dataclass_fields__})

        parent_hash = state["data"].get("journal_hash")
        node_trace = _resolve_node_trace(state)
        commit = build_commit(
            pair=pair_str,
            snapshot_summary=state["data"].get("snapshot_summary", {}),
            analyses=analyses,
            debate_rounds=state.get("debate_round", 0),
            divergence=(state.get("divergence_scores") or [0.0])[-1],
            verdict=verdict,
            risk_gate=risk_gate,
            order=None,
            parent_hash=parent_hash,
            trace_id=get_trace_id(),
            consensus_metrics=state["data"].get("consensus_metrics"),
            verdict_source=state["data"].get("verdict", {}).get("verdict_source", "ai"),
            node_trace=node_trace,
            debate_skip_reason=state["data"].get("debate_skip_reason", ""),
            challenges=state["data"].get("debate_turns") or [],
            latency_breakdown=_aggregate_latency(node_trace),
            token_usage=_snapshot_token_usage(),
        )
        await store.commit(commit)
        logger.info(
            "Journal rejection committed: hash=%s pair=%s rejected_by=%s",
            commit.hash,
            pair_str,
            raw_gate.get("rejected_by") if raw_gate else "unknown",
        )

        # Fire-and-forget rejection notification
        try:
            notifier = _get_notifier(state)
            raw_gate = state["data"].get("risk_gate", {})
            await notifier.notify(
                "rejection",
                {
                    "pair": pair_str,
                    "rejected_by": raw_gate.get("rejected_by"),
                    "reason": raw_gate.get("reason"),
                },
            )
        except Exception:
            logger.info("Rejection notification failed", exc_info=True)

        return {"data": {"journal_hash": commit.hash}}
    except Exception:
        logger.error(
            "CRITICAL: journal commit failed for rejection of %s",
            state["metadata"].get("pair"),
            exc_info=True,
        )
        return {"data": {"journal_hash": None}}
