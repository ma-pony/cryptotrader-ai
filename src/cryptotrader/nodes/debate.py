"""Debate round nodes — cross-challenge and convergence."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from cryptotrader.agents.base import create_llm, extract_content
from cryptotrader.state import ArenaState
from cryptotrader.tracing import node_logger

logger = logging.getLogger(__name__)

_DEBATE_ROLES = {
    "tech_agent": "technical analysis",
    "chain_agent": "on-chain and derivatives analysis",
    "news_agent": "news and sentiment analysis",
    "macro_agent": "macroeconomic analysis",
}

DEBATE_SYSTEM = """You are a {role} specialist in a multi-agent trading debate.

Rules:
- Base arguments ONLY on data. Cite specific numbers for every claim.
- HOLD your position when your data supports it — do NOT converge toward majority opinion.
- Only change your view if another agent presented a specific data point you hadn't considered.
- If you change, state exactly which data point changed your mind.
- Look for cross-domain contradictions (e.g., bullish technicals + bearish on-chain = important signal).

Output JSON: {{"direction": "bullish|bearish|neutral", "confidence": 0.0-1.0, "reasoning": "...",
"key_factors": [...], "risk_flags": [...], "new_findings": "cross-domain insight from other agents' data"}}"""


def _classify_move(before_dir: str, before_conf: float, after_dir: str, after_conf: float) -> str:
    """Classify how an agent's position shifted between rounds.

    Returns a short human-readable label used by the Debate UI. Invariants:
      - any direction flip is considered a concession (``让步``)
      - confidence up ≥ 0.05 → ``强化``
      - confidence down ≥ 0.05 → ``弱化``
      - otherwise ``保持``
    """
    if before_dir != after_dir:
        return f"让步(由{_dir_label(before_dir)}转{_dir_label(after_dir)})"
    delta = after_conf - before_conf
    if delta >= 0.05:
        return "强化"
    if delta <= -0.05:
        return "弱化"
    return "保持"


_DIR_LABELS = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}


def _dir_label(direction: str) -> str:
    return _DIR_LABELS.get(direction, direction)


async def _debate_one_agent(
    agent_id: str,
    analysis: dict,
    others: dict[str, dict],
    pair: str,
    model: str,
    timeout_seconds: float,
    round_number: int,
) -> tuple[str, dict, dict]:
    """Single agent's debate response — extracted for parallel execution.

    Returns ``(agent_id, merged_analysis, turn_entry)`` where ``turn_entry`` is a
    structured dict suitable for persisting into ``DecisionCommit.challenges``.
    """
    from cryptotrader.debate.challenge import build_challenge_prompt
    from cryptotrader.llm.json_retry import extract_json_with_retry

    prompt = build_challenge_prompt(agent_id, pair, analysis, others)
    role_label = _DEBATE_ROLES.get(agent_id, agent_id)
    system = DEBATE_SYSTEM.format(role=role_label)
    before_dir = analysis.get("direction", "neutral")
    before_conf = float(analysis.get("confidence", 0.0) or 0.0)
    # Pick the opponent with the lowest confidence as the "addressee" — the agent
    # most likely to be challenged. When ``others`` is empty (sole agent), ``to`` is
    # None and the UI renders the turn as a monologue.
    to_agent: str | None = None
    if others:
        to_agent = min(others, key=lambda k: float(others[k].get("confidence", 0.0) or 0.0))

    def _turn(after_dir: str, after_conf: float, reasoning: str, new_findings: str, errored: bool) -> dict:
        return {
            "round": round_number,
            "from": agent_id,
            "to": to_agent,
            "before": {"direction": before_dir, "confidence": before_conf},
            "after": {"direction": after_dir, "confidence": after_conf},
            "move": _classify_move(before_dir, before_conf, after_dir, after_conf),
            "reasoning": reasoning,
            "new_findings": new_findings,
            "errored": errored,
        }

    try:
        llm = create_llm(model=model, temperature=0.3)
        lc_msgs = [SystemMessage(content=system), HumanMessage(content=prompt)]
        resp = await asyncio.wait_for(llm.ainvoke(lc_msgs), timeout=timeout_seconds)
        text = extract_content(resp)
        data = await extract_json_with_retry(
            text,
            llm=llm,
            schema_hint="direction,confidence,reasoning,key_factors,risk_flags,new_findings",
            max_retries=2,
        )
        merged = dict(analysis)
        after_dir = data.get("direction", before_dir)
        after_conf = float(data.get("confidence", before_conf))
        reasoning = data.get("reasoning", analysis.get("reasoning", ""))
        new_findings = data.get("new_findings", "")
        merged.update(
            {
                "direction": after_dir,
                "confidence": after_conf,
                "reasoning": reasoning,
                "key_factors": data.get("key_factors", analysis.get("key_factors", [])),
                "risk_flags": data.get("risk_flags", analysis.get("risk_flags", [])),
                "new_findings": new_findings,
            }
        )
        return agent_id, merged, _turn(after_dir, after_conf, reasoning, new_findings, False)
    except TimeoutError:
        logger.warning(
            "LLM timeout for debate agent %s after %ss — keeping original analysis",
            agent_id,
            timeout_seconds,
        )
        return agent_id, analysis, _turn(before_dir, before_conf, "LLM timeout — 保持原立场", "", True)
    except Exception:
        logger.warning("Debate round LLM call failed for %s — keeping original analysis", agent_id, exc_info=True)
        return agent_id, analysis, _turn(before_dir, before_conf, "LLM failed — 保持原立场", "", True)


@node_logger()
async def debate_round(state: ArenaState) -> dict:
    """One round of cross-challenge debate between agents (parallel)."""
    from cryptotrader.config import load_config as _load_config

    round_number = state["debate_round"] + 1
    from cryptotrader.chat.runtime_registry import get_event_bus

    event_bus = get_event_bus((state.get("metadata") or {}).get("session_id"))
    if event_bus is not None:
        await event_bus.publish("debate_started", {"round_number": round_number})

    analyses = state["data"].get("analyses", {})
    _dcfg = _load_config()
    _default_debate_model = _dcfg.models.debate or _dcfg.models.fallback
    model = state["metadata"].get("debate_model", _default_debate_model)
    timeout_seconds: float = _dcfg.models.timeout_seconds

    from cryptotrader.state import get_pair

    # AI prompt — use Pair.display() for human-readable form ("BTC/USDT (perp)")
    # so the LLM understands market type context (FR-202, T020).
    pair_display = get_pair(state).display()

    tasks = []
    agent_ids = list(analyses.keys())
    for agent_id in agent_ids:
        analysis = analyses[agent_id]
        others = {k: v for k, v in analyses.items() if k != agent_id}
        tasks.append(
            _debate_one_agent(
                agent_id,
                analysis,
                others,
                pair_display,
                model,
                timeout_seconds,
                round_number,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    updated: dict[str, Any] = {}
    new_turns: list[dict] = []
    for i, result in enumerate(results):
        aid = agent_ids[i]
        if isinstance(result, BaseException):
            logger.warning(
                "Debate gather exception for %s: %r — keeping original analysis",
                aid,
                result,
                exc_info=result,
            )
            updated[aid] = analyses[aid]
        else:
            r_aid, r_analysis, r_turn = result
            updated[r_aid] = r_analysis
            new_turns.append(r_turn)

    if event_bus is not None:
        positions = {
            aid: {"direction": a.get("direction"), "confidence": a.get("confidence")} for aid, a in updated.items()
        }
        await event_bus.publish(
            "debate_round_done",
            {
                "round_number": round_number,
                "updated_positions": positions,
            },
        )

    # Append turns to the state's persistent debate_turns list so graph.py can
    # write them into DecisionCommit.challenges unchanged.
    existing_turns = list(state["data"].get("debate_turns") or [])
    existing_turns.extend(new_turns)

    return {
        "data": {
            "analyses": updated,
            "debate_turns": existing_turns,
        },
        "debate_round": state["debate_round"] + 1,
    }


@node_logger()
async def debate_gate(state: ArenaState) -> dict:
    """Compute consensus metrics; set debate_skipped flag."""
    from cryptotrader.config import load_config as _load_config
    from cryptotrader.debate.convergence import compute_consensus_strength, compute_divergence

    raw_analyses = state["data"].get("analyses", {})
    # Filter out mock analyses — they carry no real signal and pollute consensus metrics
    # (mock: confidence=0.0/0.1, direction=neutral → drags mean toward 0, fakes "confusion")
    analyses = {k: v for k, v in raw_analyses.items() if not v.get("is_mock", False)}

    config = _load_config().debate

    # Not enough real agents to evaluate consensus — force debate unconditionally
    if len(analyses) < 2:
        strength, mean_score = compute_consensus_strength(raw_analyses)
        dispersion = compute_divergence(raw_analyses)
        logger.info("< 2 real agents (%d/%d) — forcing debate", len(analyses), len(raw_analyses))
        return {
            "data": {
                "debate_skipped": False,
                "debate_skip_reason": "",
                "consensus_metrics": {
                    "strength": strength,
                    "mean_score": mean_score,
                    "dispersion": dispersion,
                    "skip_threshold": config.consensus_skip_threshold,
                    "confusion_threshold": config.confusion_skip_threshold,
                },
            }
        }

    try:
        strength, mean_score = compute_consensus_strength(analyses)
        dispersion = compute_divergence(analyses)
    except Exception:
        logger.warning("Failed to compute consensus metrics — forcing debate", exc_info=True)
        strength, mean_score, dispersion = 0.0, 0.0, 1.0

    skip = False
    reason = ""
    if config.skip_debate and strength > config.consensus_skip_threshold:
        skip = True
        reason = f"strong consensus (strength={strength:.3f})"
    elif (
        config.skip_debate
        and abs(mean_score) < config.confusion_skip_threshold
        and dispersion < config.confusion_max_dispersion
    ):
        # Low mean + low dispersion = shared confusion (all agents uncertain)
        # Low mean + high dispersion = disagreement (debate needed)
        skip = True
        reason = f"shared confusion (|mean|={abs(mean_score):.3f}, dispersion={dispersion:.3f})"

    # Normalize skip reason to a structured value for Dashboard observability
    if skip and strength > config.consensus_skip_threshold:
        skip_reason_tag = "consensus"
    elif skip:
        skip_reason_tag = "confusion"
    else:
        skip_reason_tag = ""

    if skip:
        logger.info("Debate SKIPPED: %s", reason)
        # Metrics instrumentation: ct_debate_skipped_total (req 9.5)
        from cryptotrader.metrics import get_metrics_collector

        get_metrics_collector().inc_debate_skipped()

    return {
        "data": {
            "debate_skipped": skip,
            "debate_skip_reason": skip_reason_tag,
            "consensus_metrics": {
                "strength": strength,
                "mean_score": mean_score,
                "dispersion": dispersion,
                "skip_threshold": config.consensus_skip_threshold,
                "confusion_threshold": config.confusion_skip_threshold,
            },
        }
    }


def debate_gate_router(state: ArenaState) -> str:
    """Route to 'debate' or 'skip' based on debate gate result."""
    if state["data"].get("debate_skipped"):
        return "skip"
    return "debate"


@node_logger()
async def check_stability(state: ArenaState) -> dict:
    """Compute divergence score after a debate round."""
    from cryptotrader.debate.convergence import compute_divergence

    analyses = state["data"].get("analyses", {})
    divergence = compute_divergence(analyses)
    scores = list(state.get("divergence_scores") or [])
    scores.append(divergence)
    logger.info("Divergence after round %d: %.4f (history: %s)", state["debate_round"], divergence, scores)
    return {"divergence_scores": scores}


def convergence_router(state: ArenaState) -> str:
    """Route to 'converged' or 'continue' based on divergence trend."""
    from cryptotrader.debate.convergence import check_convergence

    scores = state.get("divergence_scores") or []
    if state["debate_round"] >= state["max_debate_rounds"]:
        logger.info("Debate max rounds reached (%d) — converging", state["max_debate_rounds"])
        return "converged"
    threshold = state["metadata"].get("convergence_threshold", 0.1)
    if len(scores) >= 2 and check_convergence(scores[:-1], scores[-1], threshold=threshold):
        logger.info("Debate converged at round %d (threshold=%.2f)", state["debate_round"], threshold)
        return "converged"
    logger.info("Debate continuing to round %d", state["debate_round"] + 1)
    return "continue"


@node_logger()
async def bull_bear_debate(state: ArenaState) -> dict:
    """Run bull/bear adversarial debate."""
    from cryptotrader.config import load_config as _load_config
    from cryptotrader.debate.researchers import run_debate

    analyses = state["data"].get("analyses", {})
    _bbcfg = _load_config()
    _default_bb_model = _bbcfg.models.debate or _bbcfg.models.fallback
    model = state["metadata"].get("debate_model", _default_bb_model)
    rounds = state["metadata"].get("debate_rounds", 2)
    debate = await run_debate(analyses, rounds=rounds, model=model)
    return {"data": {"debate": debate}}


@node_logger()
async def judge_verdict(state: ArenaState) -> dict:
    """Judge evaluates bull/bear debate and issues verdict."""
    from cryptotrader.config import load_config as _load_config
    from cryptotrader.debate.researchers import judge_debate
    from cryptotrader.state import get_pair

    debate = state["data"]["debate"]
    # AI prompt — use display() form per FR-202 / T020
    pair = get_pair(state).display()
    _jcfg = _load_config()
    _default_judge_model = _jcfg.models.verdict or _jcfg.models.fallback
    model = state["metadata"].get("verdict_model", state["metadata"].get("debate_model", _default_judge_model))
    result = await judge_debate(debate, pair, model=model)
    return {
        "data": {
            "verdict": {
                "action": result["action"],
                "confidence": result["confidence"],
                "position_scale": result.get("position_scale", result["confidence"]),
                "divergence": 0.0,
                "reasoning": result["reasoning"],
            }
        }
    }
