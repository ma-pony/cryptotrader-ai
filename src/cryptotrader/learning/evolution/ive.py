"""spec 018 — IVE (Invest-Verdict-Exit) failure classification.

FR-Z15: classify_case(case: CaseRecord, llm_callable=None) -> FailureClassification
Single LLM call with 5 diagnostic questions (Decision 3).
FR-Z16: output FailureClassification dataclass
FR-Z17: run on every case (win and loss)
FR-Z18: LLM failure -> failure_type="noise" + warning log
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from cryptotrader.agents.skills.schema import CaseRecord

logger = logging.getLogger(__name__)

FailureType = Literal["implementation", "fundamental", "noise"]

# ── FailureClassification dataclass ───────────────────────────────────────────


@dataclass
class FailureClassification:
    """IVE 失败分类结果。"""

    case_id: str
    failure_type: FailureType = "noise"
    reasoning: str = ""
    confidence: float = 0.0
    diagnostic_answers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "diagnostic_answers": self.diagnostic_answers,
        }


# ── Prompt 模板 ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a crypto trading failure diagnostician. Given a trade case, "
    "answer 5 diagnostic questions and classify the failure type.\n"
    "Output ONLY valid JSON matching the schema below.\n"
)

_USER_TEMPLATE = """\
## Trade Case
- Pair: {pair}
- Verdict Action: {verdict_action}
- Final PnL: {final_pnl}
- Regime Tags: {regime_tags}

## Agent Analyses Summary
{agent_analyses}

## Trade Execution
{trade_execution}

## Same-Regime Context (recent cases)
{same_regime_context}

## 5 Diagnostic Questions
1. Did other rules under the same regime also lose? (yes/no/uncertain)
2. Were entry/exit prices within reasonable range? (yes/no/uncertain)
3. Did the trade hit stop-loss? (yes/no/uncertain)
4. Does the loss match the rule's invalidation conditions? (yes/no/uncertain)
5. Was position size too large? (yes/no/uncertain)

Output JSON:
{{
  "diagnostic_answers": ["answer1","answer2","answer3","answer4","answer5"],
  "reasoning": "brief explanation of root cause",
  "failure_type": "implementation|fundamental|noise",
  "confidence": 0.0
}}
"""

# ── JSON 提取辅助 ──────────────────────────────────────────────────────────────


def _extract_json(text: str) -> dict | None:
    """Extract JSON from LLM output using balanced-brace algorithm."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _noise_result(case_id: str) -> FailureClassification:
    """Default noise result for LLM failure."""
    return FailureClassification(
        case_id=case_id,
        failure_type="noise",
        reasoning="LLM classification unavailable",
        confidence=0.0,
        diagnostic_answers=["uncertain"] * 5,
    )


def _format_trade_execution(te: dict | None) -> str:
    if not te:
        return "(no trade execution data)"
    parts = []
    for k, v in te.items():
        parts.append(f"- {k}: {v}")
    return "\n".join(parts)


def _format_agent_analyses(aa: dict[str, str]) -> str:
    if not aa:
        return "(no agent analyses)"
    parts = []
    for agent_id, analysis in aa.items():
        # Truncate long analyses
        truncated = analysis[:200] + "..." if len(analysis) > 200 else analysis
        parts.append(f"[{agent_id}] {truncated}")
    return "\n".join(parts)


def _parse_llm_response(raw: str, case_id: str, retry_fn: Callable[[], str] | None = None) -> FailureClassification:
    """Parse LLM JSON response; retry once on failure."""
    parsed = _extract_json(raw)
    if parsed is None and retry_fn is not None:
        logger.warning("IVE: JSON parse failed on first attempt, retrying for case %s", case_id)
        try:
            raw2 = retry_fn()
            parsed = _extract_json(raw2)
        except Exception:
            logger.warning("IVE: retry also failed for case %s", case_id, exc_info=True)
            return _noise_result(case_id)

    if parsed is None:
        logger.warning("IVE: could not parse JSON from LLM output for case %s", case_id)
        return _noise_result(case_id)

    failure_type_raw = parsed.get("failure_type", "noise")
    if failure_type_raw not in ("implementation", "fundamental", "noise"):
        failure_type_raw = "noise"

    answers = parsed.get("diagnostic_answers", [])
    if not isinstance(answers, list):
        answers = []
    # Pad/truncate to exactly 5
    answers = (answers + ["uncertain"] * 5)[:5]

    return FailureClassification(
        case_id=case_id,
        failure_type=failure_type_raw,
        reasoning=str(parsed.get("reasoning", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        diagnostic_answers=answers,
    )


# ── 主函数 ────────────────────────────────────────────────────────────────────


async def classify_case(
    case: CaseRecord,
    llm_callable: Callable[[str, str], str] | None = None,
    same_regime_cases: list[CaseRecord] | None = None,
) -> FailureClassification:
    """Classify a trade case using IVE 5 diagnostic questions.

    spec 020a FR-Z10: converted to async def; default LLM path uses await llm.ainvoke().
    When llm_callable is provided (e.g. in tests), it is called synchronously as before.

    Args:
        case: The CaseRecord to classify.
        llm_callable: Optional sync callable(system_prompt, user_prompt) -> str.
            If None, uses project default LLM via await llm.ainvoke() (FR-Z10).
        same_regime_cases: Optional list of recent cases with same regime_tags for context.

    Returns:
        FailureClassification with failure_type in {implementation, fundamental, noise}.

    FR-Z18: LLM failure returns failure_type="noise" + warning log.
    """
    case_id = case.cycle_id

    # Build same-regime context
    same_regime_ctx = "(no same-regime context)"
    if same_regime_cases:
        lines = []
        for c in same_regime_cases[:3]:
            pnl_str = f"{c.final_pnl:.2f}" if c.final_pnl is not None else "N/A"
            lines.append(f"- {c.cycle_id}: action={c.verdict_action}, pnl={pnl_str}")
        same_regime_ctx = "\n".join(lines)

    user_prompt = _USER_TEMPLATE.format(
        pair=case.pair,
        verdict_action=case.verdict_action,
        final_pnl=f"{case.final_pnl:.2f}" if case.final_pnl is not None else "N/A",
        regime_tags=getattr(case, "regime_tags", []),
        agent_analyses=_format_agent_analyses(case.agent_analyses),
        trade_execution=_format_trade_execution(case.trade_execution),
        same_regime_context=same_regime_ctx,
    )

    try:
        if llm_callable is not None:
            # Test/override path: sync callable provided by caller
            raw = llm_callable(_SYSTEM_PROMPT, user_prompt)
            result = _parse_llm_response(
                raw,
                case_id,
                retry_fn=lambda: llm_callable(_SYSTEM_PROMPT, user_prompt),  # type: ignore[union-attr]
            )
        else:
            # spec 020a FR-Z10: default async path — await llm.ainvoke (no sync blocking)
            raw = await _async_llm_call(_SYSTEM_PROMPT, user_prompt)
            result = _parse_llm_response(raw, case_id, retry_fn=None)

        # spec 020a FR-Z22: record success to IveMetricsAggregator
        _record_ive_metric(success=True)
        return result

    except Exception:
        logger.warning(
            "IVE: LLM call failed for case %s, returning noise classification",
            case_id,
            exc_info=True,
        )
        # spec 020a FR-Z22: record failure to IveMetricsAggregator
        _record_ive_metric(success=False)
        return _noise_result(case_id)


async def _async_llm_call(system_prompt: str, user_prompt: str) -> str:
    """Default async LLM call path (spec 020a FR-Z10: await llm.ainvoke)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from cryptotrader.agents.base import create_llm

    llm = create_llm()
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    result = await llm.ainvoke(messages)
    return str(result.content) if hasattr(result, "content") else str(result)


def _record_ive_metric(success: bool) -> None:
    """Push success/failure to IveMetricsAggregator (spec 020a FR-Z18, non-blocking)."""
    try:
        from cryptotrader.observability.ive_metrics import get_ive_metrics_aggregator

        get_ive_metrics_aggregator().record(success=success)
    except Exception:
        pass
