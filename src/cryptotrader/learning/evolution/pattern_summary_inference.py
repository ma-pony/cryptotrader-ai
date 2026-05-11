"""pattern_summary_inference — LLM 浓缩 agent thesis 为 pattern description + body.

由 distill_patterns 在 cold-start 时调用：把 source_cycles 内对应 agent 的
原始自然语言分析（"price below SMA20 with negative MACD..." 等长篇文本）
压缩成两段人类可读结构：

- description：1 句话核心机制
- body：4-6 行 Markdown，含触发信号 / 适用 regime / 失效条件

LLM 失败 / parse 失败 / 重试 1 次仍失败 → 回退到模板（不阻塞 cold-start）。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

# ── 默认值（LLM 失败时回退）─────────────────────────────────────────────────────


def _default_description(agent: str, applied_text: str, n_cases: int) -> str:
    return f"Auto-distilled pattern: {applied_text} (from {n_cases} cases, agent={agent})"


def _default_body(applied_text: str, n_cases: int, source_cycles: list[str]) -> str:
    return (
        f"# {applied_text}\n\n"
        f"Auto-distilled from {n_cases} cases.\n\n"
        f"Source cycles (first 5): {source_cycles[:5]}\n"
    )


# ── Prompt ────────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = (
    "You are a crypto trading pattern librarian. Given 3 verbatim excerpts of "
    "what an analysis agent said when it referenced the same pattern label "
    "across multiple cycles, distill the pattern into a structured record.\n"
    "Output ONLY valid JSON matching the schema. No prose outside the JSON.\n"
)


_USER_TEMPLATE = """\
## Pattern label
{applied_text}

## Originating agent
{agent}

## Cycles count
{n_cases}

## Source-cycle agent excerpts (verbatim)
{excerpts}

## Output schema
{{
  "description": "one short sentence (<=140 chars) stating the pattern's core thesis",
  "trigger":     "concrete signal(s) the agent looks for (1-2 sentences)",
  "regime":      "market regime(s) where this pattern applies",
  "failure":     "what would invalidate this pattern"
}}
"""


def _build_prompt(
    agent: str,
    applied_text: str,
    n_cases: int,
    excerpts: list[str],
) -> str:
    excerpt_block = "\n\n".join(
        f"### excerpt {i+1}\n{e.strip()[:600]}" for i, e in enumerate(excerpts) if e
    )
    if not excerpt_block:
        excerpt_block = "(no agent text available — use the label only)"
    return _USER_TEMPLATE.format(
        applied_text=applied_text,
        agent=agent,
        n_cases=n_cases,
        excerpts=excerpt_block,
    )


# ── LLM call + parse ──────────────────────────────────────────────────────────


def _call_llm(prompt: str, llm_callable: Callable | None) -> str | None:
    """Call LLM, return raw text. None on failure."""
    if llm_callable is not None:
        try:
            result = llm_callable(_SYSTEM_PROMPT + "\n\n" + prompt)
            if hasattr(result, "content"):
                return str(result.content)
            return str(result)
        except Exception as exc:
            logger.warning("pattern_summary LLM callable failed: %s", exc, exc_info=True)
            return None

    try:
        from cryptotrader.agents.base import create_llm

        llm = create_llm("")
        # langchain ChatModel: pass system + user as a single prompt; OK for non-chat models too
        result = llm.invoke(_SYSTEM_PROMPT + "\n\n" + prompt)
        if hasattr(result, "content"):
            return str(result.content)
        return str(result)
    except Exception as exc:
        logger.warning("pattern_summary default LLM failed: %s", exc, exc_info=True)
        return None


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(raw: str) -> dict | None:
    """Try to pull a JSON object out of raw LLM text."""
    if not raw:
        return None
    # 1. fenced block
    m = _JSON_FENCE_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 2. first { ... last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _shape_body(
    applied_text: str,
    description: str,
    parsed: dict,
    n_cases: int,
    source_cycles: list[str],
) -> str:
    """Compose the human-readable body markdown from the parsed LLM fields."""
    trigger = (parsed.get("trigger") or "").strip()
    regime = (parsed.get("regime") or "").strip()
    failure = (parsed.get("failure") or "").strip()

    lines = [f"# {applied_text}", "", f"_{description}_", ""]
    if trigger:
        lines.extend(["## Trigger", trigger, ""])
    if regime:
        lines.extend(["## Applicable regime", regime, ""])
    if failure:
        lines.extend(["## Invalidation", failure, ""])
    lines.extend(
        [
            "## Provenance",
            f"Auto-distilled from {n_cases} cycles. Summary inferred by LLM "
            f"from agent thesis excerpts.",
            "",
            f"Source cycles (first 5): {source_cycles[:5]}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def infer_pattern_summary(
    agent: str,
    applied_text: str,
    case_data_list: list[dict],
    llm_callable: Callable | None = None,
) -> tuple[str, str]:
    """Return (description, body) for a freshly distilled PatternRecord.

    Args:
        agent:          originating agent id (tech / chain / news / macro).
        applied_text:   the LLM's own short label for this pattern.
        case_data_list: list of dicts; each may have an
                        ``agent_analyses_snippet`` field carrying the relevant
                        agent's verbatim text for that cycle.
        llm_callable:   optional callable(prompt) -> str / AIMessage.

    Soft-fail: any LLM / parse error returns the auto-template description+body
    so cold-start never blocks.
    """
    n = len(case_data_list)
    source_cycles = [c.get("cycle_id", "") for c in case_data_list if c.get("cycle_id")]

    excerpts: list[str] = []
    for c in case_data_list:
        snippet = (c.get("agent_analyses_snippet") or "").strip()
        if snippet:
            excerpts.append(snippet)
        if len(excerpts) >= 3:
            break

    # No agent text -> fall back to template (cheap path, no LLM call).
    if not excerpts:
        logger.debug(
            "pattern_summary: no agent excerpts for %s/%s, using template", agent, applied_text
        )
        return (
            _default_description(agent, applied_text, n),
            _default_body(applied_text, n, source_cycles),
        )

    prompt = _build_prompt(agent, applied_text, n, excerpts)

    # First attempt
    raw = _call_llm(prompt, llm_callable)
    parsed = _extract_json(raw or "")
    if parsed is None:
        # Retry once
        logger.info("pattern_summary: first attempt failed for %s, retrying", applied_text)
        raw2 = _call_llm(prompt, llm_callable)
        parsed = _extract_json(raw2 or "")

    if parsed is None:
        logger.warning(
            "pattern_summary: LLM failed twice for %s/%s, falling back to template",
            agent,
            applied_text,
        )
        return (
            _default_description(agent, applied_text, n),
            _default_body(applied_text, n, source_cycles),
        )

    description = (parsed.get("description") or "").strip()
    if not description:
        description = _default_description(agent, applied_text, n)
    # 280-char hard cap so a verbose LLM can't blow up the rules-grid UI.
    if len(description) > 280:
        description = description[:277] + "..."

    body = _shape_body(applied_text, description, parsed, n, source_cycles)
    return description, body
