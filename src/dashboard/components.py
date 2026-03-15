"""Shared rendering components for the Dashboard.

All reusable Streamlit render functions live here so that individual page
modules can import them without duplicating code.

Each function accepts real domain model objects (not raw dicts) and renders
them using Streamlit widgets.  Functions have no return value except for
render_pagination_controls which returns (offset, limit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    from cryptotrader.models import AgentAnalysis, ConsensusMetrics, GateResult, NodeTraceEntry, TradeVerdict

# Nodes that belong to the optional debate path — rendered as gray/dashed when absent.
_DEBATE_NODE_PREFIXES = ("debate_round",)

# Maximum number of grid columns for agent analysis cards.
_MAX_AGENT_COLUMNS = 4

# Direction-to-emoji mapping for quick visual scanning.
_DIRECTION_EMOJI: dict[str, str] = {
    "bullish": "🟢",
    "bearish": "🔴",
    "neutral": "🟡",
}

# verdict_source display labels and badge styles.
_SOURCE_LABELS: dict[str, str] = {
    "ai": "AI Verdict",
    "weighted": "Weighted Verdict (downgraded)",
    "hold_all_mock": "Mock Hold (all agents mocked)",
}


# ---------------------------------------------------------------------------
# render_pagination_controls
# ---------------------------------------------------------------------------


def render_pagination_controls(
    total: int,
    page_size: int = 20,
    *,
    key: str = "page",
) -> tuple[int, int]:
    """Render page navigation controls and return (offset, limit).

    Displays a numeric page selector using st.number_input.  The caller
    passes the returned offset and limit to the data loading layer.

    Args:
        total:     Total number of records in the result set.
        page_size: Number of records per page (default 20).
        key:       Unique Streamlit widget key, used to namespace the widget so
                   multiple paginators on the same page do not conflict.

    Returns:
        A (offset, limit) tuple where offset = (page - 1) * page_size and
        limit = page_size.
    """
    max_page = max(1, (total + page_size - 1) // page_size) if total > 0 else 1
    page = int(
        st.number_input(
            "Page",
            min_value=1,
            max_value=max_page,
            value=1,
            step=1,
            key=key,
        )
    )
    offset = (page - 1) * page_size
    return offset, page_size


# ---------------------------------------------------------------------------
# render_expandable_text
# ---------------------------------------------------------------------------


def render_expandable_text(
    label: str,
    text: str,
    *,
    preview_chars: int = 200,
) -> None:
    """Render text with the first preview_chars characters shown directly.

    When the text exceeds preview_chars, the remainder is placed inside an
    st.expander so the UI stays compact by default.

    Args:
        label:         Header / expander title for the text block.
        text:          The full text to render.
        preview_chars: Number of characters to show outside the expander.
    """
    if len(text) <= preview_chars:
        st.write(text)
        return

    preview = text[:preview_chars]
    remainder = text[preview_chars:]
    st.write(preview + "…")
    with st.expander(label):
        st.write(remainder)


# ---------------------------------------------------------------------------
# render_agent_analysis_grid
# ---------------------------------------------------------------------------


def render_agent_analysis_grid(
    analyses: dict[str, AgentAnalysis],
    *,
    columns: int | None = None,
) -> None:
    """Render a responsive grid of agent analysis cards.

    Each agent gets one card showing direction, confidence, data sufficiency,
    and a collapsible reasoning section.  Cards with data_sufficiency == 'low'
    display a ⚠️ warning icon next to the agent name.

    Args:
        analyses: Mapping of agent_id → AgentAnalysis.
        columns:  Number of columns override.  When None the grid uses the
                  lesser of len(analyses) and _MAX_AGENT_COLUMNS.
    """
    if not analyses:
        st.write("No agent analyses available.")
        return

    n_cols = columns if columns is not None else min(len(analyses), _MAX_AGENT_COLUMNS)
    cols = st.columns(n_cols)

    for idx, (agent_id, analysis) in enumerate(analyses.items()):
        col = cols[idx % n_cols]
        with col:
            suffix = " ⚠️" if analysis.data_sufficiency == "low" else ""
            direction_emoji = _DIRECTION_EMOJI.get(analysis.direction, "")
            st.subheader(f"{agent_id}{suffix}")
            st.write(f"{direction_emoji} **{analysis.direction.upper()}** | conf: {analysis.confidence:.0%}")
            st.write(f"Data sufficiency: `{analysis.data_sufficiency}`")
            if analysis.key_factors:
                st.write("**Key factors:** " + "; ".join(analysis.key_factors))
            if analysis.risk_flags:
                st.write("**Risk flags:** " + "; ".join(analysis.risk_flags))
            render_expandable_text(f"{agent_id} reasoning", analysis.reasoning)


# ---------------------------------------------------------------------------
# render_node_trace_pipeline
# ---------------------------------------------------------------------------


def render_node_trace_pipeline(node_trace: list[NodeTraceEntry]) -> None:
    """Render a horizontal pipeline view of node execution trace.

    Each node is displayed as a box with its name and duration in ms.
    Debate round nodes that were skipped (duration == 0) are shown with a
    gray dashed style to indicate they did not execute.

    Args:
        node_trace: List of NodeTraceEntry objects in execution order.
    """
    if not node_trace:
        st.write("No node trace available.")
        return

    n_cols = len(node_trace)
    cols = st.columns(n_cols)

    for idx, entry in enumerate(node_trace):
        col = cols[idx]
        with col:
            is_debate_node = any(entry.node.startswith(p) for p in _DEBATE_NODE_PREFIXES)
            skipped = is_debate_node and entry.duration_ms == 0
            if skipped:
                # Gray dashed styling via markdown
                st.markdown(
                    f"<div style='border:1px dashed gray;padding:4px;color:gray'>"
                    f"<b>{entry.node}</b><br/>⊘ skipped</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='border:1px solid #4CAF50;padding:4px'>"
                    f"<b>{entry.node}</b><br/>{entry.duration_ms} ms</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# render_verdict_section
# ---------------------------------------------------------------------------


def render_verdict_section(
    verdict: TradeVerdict,
    verdict_source: str,
) -> None:
    """Render the trade verdict with a visual badge indicating the source.

    The badge distinguishes AI verdicts, weighted-downgrade verdicts, and
    hold_all_mock verdicts so operators can quickly identify machine-generated
    vs rule-based decisions.

    Args:
        verdict:        The TradeVerdict domain object.
        verdict_source: One of "ai", "weighted", or "hold_all_mock".
    """
    source_label = _SOURCE_LABELS.get(verdict_source, verdict_source)
    st.markdown(f"**Source:** `{source_label}`")
    st.write(f"**Action:** `{verdict.action}` | Confidence: {verdict.confidence:.0%}")
    if verdict.position_scale > 0:
        st.write(f"**Position scale:** {verdict.position_scale:.0%}")
    if verdict.reasoning:
        render_expandable_text("Verdict reasoning", verdict.reasoning)
    if verdict.thesis:
        st.write(f"**Thesis:** {verdict.thesis}")
    if verdict.invalidation:
        st.write(f"**Invalidation:** {verdict.invalidation}")


# ---------------------------------------------------------------------------
# render_risk_gate_section
# ---------------------------------------------------------------------------


def render_risk_gate_section(risk_gate: GateResult) -> None:
    """Render the risk gate result in green (passed) or red (rejected).

    Passed gates display a success banner; rejected gates show the check
    that failed (rejected_by) and the human-readable reason.

    Args:
        risk_gate: The GateResult domain object.
    """
    if risk_gate.passed:
        st.success("Risk gate: PASSED ✓")
    else:
        st.error(f"Risk gate: REJECTED ✗  |  Check: `{risk_gate.rejected_by}`\n\n{risk_gate.reason}")


# ---------------------------------------------------------------------------
# render_consensus_metrics_chart
# ---------------------------------------------------------------------------


def render_consensus_metrics_chart(
    consensus_metrics: ConsensusMetrics,
    analyses: dict[str, AgentAnalysis],
) -> None:
    """Render agent score bar chart with consensus summary caption.

    Builds a simple score mapping from confidence (bullish positive, bearish
    negative, neutral zero) and plots it with st.bar_chart.  A st.caption
    below the chart shows mean score, dispersion, and consensus strength.

    Args:
        consensus_metrics: The ConsensusMetrics snapshot.
        analyses:          Agent analyses used to extract per-agent scores.
    """
    scores: dict[str, float] = {}
    for agent_id, analysis in analyses.items():
        if analysis.direction == "bullish":
            scores[agent_id] = analysis.confidence
        elif analysis.direction == "bearish":
            scores[agent_id] = -analysis.confidence
        else:
            scores[agent_id] = 0.0

    if scores:
        st.bar_chart(scores)
    else:
        st.bar_chart({})

    st.caption(
        f"mean={consensus_metrics.mean_score:.3f}  |  "
        f"dispersion (stdev)={consensus_metrics.dispersion:.3f}  |  "
        f"strength={consensus_metrics.strength:.3f}"
    )


# ---------------------------------------------------------------------------
# render_debate_section
# ---------------------------------------------------------------------------


def render_debate_section(
    debate_rounds: int,
    challenges: list[dict],
    debate_skip_reason: str,
    consensus_metrics: ConsensusMetrics | None,
) -> None:
    """Render the debate section, including skip information when applicable.

    When debate was skipped, the section shows the skip reason along with a
    comparison of the threshold value vs the actual consensus strength or
    confusion dispersion.  When debate ran, rounds and challenge points are
    displayed.

    Args:
        debate_rounds:      Number of debate rounds that executed (0 = skipped).
        challenges:         List of challenge dicts from the debate (may be empty).
        debate_skip_reason: "consensus", "confusion", or "" for no skip.
        consensus_metrics:  ConsensusMetrics snapshot (may be None for old records).
    """
    st.write(f"**Debate rounds:** {debate_rounds}")

    if debate_skip_reason:
        if debate_skip_reason == "consensus":
            label = "Debate skipped — consensus threshold met"
            if consensus_metrics is not None:
                st.write(
                    f"{label}: strength={consensus_metrics.strength:.3f} "
                    f"≥ threshold={consensus_metrics.skip_threshold:.3f}"
                )
            else:
                st.write(f"{label} (consensus)")
        elif debate_skip_reason == "confusion":
            label = "Debate skipped — shared confusion detected"
            if consensus_metrics is not None:
                st.write(
                    f"{label}: dispersion={consensus_metrics.dispersion:.3f} "
                    f"≤ confusion_threshold={consensus_metrics.confusion_threshold:.3f}"
                )
            else:
                st.write(f"{label} (confusion)")
        else:
            st.write(f"Debate skipped: {debate_skip_reason}")
        return

    if challenges:
        st.write(f"**Challenges ({len(challenges)}):**")
        for ch in challenges:
            round_num = ch.get("round", "?")
            challenger = ch.get("challenger", "?")
            point = ch.get("point", str(ch))
            st.write(f"- Round {round_num} | {challenger}: {point}")
    else:
        st.write("No challenges recorded.")


# ---------------------------------------------------------------------------
# render_experience_memory_section
# ---------------------------------------------------------------------------


def render_experience_memory_section(experience_memory: dict[str, Any]) -> None:
    """Render the injected experience memory for a decision.

    Displays success patterns, forbidden zones, and strategic insights from
    the GSSC pipeline's structured experience memory.

    Args:
        experience_memory: Dict with optional keys "success_patterns",
                           "forbidden_zones", and "strategic_insights".
    """
    if not experience_memory:
        st.write("No experience memory injected.")
        return

    success_patterns = experience_memory.get("success_patterns", [])
    forbidden_zones = experience_memory.get("forbidden_zones", [])
    strategic_insights = experience_memory.get("strategic_insights", [])

    if success_patterns:
        st.write("**Success patterns:**")
        for rule in success_patterns:
            pattern = rule.get("pattern", str(rule)) if isinstance(rule, dict) else str(rule)
            rate = rule.get("rate", 0) if isinstance(rule, dict) else 0
            maturity = rule.get("maturity", "") if isinstance(rule, dict) else ""
            st.markdown(f"- {pattern} _(rate={rate:.0%}, maturity={maturity})_")

    if forbidden_zones:
        st.write("**Forbidden zones:**")
        for rule in forbidden_zones:
            pattern = rule.get("pattern", str(rule)) if isinstance(rule, dict) else str(rule)
            rate = rule.get("rate", 0) if isinstance(rule, dict) else 0
            maturity = rule.get("maturity", "") if isinstance(rule, dict) else ""
            st.markdown(f"- {pattern} _(loss rate={rate:.0%}, maturity={maturity})_")

    if strategic_insights:
        st.write("**Strategic insights:**")
        for insight in strategic_insights:
            st.markdown(f"- {insight}")
