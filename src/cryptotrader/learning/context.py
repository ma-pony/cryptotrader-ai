"""GSSC context engine — Gather, Select, Structure experience for agent prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.models import DecisionCommit, ExperienceMemory, ExperienceRule

from cryptotrader.learning.regime import regime_overlap

# Priority weights by packet type
_PRIORITY = {
    "forbidden_zone": 0.9,
    "success_pattern": 0.7,
    "insight": 0.5,
    "case": 0.4,
    "correction": 0.8,
}

_MATURITY_WEIGHT = {
    "rule": 1.0,
    "hypothesis": 0.6,
    "observation": 0.3,
}


@dataclass
class ContextPacket:
    content: str
    packet_type: str  # "forbidden_zone"|"success_pattern"|"insight"|"case"|"correction"
    regime_tags: list[str] = field(default_factory=list)
    maturity: str = "observation"
    priority: float = 0.5
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            self.token_estimate = _estimate_tokens(self.content)


def _estimate_tokens(text: str) -> int:
    """Estimate token count for mixed CJK/ASCII text.

    ASCII chars average ~4 chars/token, CJK chars average ~1.5 chars/token.
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii = len(text) - ascii_chars
    return ascii_chars // 4 + int(non_ascii / 1.5)


# ── Gather ──


def gather_packets(
    memory: ExperienceMemory | None,
    cases: list[DecisionCommit],
    correction: str = "",
) -> list[ContextPacket]:
    """Convert all experience sources into a flat list of ContextPackets."""
    packets: list[ContextPacket] = []

    if memory:
        packets.extend(_packets_from_rules(memory.forbidden_zones, "forbidden_zone"))
        packets.extend(_packets_from_rules(memory.success_patterns, "success_pattern"))
        packets.extend(_packets_from_insights(memory.strategic_insights))

    packets.extend(_packets_from_cases(cases))

    if correction:
        packets.append(
            ContextPacket(
                content=correction,
                packet_type="correction",
                priority=_PRIORITY["correction"],
                maturity="rule",
            )
        )

    return packets


def _packets_from_rules(rules: list[ExperienceRule], packet_type: str) -> list[ContextPacket]:
    """Convert ExperienceRules to ContextPackets."""
    packets: list[ContextPacket] = []
    for rule in rules:
        raw_tags = rule.conditions.get("regime_tags", [])
        tags = raw_tags if isinstance(raw_tags, list) else [raw_tags]
        maturity_label = _maturity_prefix(rule.maturity)
        content = f"{maturity_label}{rule.pattern} (rate={rule.rate:.0%}, n={rule.sample_count})"
        if rule.reason:
            content += f" — {rule.reason}"
        packets.append(
            ContextPacket(
                content=content,
                packet_type=packet_type,
                regime_tags=tags,
                maturity=rule.maturity,
                priority=_PRIORITY[packet_type],
            )
        )
    return packets


def _packets_from_insights(insights: list[str]) -> list[ContextPacket]:
    """Convert strategic insights to ContextPackets."""
    return [ContextPacket(content=i, packet_type="insight", priority=_PRIORITY["insight"]) for i in insights]


def _packets_from_cases(cases: list[DecisionCommit]) -> list[ContextPacket]:
    """Convert historical DecisionCommits to ContextPackets."""
    packets: list[ContextPacket] = []
    for dc in cases:
        verdict_action = dc.verdict.action if dc.verdict else "hold"
        outcome = f"pnl={dc.pnl:+.2f}" if dc.pnl is not None else "pending"
        content = f"{dc.pair} @ {dc.timestamp:%Y-%m-%d}: {verdict_action}, {outcome}"
        if dc.retrospective:
            content += f" — {dc.retrospective}"
        # Extract regime tags from snapshot for regime-aware scoring
        case_tags = _tags_from_snapshot(dc.snapshot_summary)
        packets.append(
            ContextPacket(
                content=content,
                packet_type="case",
                regime_tags=case_tags,
                priority=_PRIORITY["case"],
            )
        )
    return packets


def _tags_from_snapshot(snapshot_summary: dict) -> list[str]:
    """Extract regime tags from a commit's snapshot summary (best-effort)."""
    if not snapshot_summary:
        return []
    try:
        from cryptotrader.config import RegimeThresholdsConfig
        from cryptotrader.learning.regime import tag_regime

        return tag_regime(snapshot_summary, RegimeThresholdsConfig())
    except Exception:
        return []


def _maturity_prefix(maturity: str) -> str:
    """Return a confidence indicator prefix based on maturity level."""
    if maturity == "rule":
        return "[VERIFIED] "
    if maturity == "hypothesis":
        return "[LIKELY] "
    return "[TENTATIVE] "


# ── Select ──


def select_packets(
    packets: list[ContextPacket],
    regime_tags: list[str],
    token_budget: int,
) -> list[ContextPacket]:
    """Score, rank, and select packets within token budget."""
    scored = [(_score_packet(p, regime_tags), p) for p in packets]
    scored.sort(key=lambda x: x[0], reverse=True)

    selected: list[ContextPacket] = []
    remaining = token_budget
    for _, packet in scored:
        if packet.token_estimate > remaining:
            continue
        selected.append(packet)
        remaining -= packet.token_estimate
        if remaining <= 0:
            break
    return selected


def _score_packet(packet: ContextPacket, regime_tags: list[str]) -> float:
    """Compute relevance score for a packet."""
    overlap = regime_overlap(packet.regime_tags, regime_tags) if packet.regime_tags and regime_tags else 0.5
    maturity_w = _MATURITY_WEIGHT.get(packet.maturity, 0.3)
    return overlap * 0.5 + maturity_w * 0.3 + packet.priority * 0.2


# ── Structure ──


_PACKET_TYPE_TO_SECTION = {
    "forbidden_zone": "warnings",
    "success_pattern": "patterns",
    "correction": "patterns",
    "case": "cases",
    "insight": "insights",
}

_SECTION_HEADERS = [
    ("warnings", "⚠ Risk Warnings:"),
    ("patterns", "✓ Verified Patterns:"),
    ("cases", "📋 Historical Cases:"),
    ("insights", "💡 Strategic Insights:"),
]


def structure_experience(packets: list[ContextPacket]) -> str:
    """Organize selected packets into a structured prompt section."""
    if not packets:
        return ""

    sections: dict[str, list[str]] = {"warnings": [], "patterns": [], "cases": [], "insights": []}
    for p in packets:
        section = _PACKET_TYPE_TO_SECTION.get(p.packet_type)
        if section:
            sections[section].append(p.content)

    parts: list[str] = []
    for key, header in _SECTION_HEADERS:
        if sections[key]:
            parts.append(header)
            parts.extend(f"  - {item}" for item in sections[key])

    return "\n".join(parts)
