"""Tests for GSSC context engine."""

from __future__ import annotations

from datetime import UTC, datetime

from cryptotrader.learning.context import (
    ContextPacket,
    gather_packets,
    select_packets,
    structure_experience,
)
from cryptotrader.models import (
    AgentAnalysis,
    DecisionCommit,
    ExperienceMemory,
    ExperienceRule,
    TradeVerdict,
)


def _make_memory() -> ExperienceMemory:
    return ExperienceMemory(
        success_patterns=[
            ExperienceRule(
                pattern="RSI oversold + high vol = bounce",
                category="success_pattern",
                conditions={"regime_tags": ["high_vol"]},
                rate=0.75,
                sample_count=20,
                maturity="hypothesis",
                reason="Consistent mean reversion",
            ),
        ],
        forbidden_zones=[
            ExperienceRule(
                pattern="Never long during high funding",
                category="forbidden_zone",
                conditions={"regime_tags": ["high_funding"]},
                rate=0.80,
                sample_count=15,
                maturity="rule",
                reason="Crowded trade unwinds",
            ),
        ],
        strategic_insights=["Watch for divergence between OI and price"],
    )


def _make_case() -> DecisionCommit:
    return DecisionCommit(
        hash="test",
        parent_hash=None,
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        snapshot_summary={"price": 50000},
        analyses={
            "tech_agent": AgentAnalysis(
                agent_id="tech_agent",
                pair="BTC/USDT",
                direction="bullish",
                confidence=0.7,
                reasoning="test",
            ),
        },
        debate_rounds=0,
        verdict=TradeVerdict(action="long", confidence=0.7),
        pnl=1.5,
        retrospective="RSI signal was correct",
    )


class TestGatherPackets:
    def test_gathers_from_all_sources(self):
        memory = _make_memory()
        cases = [_make_case()]
        packets = gather_packets(memory, cases, correction="Reduce overconfidence")

        types = {p.packet_type for p in packets}
        assert "forbidden_zone" in types
        assert "success_pattern" in types
        assert "insight" in types
        assert "case" in types
        assert "correction" in types

    def test_gathers_without_memory(self):
        packets = gather_packets(None, [_make_case()])
        assert len(packets) == 1
        assert packets[0].packet_type == "case"

    def test_gathers_without_cases(self):
        packets = gather_packets(_make_memory(), [])
        types = {p.packet_type for p in packets}
        assert "case" not in types
        assert "forbidden_zone" in types

    def test_empty_inputs(self):
        packets = gather_packets(None, [])
        assert packets == []


class TestSelectPackets:
    def test_respects_token_budget(self):
        packets = [
            ContextPacket(content="a" * 400, packet_type="forbidden_zone", priority=0.9, token_estimate=100),
            ContextPacket(content="b" * 400, packet_type="success_pattern", priority=0.7, token_estimate=100),
            ContextPacket(content="c" * 400, packet_type="case", priority=0.4, token_estimate=100),
        ]
        selected = select_packets(packets, ["high_vol"], token_budget=200)
        assert len(selected) == 2

    def test_prioritizes_higher_score(self):
        packets = [
            ContextPacket(content="low", packet_type="case", priority=0.4, regime_tags=[]),
            ContextPacket(
                content="high", packet_type="forbidden_zone", priority=0.9, regime_tags=["high_vol"], maturity="rule"
            ),
        ]
        selected = select_packets(packets, ["high_vol"], token_budget=1000)
        # Higher priority packet should come first
        assert selected[0].packet_type == "forbidden_zone"

    def test_empty_budget(self):
        packets = [ContextPacket(content="test", packet_type="case", token_estimate=100)]
        selected = select_packets(packets, [], token_budget=0)
        assert selected == []


class TestStructureExperience:
    def test_produces_sections(self):
        packets = [
            ContextPacket(content="Danger zone", packet_type="forbidden_zone"),
            ContextPacket(content="Good pattern", packet_type="success_pattern"),
            ContextPacket(content="Past trade", packet_type="case"),
            ContextPacket(content="Key insight", packet_type="insight"),
        ]
        result = structure_experience(packets)
        assert "Risk Warnings" in result
        assert "Danger zone" in result
        assert "Verified Patterns" in result
        assert "Good pattern" in result
        assert "Historical Cases" in result
        assert "Past trade" in result
        assert "Strategic Insights" in result
        assert "Key insight" in result

    def test_empty_packets(self):
        assert structure_experience([]) == ""

    def test_single_section(self):
        packets = [ContextPacket(content="Warning", packet_type="forbidden_zone")]
        result = structure_experience(packets)
        assert "Risk Warnings" in result
        assert "Verified Patterns" not in result


class TestTokenEstimation:
    def test_ascii_text(self):
        p = ContextPacket(content="Hello world test", packet_type="case")
        # 16 ASCII chars → 16 // 4 = 4
        assert p.token_estimate == 4

    def test_chinese_text(self):
        p = ContextPacket(content="这是中文测试", packet_type="case")
        # 6 CJK chars → int(6 / 1.5) = 4
        assert p.token_estimate == 4

    def test_mixed_text(self):
        p = ContextPacket(content="RSI超卖信号", packet_type="case")
        # 3 ASCII + 4 CJK → 3//4 + int(4/1.5) = 0 + 2 = 2
        assert p.token_estimate == 2


class TestCaseRegimeTags:
    def test_cases_get_regime_tags(self):
        """Cases with high volatility in snapshot should get high_vol tag."""
        dc = DecisionCommit(
            hash="test",
            parent_hash=None,
            timestamp=datetime.now(UTC),
            pair="BTC/USDT",
            snapshot_summary={"volatility": 0.04, "funding_rate": 0.0005},
            analyses={},
            debate_rounds=0,
            verdict=TradeVerdict(action="hold", confidence=0.5),
            pnl=1.0,
        )
        packets = gather_packets(None, [dc])
        assert len(packets) == 1
        assert "high_vol" in packets[0].regime_tags
        assert "high_funding" in packets[0].regime_tags


class TestFullGSSCPipeline:
    def test_end_to_end(self):
        """Full pipeline: gather → select → structure."""
        memory = _make_memory()
        cases = [_make_case()]
        correction = "Watch for false breakouts"

        packets = gather_packets(memory, cases, correction)
        selected = select_packets(packets, ["high_vol", "high_funding"], token_budget=5000)
        result = structure_experience(selected)

        assert "Never long during high funding" in result
        assert "RSI oversold" in result
        assert len(result) > 0
