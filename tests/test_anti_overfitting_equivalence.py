"""Tests for US2: 4 层防过拟合算法等价验证（FR-010）。

验证新系统 learning/memory.py 中的 4 层防过拟合算法与原 reflect.py 保持等价。
TDD: 先 FAIL，实现 learning/memory.py 后 GREEN。

4 层：
  L1 — regime-aware 胜率统计（仅同 regime 历史样本）
  L2 — 最少样本量门槛（< N case 不晋升 maturity）
  L3 — 全局 vs 区段差距（区段必须显著优于全局基线）
  L4 — 对手验证（forbidden 专属：相反方向有亏损证据）
"""

from __future__ import annotations


def _make_records(
    n: int,
    regime_tags: list[str],
    win_rate: float = 0.7,
    funding_rate: float = 0.0003,
    volatility: float = 0.02,
) -> list[dict]:
    """创建测试用历史记录集合。"""
    records = []
    for i in range(n):
        pnl = 100.0 if i < int(n * win_rate) else -50.0
        records.append(
            {
                "date": f"2026-05-{i + 1:02d} 10:00",
                "direction": "bullish",
                "confidence": 0.7,
                "reasoning": "Test reasoning",
                "key_factors": ["rsi_oversold"],
                "pnl": pnl,
                "verdict_action": "long",
                "price": 60000 + i * 100,
                "volatility": volatility,
                "funding_rate": funding_rate,
                "regime_tags": regime_tags,
            }
        )
    return records


class TestL1RegimeAware:
    """L1: regime-aware 胜率统计（仅同 regime 历史样本）。"""

    def test_regime_filter_uses_only_matching_samples(self):
        """_filter_records_by_regime 应仅返回 regime 匹配的样本。"""
        from cryptotrader.learning.memory import _filter_records_by_regime

        # 10 条 high_funding 样本 + 5 条无 regime 样本
        high_funding_records = _make_records(10, regime_tags=["high_funding"], funding_rate=0.0004)
        other_records = _make_records(5, regime_tags=[], funding_rate=0.0001)
        all_records = high_funding_records + other_records

        # 过滤 high_funding regime
        filtered = _filter_records_by_regime(all_records, regime_condition=["high_funding"])
        assert len(filtered) <= 10, "应只返回 high_funding 样本，不超过 10 条"
        # 至少过滤掉一些非匹配样本
        assert len(filtered) < len(all_records), "regime 过滤应减少样本数"

    def test_empty_regime_condition_uses_all_records(self):
        """regime_condition 为空时应使用所有样本（无限制）。"""
        from cryptotrader.learning.memory import _filter_records_by_regime

        records = _make_records(10, regime_tags=["high_funding"])
        filtered = _filter_records_by_regime(records, regime_condition=[])
        assert len(filtered) == len(records), "无 regime 限制时应返回全部样本"


class TestL2MinSampleThreshold:
    """L2: 最少样本量门槛（< N case 不晋升 maturity）。"""

    def test_too_few_samples_stays_observed(self, tmp_path):
        """样本数 < 5 时不晋升（保持 observed）。"""
        from cryptotrader.agents.skills.schema import PatternRecord, PnLTrack
        from cryptotrader.learning.evolution.fsm import evaluate_transitions

        pattern = PatternRecord(
            name="test",
            agent="tech",
            description="Test",
            body="",
            pnl_track=PnLTrack(cases=3, wins=3, win_rate=1.0, avg_pnl=100.0),
            maturity="observed",
        )
        result = evaluate_transitions(pattern)
        # 3 cases < 5 minimum → 不晋升
        assert result is None, f"样本不足时应保持 observed (no transition)，实际: {result}"

    def test_sufficient_samples_can_advance(self, tmp_path):
        """样本数 ≥ 5，win_rate 合格时可晋升。"""
        from cryptotrader.agents.skills.schema import PatternRecord, PnLTrack
        from cryptotrader.learning.evolution.fsm import evaluate_transitions

        pattern = PatternRecord(
            name="test",
            agent="tech",
            description="Test",
            body="",
            pnl_track=PnLTrack(cases=6, wins=5, win_rate=0.83, avg_pnl=100.0),
            maturity="observed",
        )
        result = evaluate_transitions(pattern)
        assert result is not None, "应晋升，实际无转移"
        assert result.maturity == "probationary", f"应到 probationary，实际: {result.maturity}"


class TestL3GlobalVsSegment:
    """L3: 全局 vs 区段差距（区段必须显著优于全局基线）。"""

    def test_segment_significantly_better_than_global(self):
        """区段胜率显著高于全局基线时通过 L3 验证。"""
        from cryptotrader.learning.memory import _check_segment_vs_global

        # 全局基线 50%，区段 80%
        result = _check_segment_vs_global(
            segment_win_rate=0.80,
            global_win_rate=0.50,
            min_delta=0.15,
        )
        assert result is True, "区段显著优于全局时应通过 L3"

    def test_segment_not_significantly_better_fails_l3(self):
        """区段胜率与全局相近时不通过 L3。"""
        from cryptotrader.learning.memory import _check_segment_vs_global

        # 全局基线 50%，区段 55%（差距不足）
        result = _check_segment_vs_global(
            segment_win_rate=0.55,
            global_win_rate=0.50,
            min_delta=0.15,
        )
        assert result is False, "区段优势不显著时应不通过 L3"


class TestL4AdversarialVerification:
    """L4: 对手验证（forbidden 专属）：相反方向有亏损证据。"""

    def test_forbidden_pattern_verified_by_adversarial_evidence(self):
        """forbidden pattern 有相反方向亏损证据时通过 L4 验证。"""
        from cryptotrader.learning.memory import _verify_forbidden_pattern

        # 有 3 条反向亏损记录
        adverse_records = _make_records(3, regime_tags=["high_funding"], win_rate=0.0)

        result = _verify_forbidden_pattern(
            forbidden_loss_rate=0.70,
            adverse_records=adverse_records,
            min_adverse_cases=2,
        )
        assert result is True, "有足够反向亏损证据时应通过 L4"

    def test_forbidden_pattern_without_evidence_fails_l4(self):
        """forbidden pattern 缺乏反向亏损证据时不通过 L4。"""
        from cryptotrader.learning.memory import _verify_forbidden_pattern

        result = _verify_forbidden_pattern(
            forbidden_loss_rate=0.70,
            adverse_records=[],  # 无证据
            min_adverse_cases=2,
        )
        assert result is False, "缺乏反向亏损证据时应不通过 L4"
