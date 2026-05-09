"""spec 019 - IDF unit tests (SC-W4).

tests/test_idf.py - >= 6 test cases PASS
"""

from __future__ import annotations

import math

import pytest


class TestComputeIdf:
    """compute_idf 函数测试。"""

    def test_single_skill_corpus_all_keywords_present(self):
        """(a) 单 skill corpus -> idf table 含全部关键词。"""
        from cryptotrader.learning.evolution.idf import compute_idf

        corpus = [["rsi", "macd", "sma"]]
        result = compute_idf(corpus)
        assert "rsi" in result
        assert "macd" in result
        assert "sma" in result
        # log(1/1) = 0 for single doc
        assert result["rsi"] == pytest.approx(0.0)

    def test_five_skill_corpus_shared_keyword_has_low_idf(self):
        """(b) 5 skill corpus -> 共享关键词低 IDF (log(5/k))。"""
        from cryptotrader.learning.evolution.idf import compute_idf

        # 5 skill，共同关键词 "funding" 出现在 5 个 skill 中
        corpus = [
            ["funding", "rsi"],
            ["funding", "macd"],
            ["funding", "news"],
            ["funding", "whale"],
            ["funding", "fed"],
        ]
        result = compute_idf(corpus)
        # "funding" 在所有 5 个 skill 中 -> IDF = log(5/5) = 0
        assert result["funding"] == pytest.approx(math.log(5 / 5))
        # "rsi" 只在 1 个 skill 中 -> IDF = log(5/1) = log(5)
        assert result["rsi"] == pytest.approx(math.log(5 / 1))

    def test_empty_corpus_returns_empty_dict(self):
        """(c) 空 corpus -> 空 dict。"""
        from cryptotrader.learning.evolution.idf import compute_idf

        result = compute_idf([])
        assert result == {}

    def test_keywords_lowercased_in_idf_table(self):
        """IDF table 中 keyword 应为小写化。"""
        from cryptotrader.learning.evolution.idf import compute_idf

        corpus = [["RSI", "MACD"], ["Rsi", "whale"]]
        result = compute_idf(corpus)
        # "RSI" 和 "Rsi" 都应归并为 "rsi"（2 个 doc 中出现）
        assert "rsi" in result
        assert "RSI" not in result
        assert result["rsi"] == pytest.approx(math.log(2 / 2))  # = 0

    def test_duplicate_keywords_within_skill_not_double_counted(self):
        """同一 skill 内重复 keyword 只计 1 次 df。"""
        from cryptotrader.learning.evolution.idf import compute_idf

        # "rsi" 在 skill 0 内出现 3 次，但 df 应只记 1
        corpus = [["rsi", "rsi", "rsi"], ["rsi"]]
        result = compute_idf(corpus)
        # "rsi" 出现在 2 个 docs -> IDF = log(2/2) = 0
        assert result["rsi"] == pytest.approx(0.0)


class TestExtractQueryKeywords:
    """extract_query_keywords 函数测试。"""

    def test_extracts_field_names_from_snapshot(self):
        """(d) 从 snapshot dict 提取字段名。"""
        from cryptotrader.learning.evolution.idf import extract_query_keywords

        snapshot = {"funding_rate": 0.0003, "rsi_14": 65.0, "pair": "BTC/USDT"}
        result = extract_query_keywords(snapshot)
        assert "funding_rate" in result
        assert "rsi_14" in result
        assert "pair" in result

    def test_extracts_string_values(self):
        """字符串值也应被提取（小写化）。"""
        from cryptotrader.learning.evolution.idf import extract_query_keywords

        snapshot = {"regime": "HIGH_FUNDING", "pair": "BTC/USDT"}
        result = extract_query_keywords(snapshot)
        assert "high_funding" in result
        assert "btc/usdt" in result

    def test_nested_dict_fields_extracted(self):
        """嵌套 dict 的字段名也应被提取。"""
        from cryptotrader.learning.evolution.idf import extract_query_keywords

        snapshot = {"market": {"open_interest": 1000, "funding_rate": 0.0003}}
        result = extract_query_keywords(snapshot)
        assert "market" in result
        assert "open_interest" in result
        assert "funding_rate" in result

    def test_empty_snapshot_returns_empty_set(self):
        """空 snapshot 返回空 set。"""
        from cryptotrader.learning.evolution.idf import extract_query_keywords

        result = extract_query_keywords({})
        assert result == set()


class TestScoreSkill:
    """score_skill 函数测试。"""

    def test_score_sums_idf_for_matching_keywords(self):
        """(e) score_skill 加和 IDF（含小写匹配）。"""
        from cryptotrader.learning.evolution.idf import score_skill

        idf_table = {"rsi": 1.6, "macd": 1.2, "funding": 0.0}
        query_keywords = {"rsi", "funding", "btc"}
        skill_keywords = ["RSI", "macd", "funding"]  # RSI 大写，应仍匹配
        score = score_skill(skill_keywords, query_keywords, idf_table)
        # "RSI".lower()="rsi" 在 query -> +1.6
        # "macd" 不在 query -> +0
        # "funding" 在 query -> +0.0
        assert score == pytest.approx(1.6)

    def test_empty_intersection_returns_zero(self):
        """(f) score_skill 空交集 -> 0。"""
        from cryptotrader.learning.evolution.idf import score_skill

        idf_table = {"rsi": 1.6, "macd": 1.2}
        query_keywords = {"news", "fed", "inflation"}
        skill_keywords = ["rsi", "macd"]
        score = score_skill(skill_keywords, query_keywords, idf_table)
        assert score == pytest.approx(0.0)

    def test_empty_skill_keywords_returns_zero(self):
        """triggers_keywords=[] 时 IDF 评分=0（FR-W8）。"""
        from cryptotrader.learning.evolution.idf import score_skill

        idf_table = {"rsi": 1.6}
        query_keywords = {"rsi", "btc"}
        score = score_skill([], query_keywords, idf_table)
        assert score == pytest.approx(0.0)

    def test_empty_query_returns_zero(self):
        """empty query_keywords returns 0."""
        from cryptotrader.learning.evolution.idf import score_skill

        idf_table = {"rsi": 1.6}
        score = score_skill(["rsi"], set(), idf_table)
        assert score == pytest.approx(0.0)
