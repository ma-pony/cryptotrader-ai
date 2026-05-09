"""IDF 算法模块 — spec 019 D-RT-01 第二层排序的关键词权重计算。

pure Python 实现，无 sklearn / nltk 依赖。
用于 EvolvingSkillProvider 的第二层排序。
"""

from __future__ import annotations

import math
from collections import defaultdict


def compute_idf(corpus_keywords: list[list[str]]) -> dict[str, float]:
    """从 skill 关键词语料库计算 IDF 表。

    参数：
        corpus_keywords: 每个 skill 的 triggers_keywords list（list of lists）。

    返回：
        {keyword: idf_score} dict，keyword 全小写化。
        单 doc 中 keyword=log(N/1)；全部 doc 共享 keyword=log(N/N)=0。

    空 corpus 返回 {}。
    """
    n_docs = len(corpus_keywords)
    if n_docs == 0:
        return {}

    # 文档频率（document frequency）：每个 keyword 出现在几个 doc 中
    df: dict[str, int] = defaultdict(int)
    for skill_kw in corpus_keywords:
        # 去重（每个 skill 内部不重复计数）
        unique = {kw.lower() for kw in skill_kw}
        for kw in unique:
            df[kw] += 1

    # IDF = log(N / df)；df=0 情况理论不存在（只有出现过才进 df）
    return {kw: math.log(n_docs / count) for kw, count in df.items()}


def extract_query_keywords(snapshot: dict) -> set[str]:
    """从 snapshot dict 提取 query keywords。

    提取策略：
    1. 所有顶层字段名（小写化）
    2. str/int/float 类型的值，小写化（过滤 None / 嵌套 dict/list）

    返回小写 keyword 集合，供 score_skill 使用。
    """
    keywords: set[str] = set()

    def _add(value: object) -> None:
        """递归提取值中的关键字。"""
        if value is None:
            return
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped:
                keywords.add(stripped)
        elif isinstance(value, int | float):
            keywords.add(str(value).lower())
        elif isinstance(value, dict):
            for k, v in value.items():
                keywords.add(str(k).lower())
                _add(v)
        elif isinstance(value, list | tuple):
            for item in value:
                _add(item)

    for key, val in snapshot.items():
        keywords.add(str(key).lower())
        _add(val)

    return keywords


def score_skill(
    skill_keywords: list[str],
    query_keywords: set[str],
    idf_table: dict[str, float],
) -> float:
    """计算 skill 在当前 query（snapshot）上的 IDF score。

    score = sum(idf_table[kw] for kw in skill_keywords if kw.lower() in query_keywords)

    参数：
        skill_keywords: skill 的 triggers_keywords list
        query_keywords: 从 snapshot 提取的关键词集合（小写）
        idf_table: compute_idf 返回的 {keyword: idf_score}

    返回：
        float score；skill_keywords=[] 或无交集返回 0.0。
    """
    if not skill_keywords or not query_keywords:
        return 0.0

    total = 0.0
    for kw in skill_keywords:
        kw_lower = kw.lower()
        if kw_lower in query_keywords:
            total += idf_table.get(kw_lower, 0.0)
    return total
