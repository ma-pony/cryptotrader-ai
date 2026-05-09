"""skill_metadata_inference — LLM 推断 skill metadata（spec 019 FR-W16/W17/W18/W19）。

propose_new_skill 创建 .draft 前调此模块推断：
  regime_tags / triggers_keywords / importance / confidence

LLM 调用失败 / 输出非合法 JSON / 重试 1 次后仍失败 -> 默认值 + warning log。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ── 默认值（失败时回退）──────────────────────────────────────────────────────────

_DEFAULT_METADATA: dict[str, Any] = {
    "regime_tags": [],
    "triggers_keywords": [],
    "importance": 0.5,
    "confidence": 0.5,
}

# ── spec 014 既有 regime taxonomy（FR-W17）──────────────────────────────────────

_REGIME_TAXONOMY = [
    "high_funding",
    "negative_funding",
    "high_vol",
    "low_vol",
    "trending_up",
    "trending_down",
    "extreme_fear",
    "extreme_greed",
]

# ── 现有 5 skill mapping 示例（FR-W17，与 scripts/migrate_018_to_019.py 一致）──

_EXAMPLES = """
chain-analysis: regime_tags=[], triggers_keywords=["funding rate","exchange flow","whale","OI","on-chain"], importance=0.7, confidence=0.7
macro-analysis: regime_tags=[], triggers_keywords=["fed","dxy","fear greed","etf","macro","cpi"], importance=0.7, confidence=0.7
news-analysis: regime_tags=[], triggers_keywords=["news","headline","regulatory","hack","catalyst"], importance=0.6, confidence=0.6
tech-analysis: regime_tags=[], triggers_keywords=["rsi","macd","sma","bollinger","momentum","breakout"], importance=0.7, confidence=0.7
trading-knowledge: regime_tags=[], triggers_keywords=["funding","regime","perp","basis","calibration"], importance=0.8, confidence=0.8
"""


def _build_prompt(name: str, description: str, body: str) -> str:
    """构建 LLM 推断 prompt（FR-W17）。"""
    # 截取 body 前 500 + 后 200 字符作为摘要
    if len(body) > 700:
        body_summary = body[:500] + "\n...\n" + body[-200:]
    else:
        body_summary = body

    return f"""You are a crypto trading skill metadata inference expert.
Given the new skill content below, output JSON metadata.

[New skill name: {name}]
[Description: {description}]
[Body summary:
{body_summary}
]

[Spec 014 regime taxonomy (valid values for regime_tags):]
{chr(10).join(f"- {r}" for r in _REGIME_TAXONOMY)}

[Existing 5 skill mapping examples:]
{_EXAMPLES}

Output JSON ONLY (no markdown, no explanation):
{{
  "regime_tags": [...],
  "triggers_keywords": [...],
  "importance": 0.0-1.0,
  "confidence": 0.0-1.0
}}

Rules:
- regime_tags: subset of valid taxonomy values; use [] for universal skills
- triggers_keywords: 5-15 keywords that best match when this skill is relevant
- importance: 0.0-1.0 (foundational=0.8, specific=0.6, niche=0.4)
- confidence: 0.0-1.0 (initial estimate, usually 0.5-0.7 for new skills)"""


def _parse_response(raw: str) -> dict[str, Any] | None:
    """从 LLM 输出解析 JSON，失败返回 None。"""
    if not raw:
        return None
    # 尝试直接解析
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 尝试找 JSON block
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


def _validate_and_normalize(data: dict[str, Any]) -> dict[str, Any]:
    """校验并规范化 LLM 输出 dict，返回合法 metadata dict。"""
    result: dict[str, Any] = dict(_DEFAULT_METADATA)

    # regime_tags: 过滤非法值
    raw_tags = data.get("regime_tags", [])
    if isinstance(raw_tags, list):
        result["regime_tags"] = [t for t in raw_tags if t in _REGIME_TAXONOMY]
    else:
        result["regime_tags"] = []

    # triggers_keywords
    raw_kws = data.get("triggers_keywords", [])
    if isinstance(raw_kws, list):
        result["triggers_keywords"] = [str(k) for k in raw_kws if k]
    else:
        result["triggers_keywords"] = []

    # importance: clamp to [0, 1]
    try:
        imp = float(data.get("importance", 0.5))
        result["importance"] = max(0.0, min(1.0, imp))
    except (TypeError, ValueError):
        result["importance"] = 0.5

    # confidence: clamp to [0, 1]
    try:
        conf = float(data.get("confidence", 0.5))
        result["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        result["confidence"] = 0.5

    return result


def _call_llm(prompt: str, llm_callable: Callable | None) -> str | None:
    """调用 LLM，返回原始文本；失败返回 None。"""
    if llm_callable is not None:
        try:
            result = llm_callable(prompt)
            if hasattr(result, "content"):
                return str(result.content)
            return str(result)
        except Exception as exc:
            logger.warning("LLM callable failed: %s", exc, exc_info=True)
            return None

    # 默认使用 config 的 LLM
    try:
        from cryptotrader.agents.base import create_llm

        llm = create_llm("")
        result = llm.invoke(prompt)
        if hasattr(result, "content"):
            return str(result.content)
        return str(result)
    except Exception as exc:
        logger.warning("Default LLM call failed: %s", exc, exc_info=True)
        return None


def infer_skill_metadata(
    name: str,
    description: str,
    body: str,
    llm_callable: Callable | None = None,
) -> dict[str, Any]:
    """LLM 推断 skill metadata（FR-W16/W17/W18/W19）。

    参数：
        name: skill 名（kebab-case）
        description: skill 一句话描述
        body: skill markdown body
        llm_callable: 可选 LLM 调用函数（callable(prompt: str) -> str/AIMessage）
                      None 时走 create_llm("") 默认路径

    返回：
        {"regime_tags": [...], "triggers_keywords": [...],
         "importance": float, "confidence": float}

    LLM 失败 / parse 失败 / 重试失败 -> 默认值（FR-W18）+ warning log。
    """
    prompt = _build_prompt(name, description, body)

    # 首次尝试
    raw = _call_llm(prompt, llm_callable)
    if raw is not None:
        parsed = _parse_response(raw)
        if parsed is not None:
            return _validate_and_normalize(parsed)
        logger.warning("infer_skill_metadata: first parse failed for '%s', retrying", name)
    else:
        logger.warning("infer_skill_metadata: LLM call failed for '%s', retrying", name)

    # 重试 1 次（FR-W18）
    raw2 = _call_llm(prompt, llm_callable)
    if raw2 is not None:
        parsed2 = _parse_response(raw2)
        if parsed2 is not None:
            return _validate_and_normalize(parsed2)

    # 最终回退默认值
    logger.warning(
        "infer_skill_metadata: retry also failed for '%s'; using default metadata",
        name,
    )
    return dict(_DEFAULT_METADATA)
