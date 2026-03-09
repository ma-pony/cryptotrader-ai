"""Base agent with LLM-powered analysis.

Two agent types:
- BaseAgent: Single LLM call, for agents with complete pre-computed data (TechAgent, MacroAgent)
- ToolAgent: LangChain create_agent loop, for agents that need to actively query data (ChainAgent, NewsAgent)

All LLM calls go through unified LangChain gateway with SQLiteCache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from cryptotrader.models import AgentAnalysis, DataSnapshot

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = logging.getLogger(__name__)
logger = _logger

# Funding rate thresholds shared with debate/verdict.py
FUNDING_RATE_HIGH = 0.0003  # above → crowded long
FUNDING_RATE_LOW = -0.0001  # below → crowded short

# ── LangChain SQLiteCache initialization ──

_cache_initialized = False


def _init_cache() -> None:
    """Initialize LangChain SQLiteCache for exact-match LLM caching."""
    global _cache_initialized
    if _cache_initialized:
        return
    try:
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache

        cache_dir = Path.home() / ".cryptotrader"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_db = cache_dir / "llm_cache.db"
        set_llm_cache(SQLiteCache(database_path=str(cache_db)))
        _logger.info("LLM cache initialized: %s", cache_db)
    except Exception:
        _logger.warning("Failed to initialize LLM cache, continuing without cache", exc_info=True)
    _cache_initialized = True


# ── Unified LLM factory ──


def create_llm(
    model: str,
    temperature: float = 0.2,
    timeout: int = 120,
    json_mode: bool = False,
    *,
    with_fallback: bool = True,
) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance with unified config.

    All LLM calls in the project route through this factory to ensure
    consistent configuration, caching, and fallback behavior.
    """
    from cryptotrader.config import load_config

    _init_cache()

    cfg = load_config()
    llm_cfg = cfg.llm

    # Resolve empty model to config default
    if not model:
        model = cfg.models.analysis or cfg.models.fallback
    if not model:
        raise ValueError("No LLM model configured — set models.analysis or models.fallback in config/default.toml")

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "timeout": timeout,
    }
    if llm_cfg.base_url:
        kwargs["base_url"] = llm_cfg.base_url
    if llm_cfg.api_key:
        kwargs["api_key"] = llm_cfg.api_key
    # Note: response_format is NOT passed — many models (e.g. gpt-5.4)
    # don't support it. JSON output enforced via prompts + _extract_json().

    # Some models require streaming via third-party proxy restrictions
    if model in llm_cfg.streaming_models:
        kwargs["streaming"] = True

    llm = ChatOpenAI(**kwargs)

    # Add fallback model
    if with_fallback:
        fallback_model = cfg.models.fallback
        if fallback_model and fallback_model != model:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["model"] = fallback_model
            # Reset streaming for fallback — it has its own streaming requirements
            fallback_kwargs.pop("streaming", None)
            if fallback_model in llm_cfg.streaming_models:
                fallback_kwargs["streaming"] = True
            fallback_llm = ChatOpenAI(**fallback_kwargs)
            llm = llm.with_fallbacks([fallback_llm])

    return llm


def _to_langchain_messages(messages: list[dict]) -> list:
    """Convert OpenAI-format message dicts to LangChain message objects."""
    result = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(HumanMessage(content=content))
        else:
            result.append(AIMessage(content=content))
    return result


def extract_content(response: AIMessage | Any) -> str:
    """Extract text content from LLM response.

    Handles both LangChain AIMessage and reasoning models (deepseek-reasoner)
    that may return content in additional_kwargs['reasoning_content'].
    """
    if isinstance(response, AIMessage):
        text = response.content or ""
        if not text:
            # deepseek-reasoner puts answer in additional_kwargs
            text = response.additional_kwargs.get("reasoning_content", "")
        return text
    return str(response)


async def acompletion_with_fallback(*, model: str, **kwargs) -> AIMessage:
    """Unified LLM call via LangChain with automatic fallback.

    Accepts OpenAI-format kwargs and returns AIMessage.
    """
    messages = kwargs.pop("messages", [])
    temperature = kwargs.pop("temperature", 0.2)
    timeout = kwargs.pop("timeout", 120)
    response_format = kwargs.pop("response_format", None)
    json_mode = response_format is not None and response_format.get("type") == "json_object"

    llm = create_llm(model=model, temperature=temperature, timeout=timeout, json_mode=json_mode)
    lc_messages = _to_langchain_messages(messages)
    return await llm.ainvoke(lc_messages)


ANALYSIS_FRAMEWORK = """
Rules:
- Base your analysis ONLY on the provided data. Do not rely on general market knowledge or historical patterns.
- Every claim must reference a specific data point from the input.
- If data is missing or insufficient, say so and lower your confidence accordingly.
- Do NOT default to neutral. Take a directional stance when the data supports one.

Pre-signal checklist (you MUST verify each before outputting your signal):
1. Contradiction check: Are there signals in the data that CONTRADICT my direction? If yes, have I explicitly
acknowledged them and explained why I'm overriding?
2. Evidence grounding: Does every claim in my reasoning reference a specific number or data point? If I catch
myself saying "the market looks..." without citing data, stop and fix it.
3. Confidence sanity: Would I bet real money at this confidence level? 0.8+ means I see strong convergence with
no red flags. If I'm unsure, my confidence should be below 0.6.
4. Base rate awareness: Most of the time, the correct signal is hold. A directional call requires clear evidence,
not just a slight lean.
5. Recency trap: Am I overweighting the most recent data point while ignoring the broader context in the window?

Confidence calibration:
- 0.9-1.0: Multiple strong, converging signals with no contradictions
- 0.7-0.8: Clear directional signal from primary indicators, minor contradictions
- 0.5-0.6: Mixed signals, slight lean in one direction
- 0.3-0.4: Weak or conflicting signals, low conviction
- 0.1-0.2: Almost no signal, data insufficient or contradictory

Data sufficiency self-assessment:
- "high": Your core data sources are present and complete. You can make a well-informed directional call.
- "medium": Some data is present but key sources are missing or stale. Moderate confidence at best.
- "low": Most of your core data is missing, zero, or placeholder. You MUST set confidence ≤ 0.3 and direction
  to "neutral". Do NOT guess a direction without data — say "insufficient data" in reasoning.

Output JSON: {"direction": "bullish|bearish|neutral", "confidence": 0.0-1.0, "data_sufficiency": "high|medium|low",
"reasoning": "2-3 sentences citing specific data", "key_factors": ["factor1", ...], "risk_flags": ["risk1", ...],
"data_points": {"indicator": value, ...}}"""


class BaseAgent:
    def __init__(self, agent_id: str, role_description: str, model: str = "") -> None:
        self.agent_id = agent_id
        self.role_description = role_description
        self.model = model

    def _resolve_model(self) -> str:
        """Return model name, falling back to config if not set."""
        if self.model:
            return self.model
        from cryptotrader.config import load_config

        cfg = load_config()
        return cfg.models.analysis or cfg.models.fallback

    async def analyze(self, snapshot: DataSnapshot, experience: str = "") -> AgentAnalysis:
        prompt = self._build_prompt(snapshot, experience)
        system = self.role_description + ANALYSIS_FRAMEWORK
        try:
            llm = create_llm(model=self._resolve_model(), temperature=0.2, timeout=120, json_mode=True)
            messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
            response = await llm.ainvoke(messages)
            text = extract_content(response)
            return self._parse_response(text, snapshot.pair)
        except Exception:
            logger.exception("LLM call failed for %s, returning mock analysis", self.agent_id)
            return AgentAnalysis(
                agent_id=self.agent_id,
                pair=snapshot.pair,
                direction="neutral",
                confidence=0.1,
                reasoning="LLM unavailable - mock analysis",
                is_mock=True,
                data_sufficiency="low",
            )

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        fr = snapshot.market.funding_rate
        liq = snapshot.onchain.liquidations_24h
        vol_ratio = liq.get("volume_ratio", 0)
        fut_vol = liq.get("futures_volume", 0)

        parts = [
            f"Pair: {snapshot.pair}",
            f"Timestamp: {snapshot.timestamp}",
            f"Ticker: {snapshot.market.ticker}",
            f"Volatility: {snapshot.market.volatility:.4f}",
            f"Funding rate: {fr:.6f}"
            + (
                " (ELEVATED — crowded long)"
                if fr > FUNDING_RATE_HIGH
                else " (NEGATIVE — crowded short)"
                if fr < FUNDING_RATE_LOW
                else ""
            ),
        ]
        if fut_vol > 0:
            parts.append(
                f"Futures volume: {fut_vol:,.0f} BTC, vs 20d avg: {vol_ratio:.2f}x"
                + (" (SPIKE)" if vol_ratio > 1.5 else " (LOW)" if vol_ratio < 0.7 else "")
            )
        if snapshot.onchain.open_interest > 0:
            parts.append(f"Open interest: {snapshot.onchain.open_interest:,.0f}")
        # Data quality warnings — tell agents when sources are empty
        warnings = []
        if snapshot.onchain.open_interest == 0 and snapshot.onchain.exchange_netflow == 0:
            warnings.append("On-chain data unavailable (no API keys configured). Do NOT infer from missing data.")
        if snapshot.news.sentiment_score == 0 and not snapshot.news.headlines:
            warnings.append("News sentiment unavailable. Do NOT assume neutral sentiment from missing data.")
        if hasattr(snapshot, "macro") and snapshot.macro.fed_rate == 0 and snapshot.macro.dxy == 0:
            warnings.append("Macro data unavailable (FRED/DXY). Do NOT infer from zero values — they are missing.")
        if warnings:
            parts.append("⚠ DATA QUALITY WARNINGS:\n" + "\n".join(f"  - {w}" for w in warnings))

        if experience:
            parts.append(f"Past experience:\n{experience}")
        return "\n".join(parts)

    def _parse_response(self, response_text: str, pair: str) -> AgentAnalysis:
        try:
            from cryptotrader.debate.verdict import _extract_json

            data = _extract_json(response_text)
        except (ValueError, json.JSONDecodeError):
            logger.warning("Failed to parse LLM response for %s", self.agent_id)
            return AgentAnalysis(
                agent_id=self.agent_id,
                pair=pair,
                direction="neutral",
                confidence=0.5,
                reasoning=response_text[:500],
            )
        # Standard fields
        standard = {
            "direction",
            "confidence",
            "reasoning",
            "key_factors",
            "risk_flags",
            "data_points",
            "data_sufficiency",
        }
        # Everything else goes into data_points for downstream rules engine
        extra = {k: v for k, v in data.items() if k not in standard}
        dp = data.get("data_points", {})
        dp.update(extra)

        # Parse data_sufficiency and enforce confidence cap for low-data agents
        sufficiency = data.get("data_sufficiency", "medium")
        if sufficiency not in ("high", "medium", "low"):
            sufficiency = "medium"
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        if sufficiency == "low":
            confidence = min(confidence, 0.3)

        return AgentAnalysis(
            agent_id=self.agent_id,
            pair=pair,
            direction=data.get("direction", "neutral"),
            confidence=confidence,
            reasoning=data.get("reasoning", ""),
            key_factors=data.get("key_factors", []),
            risk_flags=data.get("risk_flags", []),
            data_points=dp,
            data_sufficiency=sufficiency,
        )


def _create_chat_model(model: str, temperature: float = 0.2):
    """Create a LangChain chat model without fallback (for ToolAgent / create_agent)."""
    return create_llm(model=model, temperature=temperature, with_fallback=False)


class ToolAgent(BaseAgent):
    """Agent with tool-calling capability via LangChain create_agent.

    Unlike BaseAgent (single LLM call), ToolAgent runs a model→tool→model loop,
    allowing the AI to actively query data sources during analysis.
    """

    def __init__(
        self,
        agent_id: str,
        role_description: str,
        tools: Sequence,
        model: str = "",
    ) -> None:
        super().__init__(agent_id, role_description, model)
        self.tools = list(tools)

    async def analyze(self, snapshot: DataSnapshot, experience: str = "") -> AgentAnalysis:
        prompt = self._build_prompt(snapshot, experience)
        system = self.role_description + ANALYSIS_FRAMEWORK

        try:
            from langchain.agents import create_agent

            llm = _create_chat_model(self.model, temperature=0.2)
            agent = create_agent(llm, tools=self.tools, system_prompt=system)

            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
            )

            # Extract final message text from agent output
            final_msg = result["messages"][-1]
            text = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
            return self._parse_response(text, snapshot.pair)

        except Exception:
            logger.exception("ToolAgent call failed for %s, falling back to single-call", self.agent_id)
            # Fallback to single LLM call (same as BaseAgent)
            return await super().analyze(snapshot, experience)
