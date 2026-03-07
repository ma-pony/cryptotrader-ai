"""Base agent with LLM-powered analysis.

Two agent types:
- BaseAgent: Single LLM call, for agents with complete pre-computed data (TechAgent, MacroAgent)
- ToolAgent: LangChain create_agent loop, for agents that need to actively query data (ChainAgent, NewsAgent)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

import litellm

from cryptotrader.models import AgentAnalysis, DataSnapshot

# ── Unified LLM gateway ──


def get_llm_kwargs() -> dict:
    """Return api_base/api_key/custom_llm_provider kwargs from config for litellm calls."""
    from cryptotrader.config import load_config

    llm_cfg = load_config().llm
    kwargs: dict = {}
    if llm_cfg.base_url:
        kwargs["api_base"] = llm_cfg.base_url
        kwargs["custom_llm_provider"] = "openai"
    if llm_cfg.api_key:
        kwargs["api_key"] = llm_cfg.api_key
    return kwargs


async def acompletion_with_fallback(*, model: str, **kwargs) -> Any:
    """litellm.acompletion with automatic fallback to config fallback model.

    On timeout or error, retries once with the fallback model defined in config.
    """
    from cryptotrader.config import load_config

    llm_kw = get_llm_kwargs()
    try:
        return await litellm.acompletion(model=model, **llm_kw, **kwargs)
    except Exception as primary_err:
        fallback = load_config().models.fallback
        if fallback and fallback != model:
            _logger.warning("Model %s failed (%s), falling back to %s", model, primary_err, fallback)
            return await litellm.acompletion(model=fallback, **llm_kw, **kwargs)
        raise


_logger = logging.getLogger(__name__)
logger = _logger

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

Output JSON: {"direction": "bullish|bearish|neutral", "confidence": 0.0-1.0, "reasoning": "2-3 sentences citing
specific data", "key_factors": ["factor1", ...], "risk_flags": ["risk1", ...], "data_points": {"indicator": value,
...}}"""


class BaseAgent:
    def __init__(self, agent_id: str, role_description: str, model: str = "gpt-4o-mini") -> None:
        self.agent_id = agent_id
        self.role_description = role_description
        self.model = model

    async def analyze(self, snapshot: DataSnapshot, experience: str = "") -> AgentAnalysis:
        prompt = self._build_prompt(snapshot, experience)
        system = self.role_description + ANALYSIS_FRAMEWORK
        try:
            response = await acompletion_with_fallback(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
                timeout=60,
            )
            text = response.choices[0].message.content
            return self._parse_response(text, snapshot.pair)
        except Exception:
            logger.exception("LLM call failed for %s, returning mock analysis", self.agent_id)
            return AgentAnalysis(
                agent_id=self.agent_id,
                pair=snapshot.pair,
                direction="neutral",
                confidence=0.5,
                reasoning="LLM unavailable - mock analysis",
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
            + (" (ELEVATED — crowded long)" if fr > 0.0003 else " (NEGATIVE — crowded short)" if fr < -0.0001 else ""),
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
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found in LLM response")
            data = json.loads(response_text[start : end + 1])
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
        standard = {"direction", "confidence", "reasoning", "key_factors", "risk_flags", "data_points"}
        # Everything else goes into data_points for downstream rules engine
        extra = {k: v for k, v in data.items() if k not in standard}
        dp = data.get("data_points", {})
        dp.update(extra)

        return AgentAnalysis(
            agent_id=self.agent_id,
            pair=pair,
            direction=data.get("direction", "neutral"),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            reasoning=data.get("reasoning", ""),
            key_factors=data.get("key_factors", []),
            risk_flags=data.get("risk_flags", []),
            data_points=dp,
        )


def _create_chat_model(model: str, temperature: float = 0.2, max_tokens: int = 1024):
    """Create a LangChain chat model using the unified LLM gateway.

    All models route through the same base_url/api_key configured in config/default.toml.
    """
    from langchain_openai import ChatOpenAI

    from cryptotrader.config import load_config

    llm_cfg = load_config().llm
    kwargs: dict = {"model": model, "temperature": temperature, "max_completion_tokens": max_tokens}
    if llm_cfg.base_url:
        kwargs["base_url"] = llm_cfg.base_url
    if llm_cfg.api_key:
        kwargs["api_key"] = llm_cfg.api_key
    return ChatOpenAI(**kwargs)


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
        model: str = "gpt-4o-mini",
    ) -> None:
        super().__init__(agent_id, role_description, model)
        self.tools = list(tools)

    async def analyze(self, snapshot: DataSnapshot, experience: str = "") -> AgentAnalysis:
        prompt = self._build_prompt(snapshot, experience)
        system = self.role_description + ANALYSIS_FRAMEWORK

        try:
            from langchain.agents import create_agent

            llm = _create_chat_model(self.model, temperature=0.2, max_tokens=1024)
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
