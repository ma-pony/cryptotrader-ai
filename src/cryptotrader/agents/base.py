"""Base agent with LLM-powered analysis."""

from __future__ import annotations

import json
import logging
from abc import ABC

import litellm

from cryptotrader.models import AgentAnalysis, DataSnapshot

logger = logging.getLogger(__name__)

ANALYSIS_FRAMEWORK = """
Rules:
- Base your analysis ONLY on the provided data. Do not rely on general market knowledge or historical patterns from your training data.
- Every claim must reference a specific data point from the input.
- If data is missing or insufficient, say so and lower your confidence accordingly.
- Do NOT default to neutral. Take a directional stance when the data supports one.

Confidence calibration:
- 0.9-1.0: Multiple strong, converging signals with no contradictions
- 0.7-0.8: Clear directional signal from primary indicators, minor contradictions
- 0.5-0.6: Mixed signals, slight lean in one direction
- 0.3-0.4: Weak or conflicting signals, low conviction
- 0.1-0.2: Almost no signal, data insufficient or contradictory

Output JSON: {"direction": "bullish|bearish|neutral", "confidence": 0.0-1.0, "reasoning": "2-3 sentences citing specific data", "key_factors": ["factor1", ...], "risk_flags": ["risk1", ...], "data_points": {"indicator": value, ...}}"""


class BaseAgent(ABC):
    def __init__(self, agent_id: str, role_description: str, model: str = "gpt-4o-mini") -> None:
        self.agent_id = agent_id
        self.role_description = role_description
        self.model = model

    async def analyze(self, snapshot: DataSnapshot, experience: str = "") -> AgentAnalysis:
        prompt = self._build_prompt(snapshot, experience)
        system = self.role_description + ANALYSIS_FRAMEWORK
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
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
            f"Funding rate: {fr:.6f}" + (
                " (ELEVATED — crowded long)" if fr > 0.0003
                else " (NEGATIVE — crowded short)" if fr < -0.0001
                else ""),
        ]
        if fut_vol > 0:
            parts.append(f"Futures volume: {fut_vol:,.0f} BTC, vs 20d avg: {vol_ratio:.2f}x" + (
                " (SPIKE)" if vol_ratio > 1.5 else " (LOW)" if vol_ratio < 0.7 else ""))
        if snapshot.onchain.open_interest > 0:
            parts.append(f"Open interest: {snapshot.onchain.open_interest:,.0f}")
        if experience:
            parts.append(f"Past experience:\n{experience}")
        return "\n".join(parts)

    def _parse_response(self, response_text: str, pair: str) -> AgentAnalysis:
        try:
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found in LLM response")
            data = json.loads(response_text[start:end + 1])
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
