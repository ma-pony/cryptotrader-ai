"""Base agent with LLM-powered analysis."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime

import litellm

from cryptotrader.models import AgentAnalysis, DataSnapshot

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, agent_id: str, role_description: str, model: str = "gpt-4o-mini") -> None:
        self.agent_id = agent_id
        self.role_description = role_description
        self.model = model

    async def analyze(self, snapshot: DataSnapshot, experience: str = "") -> AgentAnalysis:
        prompt = self._build_prompt(snapshot, experience)
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.role_description},
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
        parts.append(
            "\nDecision framework:"
            "\n1. What is the dominant trend (up/down/sideways)?"
            "\n2. Are there signals CONTRADICTING the trend (reversal risk)?"
            "\n3. If you're wrong, what's the max downside vs upside if right?"
            "\n4. Confidence calibration: 0.5=coin flip, 0.6=slight edge, 0.7=clear signal, 0.8+=multiple strong signals aligned, 0.9+=extreme conviction (rare)"
            "\n\nRespond with JSON: {\"direction\": \"bullish|bearish|neutral\", "
            "\"confidence\": 0.0-1.0, \"reasoning\": \"...\", "
            "\"key_factors\": [...], \"risk_flags\": [...], "
            "\"upside_pct\": estimated upside %, \"downside_pct\": estimated downside %, "
            "\"data_points\": {...}}"
        )
        return "\n".join(parts)

    def _parse_response(self, response_text: str, pair: str) -> AgentAnalysis:
        try:
            start = response_text.index("{")
            end = response_text.rindex("}") + 1
            data = json.loads(response_text[start:end])
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
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            key_factors=data.get("key_factors", []),
            risk_flags=data.get("risk_flags", []),
            data_points=dp,
        )
