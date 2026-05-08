"""Technical analysis agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cryptotrader.agents._tech_indicators import compute_indicators
from cryptotrader.agents.base import BaseAgent, create_llm, extract_content, log_llm_usage

if TYPE_CHECKING:
    from cryptotrader.agents.prompt_builder import PromptBuilder
    from cryptotrader.models import AgentAnalysis, DataSnapshot

logger = logging.getLogger(__name__)


class TechAgent(BaseAgent):
    def __init__(self, *, prompt_builder: PromptBuilder, model: str = "") -> None:
        super().__init__(agent_id="tech", prompt_builder=prompt_builder, model=model)

    async def analyze(self, snapshot: DataSnapshot, experience: str = "") -> AgentAnalysis:  # type: ignore[override]
        snapshot_dict = self._snapshot_to_dict(snapshot)
        snapshot_dict["indicators"] = compute_indicators(snapshot.market.ohlcv)
        try:
            sys_msg, usr_msg = self._prompt_builder.build(
                snapshot=snapshot_dict,
                portfolio={},
                experience=experience,
            )
            model = self._resolve_model()
            llm = create_llm(model=model)
            messages = [sys_msg, usr_msg]
            from cryptotrader.llm.prompt_cache import apply_cache_control, should_cache

            if should_cache(model=model, role=self.agent_id):
                messages = apply_cache_control(messages)
            response = await llm.ainvoke(messages)
            log_llm_usage(response, caller=self.agent_id)
            text = extract_content(response)
            return await self._parse_response(text, snapshot.pair, llm=llm)
        except Exception:
            logger.exception("TechAgent LLM call failed, returning mock")
            from cryptotrader.models import AgentAnalysis

            return AgentAnalysis(
                agent_id=self.agent_id,
                pair=snapshot.pair,
                direction="neutral",
                confidence=0.1,
                reasoning="LLM unavailable - mock analysis",
                is_mock=True,
                data_sufficiency="low",
            )
