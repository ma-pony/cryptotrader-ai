"""Four-domain LLM committee with an internal LangGraph debate."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from cryptotrader.cycle_events import CycleEvent, NullCycleEventSink
from cryptotrader.debate.challenge import challenge_agent
from cryptotrader.debate.convergence import check_convergence, compute_divergence, debate_gate_decision
from cryptotrader.signals.component import ComponentExecutionError
from cryptotrader.signals.models import CandleRequirement, ComponentSignal, DataRequirements, SignalContext

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from cryptotrader.config import AppConfig
    from cryptotrader.cycle_events import CycleEventSink
    from cryptotrader.runtime_config.models import RuntimeConfigDocument


class CommitteeState(TypedDict):
    context: SignalContext
    analyses: dict[str, dict[str, Any]]
    debate_round: int
    debate_turns: list[dict[str, Any]]
    divergence_scores: list[float]
    debate_skipped: bool
    debate_skip_reason: str
    consensus_metrics: dict[str, float]
    final_signal: ComponentSignal | None


def normalize_summary_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    direction = payload["direction"]
    if direction not in {"long", "short", "neutral"}:
        raise ValueError(f"invalid committee direction: {direction}")
    confidence = float(payload["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("committee confidence must be in [0, 1]")
    reasoning = str(payload["reasoning"]).strip()
    if not reasoning:
        raise ValueError("committee reasoning must not be empty")
    return {"direction": direction, "confidence": confidence, "reasoning": reasoning}


class LLMCommitteeComponent:
    id = "llm_committee"
    display_name = "LLM 四智能体委员会"
    description = "技术、链上、新闻和宏观智能体的内部交叉辩论"

    def __init__(
        self,
        config: AppConfig,
        *,
        agents: Mapping[str, Any] | None = None,
        summary: Callable[[CommitteeState], Awaitable[Mapping[str, Any]]] | None = None,
        challenger: Callable[..., Awaitable[tuple[dict, dict]]] | None = None,
        sink: CycleEventSink | None = None,
    ) -> None:
        self.config = config
        self.events = sink or NullCycleEventSink()
        self.agents = dict(agents) if agents is not None else self._build_agents()
        self._summary = summary or self._summarize_with_llm
        self._challenger = challenger or self._challenge_with_llm
        self.graph = self._build_graph()

    def requirements(self) -> DataRequirements:
        return DataRequirements(
            candles=(CandleRequirement(self.config.data.default_timeframe, self.config.data.ohlcv_limit),),
            onchain=True,
            news=True,
            macro=True,
        )

    async def evaluate(self, context: SignalContext) -> ComponentSignal:
        initial: CommitteeState = {
            "context": context,
            "analyses": {},
            "debate_round": 0,
            "debate_turns": [],
            "divergence_scores": [],
            "debate_skipped": False,
            "debate_skip_reason": "",
            "consensus_metrics": {},
            "final_signal": None,
        }
        result = await self.graph.ainvoke(initial)
        signal = result["final_signal"]
        if signal is None:
            raise self._error("summary", RuntimeError("committee graph returned no signal"))
        return signal

    def _build_graph(self):
        graph = StateGraph(CommitteeState)
        graph.add_node("analyze_all", self._analyze_all)
        graph.add_node("debate_gate", self._debate_gate)
        graph.add_node("debate_round", self._debate_round)
        graph.add_node("convergence", self._convergence)
        graph.add_node("summarize", self._summarize)
        graph.add_edge(START, "analyze_all")
        graph.add_edge("analyze_all", "debate_gate")
        graph.add_conditional_edges(
            "debate_gate",
            self._debate_gate_route,
            {"debate": "debate_round", "skip": "summarize"},
        )
        graph.add_edge("debate_round", "convergence")
        graph.add_conditional_edges(
            "convergence",
            self._convergence_route,
            {"continue": "debate_round", "converged": "summarize"},
        )
        graph.add_edge("summarize", END)
        return graph.compile()

    async def _analyze_all(self, state: CommitteeState) -> dict:
        snapshot = state["context"].snapshots[self.config.data.default_timeframe]
        names = list(self.agents)
        results = await asyncio.gather(
            *(self._analyze_one(name, self.agents[name], snapshot) for name in names),
            return_exceptions=True,
        )
        for name, result in zip(names, results, strict=True):
            if isinstance(result, BaseException):
                raise self._error("analysis", RuntimeError(f"{name}: {result}")) from result
        return {"analyses": dict(zip(names, results, strict=True))}

    async def _analyze_one(self, name: str, agent, snapshot) -> dict[str, Any]:
        await self.events.publish(CycleEvent("committee_agent_started", {"agent_id": name}))
        try:
            result = await agent.analyze(snapshot)
            analysis = asdict(result) if is_dataclass(result) else dict(result)
            if analysis.get("is_mock"):
                raise RuntimeError("agent returned mock analysis")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.events.publish(
                CycleEvent(
                    "committee_agent_failed",
                    {"agent_id": name, "error_type": type(error).__name__, "error": str(error)},
                )
            )
            raise
        await self.events.publish(CycleEvent("agent_analysis_completed", {"agent_id": name, "analysis": analysis}))
        return analysis

    async def _debate_gate(self, state: CommitteeState) -> dict:
        skipped, reason, metrics = debate_gate_decision(state["analyses"], self.config.debate)
        return {
            "debate_skipped": skipped,
            "debate_skip_reason": reason,
            "consensus_metrics": metrics,
        }

    @staticmethod
    def _debate_gate_route(state: CommitteeState) -> str:
        return "skip" if state["debate_skipped"] else "debate"

    async def _debate_round(self, state: CommitteeState) -> dict:
        round_number = state["debate_round"] + 1
        await self.events.publish(CycleEvent("debate_round_started", {"round_number": round_number}))
        analyses = state["analyses"]
        names = list(analyses)
        results = await asyncio.gather(
            *(
                self._challenger(
                    name,
                    analyses[name],
                    {other: analyses[other] for other in names if other != name},
                    state["context"],
                    round_number,
                )
                for name in names
            ),
            return_exceptions=True,
        )
        for name, result in zip(names, results, strict=True):
            if isinstance(result, BaseException):
                raise self._error("debate", RuntimeError(f"{name}: {result}")) from result
        updated = {}
        turns = list(state["debate_turns"])
        for name, result in zip(names, results, strict=True):
            updated[name], turn = result
            turns.append(turn)
        await self.events.publish(
            CycleEvent(
                "debate_round_completed",
                {"round_number": round_number, "analyses": updated},
            )
        )
        return {"analyses": updated, "debate_turns": turns, "debate_round": round_number}

    async def _convergence(self, state: CommitteeState) -> dict:
        scores = [*state["divergence_scores"], compute_divergence(state["analyses"])]
        return {"divergence_scores": scores}

    def _convergence_route(self, state: CommitteeState) -> str:
        if state["debate_round"] >= self.config.debate.max_rounds:
            return "converged"
        scores = state["divergence_scores"]
        if len(scores) >= 2 and check_convergence(
            scores[:-1],
            scores[-1],
            threshold=self.config.debate.convergence_threshold,
        ):
            return "converged"
        return "continue"

    async def _summarize(self, state: CommitteeState) -> dict:
        try:
            payload = normalize_summary_payload(await self._summary(state))
        except Exception as error:
            raise self._error("summary", error) from error
        signal = ComponentSignal(
            component_id=self.id,
            direction=payload["direction"],
            confidence=payload["confidence"],
            reasoning=payload["reasoning"],
            details={
                "analyses": state["analyses"],
                "debate_turns": state["debate_turns"],
                "consensus_metrics": state["consensus_metrics"],
                "debate_skipped": state["debate_skipped"],
                "debate_skip_reason": state["debate_skip_reason"],
            },
        )
        await self.events.publish(CycleEvent("committee_summary_completed", {"signal": signal}))
        return {"final_signal": signal}

    async def _challenge_with_llm(
        self,
        agent_id: str,
        analysis: dict,
        others: dict[str, dict],
        context: SignalContext,
        round_number: int,
    ) -> tuple[dict, dict]:
        model = self.config.models.debate or self.config.models.fallback
        return await challenge_agent(
            agent_id,
            analysis,
            others,
            context.pair.display(),
            model,
            self.config.models.timeout_seconds,
            round_number,
        )

    async def _summarize_with_llm(self, state: CommitteeState) -> Mapping[str, Any]:
        from cryptotrader.agents.base import create_llm, extract_content
        from cryptotrader.llm.json_retry import extract_json_with_retry

        model = self.config.models.committee_summary or self.config.models.debate or self.config.models.fallback
        llm = create_llm(model=model, temperature=0.1, json_mode=True)
        system = SystemMessage(
            content=(
                "Summarize a four-domain market debate into one market view. Return JSON with exactly "
                "direction (long|short|neutral), confidence (0..1), and reasoning. Do not output trading actions, "
                "position size, leverage, orders, stop loss, take profit, or price targets."
            )
        )
        evidence = {
            "pair": state["context"].pair.display(),
            "analyses": state["analyses"],
            "debate_turns": state["debate_turns"],
            "consensus_metrics": state["consensus_metrics"],
        }
        response = await llm.ainvoke([system, HumanMessage(content=json.dumps(evidence, ensure_ascii=False))])
        return await extract_json_with_retry(
            extract_content(response),
            llm=llm,
            schema_hint="direction,confidence,reasoning",
            max_retries=2,
        )

    def _build_agents(self) -> dict[str, Any]:
        from cryptotrader.agents.prompt_builder import PromptBuilder
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        project_root = Path(__file__).resolve().parents[4]
        provider = EvolvingSkillProvider(skill_root=project_root / "agent_skills/_internal")
        models = {
            "tech_agent": self.config.models.tech_agent,
            "chain_agent": self.config.models.chain_agent,
            "news_agent": self.config.models.news_agent,
            "macro_agent": self.config.models.macro_agent,
        }
        result = {}
        for agent_id, model in models.items():
            prompt_builder = PromptBuilder(
                agent_id=agent_id.removesuffix("_agent"),
                config_dir=project_root / "config/agents",
                skill_provider=provider,
                model=model,
            )
            result[agent_id] = self.config.agents.build(
                agent_id,
                prompt_builder=prompt_builder,
                backtest_mode=True,
                model_override=model,
            )
        return result

    def _error(self, stage: str, cause: BaseException) -> ComponentExecutionError:
        return ComponentExecutionError(self.id, RuntimeError(f"{stage}: {cause}"))


def create_component(document: RuntimeConfigDocument, sink: CycleEventSink) -> LLMCommitteeComponent:
    """Build the committee from database LLM settings without execution state."""
    from cryptotrader.config import AppConfig, LLMConfig, LLMModelCostConfig, ModelConfig, RetryConfig

    configured = next(item for item in document.signals.components if item.component_id == LLMCommitteeComponent.id)
    if configured.parameters:
        unknown = ", ".join(sorted(configured.parameters))
        raise ValueError(f"unsupported llm_committee parameters: {unknown}")

    runtime_llm = document.llm
    config = AppConfig(
        llm=LLMConfig(
            base_url=runtime_llm.base_url,
            streaming_models=list(runtime_llm.streaming_models),
            default_temperature=runtime_llm.default_temperature,
            timeout=runtime_llm.timeout,
            prompt_caching=runtime_llm.prompt_caching,
            retry=RetryConfig(**runtime_llm.retry.model_dump()),
            model_costs=[LLMModelCostConfig(**item.model_dump()) for item in runtime_llm.model_costs],
        ),
        models=ModelConfig(**runtime_llm.models.model_dump()),
    )
    return LLMCommitteeComponent(config, sink=sink)
