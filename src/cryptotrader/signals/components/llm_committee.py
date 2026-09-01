"""Four-domain LLM committee with an internal LangGraph debate."""
# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from cryptotrader.configuration.parameters import LlmCommitteeParameters
from cryptotrader.cycle_events import CycleEvent, NullCycleEventSink
from cryptotrader.debate.challenge import challenge_agent
from cryptotrader.debate.convergence import check_convergence, compute_divergence, debate_gate_decision
from cryptotrader.signals.component import ComponentExecutionError
from cryptotrader.signals.models import CandleRequirement, ComponentSignal, DataRequirements, SignalContext
from cryptotrader.signals.presentation import TextBlock, TimelineBlock, TimelineEntry

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from cryptotrader.cycle_events import CycleEventSink
    from cryptotrader.runtime_config.models import RuntimeConfigDocument


class CommitteeState(TypedDict):
    context: SignalContext
    analyses: dict[str, dict[str, Any]]
    timeline: list[TimelineEntry]
    debate_round: int
    debate_turns: list[dict[str, Any]]
    divergence_scores: list[float]
    debate_skipped: bool
    debate_skip_reason: str
    consensus_metrics: dict[str, float]
    final_signal: ComponentSignal | None


@dataclass(frozen=True)
class DebateSettings:
    max_rounds: int = 3
    convergence_threshold: float = 0.1
    skip_debate: bool = True
    consensus_skip_threshold: float = 0.5
    confusion_skip_threshold: float = 0.05
    confusion_max_dispersion: float = 0.2


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
    display_name = "大模型四智能体委员会"
    description = "技术、链上、新闻和宏观智能体的内部交叉辩论"

    def __init__(
        self,
        config: Any | None,
        *,
        agents: Mapping[str, Any] | None = None,
        summary: Callable[[CommitteeState], Awaitable[Mapping[str, Any]]] | None = None,
        challenger: Callable[..., Awaitable[tuple[dict, dict]]] | None = None,
        sink: CycleEventSink | None = None,
        default_timeframe: str | None = None,
        ohlcv_limit: int | None = None,
        debate=None,
        models=None,
        llm_factory: Callable[..., Any] | None = None,
        prompt_caching: bool | None = None,
    ) -> None:
        if config is None and any(value is None for value in (default_timeframe, ohlcv_limit, debate, models)):
            raise ValueError("runtime committee requires explicit market, debate, and model settings")
        self.config = config
        self.default_timeframe = config.data.default_timeframe if config is not None else default_timeframe
        self.ohlcv_limit = config.data.ohlcv_limit if config is not None else ohlcv_limit
        self.debate = config.debate if config is not None else debate
        self.models = config.models if config is not None else models
        self._legacy_agents = config.agents if config is not None else None
        self._llm_factory = llm_factory
        self._prompt_caching = prompt_caching
        self.events = sink or NullCycleEventSink()
        self.agents = dict(agents) if agents is not None else self._build_agents()
        self._summary = summary or self._summarize_with_llm
        self._challenger = challenger or self._challenge_with_llm
        self.graph = self._build_graph()

    def requirements(self) -> DataRequirements:
        return DataRequirements(
            candles=(CandleRequirement(self.default_timeframe, self.ohlcv_limit),),
            onchain=True,
            news=True,
            macro=True,
        )

    async def evaluate(self, context: SignalContext) -> ComponentSignal:
        initial: CommitteeState = {
            "context": context,
            "analyses": {},
            "timeline": [],
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
        snapshot = state["context"].snapshots[self.default_timeframe]
        names = list(self.agents)
        results = await asyncio.gather(
            *(self._analyze_one(name, self.agents[name], snapshot) for name in names),
            return_exceptions=True,
        )
        for name, result in zip(names, results, strict=True):
            if isinstance(result, BaseException):
                raise self._error("analysis", result, subject=name) from result
        return {
            "analyses": dict(zip(names, results, strict=True)),
            "timeline": [
                TimelineEntry(time=datetime.now(UTC), actor=self._actor(name), body=self._opinion(result))
                for name, result in zip(names, results, strict=True)
            ],
        }

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
                    {
                        "agent_id": name,
                        "stage": "analysis",
                        "error_type": type(error).__name__,
                    },
                )
            )
            raise
        await self.events.publish(
            CycleEvent(
                "agent_analysis_completed",
                {"agent_id": name, "stage": "analysis"},
            )
        )
        return analysis

    async def _debate_gate(self, state: CommitteeState) -> dict:
        skipped, reason, metrics = debate_gate_decision(state["analyses"], self.debate)
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
                raise self._error("debate", result, subject=name) from result
        updated = {}
        turns = list(state["debate_turns"])
        timeline = list(state["timeline"])
        for name, result in zip(names, results, strict=True):
            updated[name], turn = result
            turns.append(turn)
            body = f"第 {round_number} 轮\n{self._opinion(updated[name])}"
            for key, label in (
                ("challenge", "质疑"),
                ("response", "回应"),
                ("reasoning", "辩论意见"),
                ("new_findings", "新发现"),
                ("move", "立场变化"),
            ):
                if turn.get(key):
                    body += f"\n{label}：{turn[key]}"
            timeline.append(TimelineEntry(time=datetime.now(UTC), actor=self._actor(name), body=body))
        await self.events.publish(
            CycleEvent(
                "debate_round_completed",
                {"round_number": round_number, "stage": "debate"},
            )
        )
        return {"analyses": updated, "debate_turns": turns, "debate_round": round_number, "timeline": timeline}

    async def _convergence(self, state: CommitteeState) -> dict:
        scores = [*state["divergence_scores"], compute_divergence(state["analyses"])]
        return {"divergence_scores": scores}

    def _convergence_route(self, state: CommitteeState) -> str:
        if state["debate_round"] >= self.debate.max_rounds:
            return "converged"
        scores = state["divergence_scores"]
        if len(scores) >= 2 and check_convergence(
            scores[:-1],
            scores[-1],
            threshold=self.debate.convergence_threshold,
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
            blocks=(
                TextBlock(title="委员会观点", body=payload["reasoning"]),
                TimelineBlock(title="四智能体意见与辩论", entries=tuple(state["timeline"])),
            ),
            evaluation_reference=state["context"].evaluation_reference,
        )
        await self.events.publish(
            CycleEvent(
                "committee_summary_completed",
                {
                    "component_id": self.id,
                    "stage": "summary",
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                },
            )
        )
        return {"final_signal": signal}

    @staticmethod
    def _actor(name: str) -> str:
        return {
            "tech_agent": "技术智能体",
            "chain_agent": "链上智能体",
            "news_agent": "新闻智能体",
            "macro_agent": "宏观智能体",
        }.get(name, name)

    @staticmethod
    def _opinion(analysis: dict) -> str:
        direction = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}.get(analysis.get("direction"), "未知")
        body = f"{direction} · 置信度 {float(analysis.get('confidence', 0)):.0%}\n{analysis.get('reasoning', '')}"
        for key, label in (("key_factors", "关键因素"), ("risk_flags", "风险提示")):
            if analysis.get(key):
                body += f"\n{label}：" + "；".join(str(value) for value in analysis[key])
        return body

    async def _challenge_with_llm(
        self,
        agent_id: str,
        analysis: dict,
        others: dict[str, dict],
        context: SignalContext,
        round_number: int,
    ) -> tuple[dict, dict]:
        model = self.models.debate or self.models.fallback
        return await challenge_agent(
            agent_id,
            analysis,
            others,
            context.pair.display(),
            model,
            self.models.timeout_seconds,
            round_number,
            llm_factory=(lambda **kwargs: self._llm_factory(**kwargs, role="debate")),
            prompt_caching=bool(self._prompt_caching),
        )

    async def _summarize_with_llm(self, state: CommitteeState) -> Mapping[str, Any]:
        from cryptotrader.agents.base import create_llm, extract_content
        from cryptotrader.llm.json_retry import extract_json_with_retry

        model = self.models.committee_summary or self.models.debate or self.models.fallback
        llm = (self._llm_factory or create_llm)(
            model=model,
            temperature=0.1,
            json_mode=True,
            role="committee_summary",
        )
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
        messages = [system, HumanMessage(content=json.dumps(evidence, ensure_ascii=False, default=str))]
        if self._prompt_caching:
            from cryptotrader.llm.prompt_cache import apply_cache_control, is_anthropic_model

            if is_anthropic_model(model):
                messages = apply_cache_control(messages)
        response = await llm.ainvoke(messages)
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
            "tech_agent": self.models.tech_agent,
            "chain_agent": self.models.chain_agent,
            "news_agent": self.models.news_agent,
            "macro_agent": self.models.macro_agent,
        }
        if self._legacy_agents is None:
            analysis_model = self.models.analysis or self.models.fallback
            unresolved = tuple(agent_id for agent_id, model in models.items() if not model)
            if unresolved and not analysis_model:
                raise ValueError(f"No runtime LLM model configured for empty roles: {', '.join(unresolved)}")
            models = {agent_id: model or analysis_model for agent_id, model in models.items()}
        result = {}
        for agent_id, model in models.items():
            prompt_builder = PromptBuilder(
                agent_id=agent_id.removesuffix("_agent"),
                config_dir=project_root / "config/agents",
                skill_provider=provider,
                model=model,
            )
            if self._legacy_agents is not None:
                result[agent_id] = self._legacy_agents.build(
                    agent_id,
                    prompt_builder=prompt_builder,
                    backtest_mode=True,
                    model_override=model,
                )
                continue
            from cryptotrader.agents.chain import ChainAgent
            from cryptotrader.agents.macro import MacroAgent
            from cryptotrader.agents.news import NewsAgent
            from cryptotrader.agents.tech import TechAgent

            if agent_id == "tech_agent":
                result[agent_id] = TechAgent(
                    prompt_builder=prompt_builder,
                    model=model,
                    llm_factory=self._llm_factory,
                    prompt_caching=self._prompt_caching,
                )
            elif agent_id == "chain_agent":
                result[agent_id] = ChainAgent(
                    prompt_builder=prompt_builder,
                    model=model,
                    backtest_mode=True,
                    llm_factory=self._llm_factory,
                    prompt_caching=self._prompt_caching,
                )
            elif agent_id == "news_agent":
                result[agent_id] = NewsAgent(
                    prompt_builder=prompt_builder,
                    model=model,
                    backtest_mode=True,
                    llm_factory=self._llm_factory,
                    prompt_caching=self._prompt_caching,
                )
            else:
                result[agent_id] = MacroAgent(
                    prompt_builder=prompt_builder,
                    model=model,
                    llm_factory=self._llm_factory,
                    prompt_caching=self._prompt_caching,
                )
        return result

    def _error(
        self,
        stage: str,
        cause: BaseException,
        *,
        subject: str | None = None,
    ) -> ComponentExecutionError:
        identity = stage if subject is None else f"{stage}:{subject}"
        return ComponentExecutionError(self.id, RuntimeError(f"{identity}:{type(cause).__name__}"))


def create_component(
    document: RuntimeConfigDocument,
    sink: CycleEventSink,
    *,
    llm_factory_builder=None,
    llm_gateway_key: str = "",
) -> LLMCommitteeComponent:
    """Build the committee from database LLM settings without execution state."""
    from cryptotrader.agents.base import create_runtime_llm_factory

    configured = next(item for item in document.signals.components if item.component_id == LLMCommitteeComponent.id)
    parameters = LlmCommitteeParameters.model_validate(dict(configured.parameters))
    default_timeframe = parameters.default_timeframe
    ohlcv_limit = parameters.ohlcv_limit
    debate = DebateSettings(**parameters.debate.model_dump())
    builder = llm_factory_builder or (lambda config: create_runtime_llm_factory(config, api_key=llm_gateway_key))
    return LLMCommitteeComponent(
        None,
        sink=sink,
        default_timeframe=default_timeframe,
        ohlcv_limit=ohlcv_limit,
        debate=debate,
        models=document.llm.models,
        llm_factory=builder(document.llm),
        prompt_caching=document.llm.prompt_caching,
    )
