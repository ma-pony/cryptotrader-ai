"""LLM 四智能体委员会保留内部辩论, 外部只输出一个组件信号。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pandas as pd
import pytest

from cryptotrader.config import AppConfig, DebateConfig
from cryptotrader.models import AgentAnalysis, DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData
from tests.factories.signal_fusion import context


class RecordingSink:
    def __init__(self) -> None:
        self.events = []

    async def publish(self, event) -> None:
        self.events.append(event)


class FakeAgent:
    def __init__(self, agent_id: str, direction="bullish", confidence=0.7, error=None, is_mock=False) -> None:
        self.agent_id = agent_id
        self.direction = direction
        self.confidence = confidence
        self.error = error
        self.is_mock = is_mock

    async def analyze(self, snapshot):
        if self.error is not None:
            raise self.error
        return AgentAnalysis(
            agent_id=self.agent_id,
            pair=snapshot.pair,
            direction=self.direction,
            confidence=self.confidence,
            reasoning=f"{self.agent_id} initial",
            is_mock=self.is_mock,
        )


def _context():
    rows = 30
    frame = pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.0] * rows,
            "volume": [10.0] * rows,
        }
    )
    snapshot = DataSnapshot(
        datetime(2026, 1, 1, tzinfo=UTC),
        "BTC/USDT:USDT",
        MarketData("BTC/USDT:USDT", frame, {"last": 100.0}, 0.0, 0.0, 0.0),
        OnchainData(),
        NewsSentiment(),
        MacroData(),
    )
    return context(snapshots={"1h": snapshot})


def _agents(**overrides):
    defaults = {
        "tech_agent": FakeAgent("tech", "bullish", 0.8),
        "chain_agent": FakeAgent("chain", "bearish", 0.7),
        "news_agent": FakeAgent("news", "neutral", 0.2),
        "macro_agent": FakeAgent("macro", "bullish", 0.6),
    }
    defaults.update(overrides)
    return defaults


def _config(*, skip_debate=False, rounds=1):
    return AppConfig(
        debate=DebateConfig(
            max_rounds=rounds,
            skip_debate=skip_debate,
            consensus_skip_threshold=0.5,
        )
    )


async def _summary(state):
    return {"direction": "long", "confidence": 0.65, "reasoning": "committee summary"}


@pytest.mark.asyncio
async def test_committee_runs_four_agents_and_debate_before_summary():
    from cryptotrader.signals.components.llm_committee import LLMCommitteeComponent

    order = []

    async def challenger(agent_id, analysis, others, signal_context, round_number):
        order.append(("debate", agent_id))
        updated = dict(analysis)
        turn = {
            "round": round_number,
            "from": agent_id,
            "to": next(iter(others)),
            "before": {"direction": analysis["direction"], "confidence": analysis["confidence"]},
            "after": {"direction": analysis["direction"], "confidence": analysis["confidence"]},
            "move": "保持",
            "reasoning": "debated",
            "new_findings": "",
            "errored": False,
        }
        return updated, turn

    async def summary(state):
        order.append(("summary", "committee"))
        return await _summary(state)

    sink = RecordingSink()
    component = LLMCommitteeComponent(
        _config(),
        agents=_agents(),
        summary=summary,
        challenger=challenger,
        sink=sink,
    )

    result = await component.evaluate(_context())

    assert result.component_id == "llm_committee"
    assert (result.direction, result.confidence, result.reasoning) == ("long", 0.65, "committee summary")
    assert len(result.details["analyses"]) == 4
    assert len(result.details["debate_turns"]) == 4
    assert order[-1][0] == "summary"
    assert len([event for event in sink.events if event.name == "agent_analysis_completed"]) == 4
    assert any(event.name == "debate_round_completed" for event in sink.events)


@pytest.mark.asyncio
async def test_analysis_agents_run_in_parallel():
    from cryptotrader.signals.components.llm_committee import LLMCommitteeComponent

    starts = []

    class DelayedAgent(FakeAgent):
        async def analyze(self, snapshot):
            starts.append(asyncio.get_running_loop().time())
            await asyncio.sleep(0.02)
            return await super().analyze(snapshot)

    agents = {name: DelayedAgent(name) for name in _agents()}
    component = LLMCommitteeComponent(
        _config(skip_debate=True),
        agents=agents,
        summary=_summary,
        sink=RecordingSink(),
    )

    await component.evaluate(_context())

    assert len(starts) == 4
    assert max(starts) - min(starts) < 0.01


@pytest.mark.asyncio
async def test_one_agent_failure_fails_whole_component():
    from cryptotrader.signals.component import ComponentExecutionError
    from cryptotrader.signals.components.llm_committee import LLMCommitteeComponent

    component = LLMCommitteeComponent(
        _config(),
        agents=_agents(tech_agent=FakeAgent("tech", error=RuntimeError("timeout"))),
        summary=_summary,
        sink=RecordingSink(),
    )

    with pytest.raises(ComponentExecutionError, match="tech_agent"):
        await component.evaluate(_context())


@pytest.mark.asyncio
async def test_mock_agent_result_is_a_component_failure():
    from cryptotrader.signals.component import ComponentExecutionError
    from cryptotrader.signals.components.llm_committee import LLMCommitteeComponent

    component = LLMCommitteeComponent(
        _config(),
        agents=_agents(news_agent=FakeAgent("news", is_mock=True)),
        summary=_summary,
        sink=RecordingSink(),
    )

    with pytest.raises(ComponentExecutionError, match="news_agent"):
        await component.evaluate(_context())


@pytest.mark.asyncio
async def test_debate_failure_fails_whole_component():
    from cryptotrader.signals.component import ComponentExecutionError
    from cryptotrader.signals.components.llm_committee import LLMCommitteeComponent

    async def fail(*args):
        raise RuntimeError("debate timeout")

    component = LLMCommitteeComponent(
        _config(),
        agents=_agents(),
        summary=_summary,
        challenger=fail,
        sink=RecordingSink(),
    )

    with pytest.raises(ComponentExecutionError, match="debate"):
        await component.evaluate(_context())


@pytest.mark.asyncio
async def test_committee_events_do_not_expose_provider_payload_or_error_secrets():
    from cryptotrader.signals.component import ComponentExecutionError
    from cryptotrader.signals.components.llm_committee import LLMCommitteeComponent

    sensitive_marker = "provider-sensitive-marker"
    sink = RecordingSink()
    component = LLMCommitteeComponent(
        _config(),
        agents=_agents(tech_agent=FakeAgent("tech", error=RuntimeError(sensitive_marker))),
        summary=_summary,
        sink=sink,
    )

    with pytest.raises(ComponentExecutionError):
        await component.evaluate(_context())

    assert sensitive_marker not in repr(sink.events)
    failed = next(event for event in sink.events if event.name == "committee_agent_failed")
    assert failed.data == {
        "agent_id": "tech_agent",
        "stage": "analysis",
        "error_type": "RuntimeError",
    }


@pytest.mark.asyncio
async def test_completed_committee_events_contain_only_safe_metadata():
    from cryptotrader.signals.components.llm_committee import LLMCommitteeComponent

    sensitive_marker = "provider-sensitive-marker"

    class SecretAgent(FakeAgent):
        async def analyze(self, snapshot):
            analysis = await super().analyze(snapshot)
            return analysis.__class__(
                agent_id=analysis.agent_id,
                pair=analysis.pair,
                direction=analysis.direction,
                confidence=analysis.confidence,
                reasoning=sensitive_marker,
                is_mock=False,
            )

    async def challenger(agent_id, analysis, others, signal_context, round_number):
        return {**analysis, "reasoning": sensitive_marker}, {"reasoning": sensitive_marker}

    sink = RecordingSink()
    component = LLMCommitteeComponent(
        _config(),
        agents={name: SecretAgent(name) for name in _agents()},
        summary=_summary,
        challenger=challenger,
        sink=sink,
    )

    await component.evaluate(_context())

    assert sensitive_marker not in repr(sink.events)
    assert all("analysis" not in event.data for event in sink.events)
    assert all("analyses" not in event.data for event in sink.events)
    summary = next(event for event in sink.events if event.name == "committee_summary_completed")
    assert summary.data == {
        "component_id": "llm_committee",
        "stage": "summary",
        "direction": "long",
        "confidence": 0.65,
    }


def test_summary_payload_contains_only_component_signal_fields():
    from cryptotrader.signals.components.llm_committee import normalize_summary_payload

    payload = normalize_summary_payload(
        {
            "direction": "short",
            "confidence": 0.6,
            "reasoning": "committee summary",
            "position_scale": 0.9,
            "stop_loss": 90,
        }
    )

    assert payload == {"direction": "short", "confidence": 0.6, "reasoning": "committee summary"}


def test_requirements_use_shared_snapshot_data_only():
    from cryptotrader.signals.components.llm_committee import LLMCommitteeComponent

    requirements = LLMCommitteeComponent(
        _config(),
        agents=_agents(),
        summary=_summary,
        sink=RecordingSink(),
    ).requirements()

    assert [(item.timeframe, item.limit) for item in requirements.candles] == [("1h", 100)]
    assert requirements.onchain is True
    assert requirements.news is True
    assert requirements.macro is True
    assert requirements.kronos_aux is False
