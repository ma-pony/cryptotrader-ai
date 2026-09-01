"""Business contracts for the database-configured built-in signal components."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cryptotrader.models import AgentAnalysis, DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData
from cryptotrader.runtime_config.models import SignalComponentConfig
from tests.factories.runtime_config import runtime_document, signal_config
from tests.factories.signal_fusion import context


def _snapshot(rows: int = 30, price: float = 100.0) -> DataSnapshot:
    closes = np.linspace(price * 0.9, price, rows)
    market = MarketData(
        pair="BTC/USDT:USDT",
        ohlcv=pd.DataFrame(
            {
                "timestamp": pd.date_range(end="2025-12-31T20:00:00Z", periods=rows, freq="4h"),
                "open": closes,
                "high": closes + 1,
                "low": closes - 1,
                "close": closes,
                "volume": np.full(rows, 10.0),
            }
        ),
        ticker={"last": price},
        funding_rate=0.0,
        orderbook_imbalance=0.0,
        volatility=0.02,
    )
    market.premium_index_5d = 0.001
    onchain = OnchainData(open_interest=1_000_000.0)
    onchain.lsr_top_count = 1.5
    macro = MacroData()
    macro.spy_btc_corr_30d = 0.3
    return DataSnapshot(datetime(2026, 1, 1, tzinfo=UTC), "BTC/USDT:USDT", market, onchain, NewsSentiment(), macro)


def _kronos_context(rows: int = 30):
    return context(price=100.0, snapshots={"4h": _snapshot(rows)})


def _features(snapshot, columns, medians):
    del snapshot, medians
    return {**dict.fromkeys(columns, 0.0), "_vol5": 0.2}


class _Predictor:
    def __init__(self, move: float = 0.1, error: Exception | None = None) -> None:
        self.move, self.error = move, error

    def predict(self, **kwargs):
        if self.error:
            raise self.error
        last_close = float(kwargs["df"]["close"].iloc[-1])
        return pd.DataFrame({"close": np.full(kwargs["pred_len"], last_close * (1 + self.move))})


def _gate(probability: float = 0.8, error: Exception | None = None):
    class _Classifier:
        def predict_proba(self, values):
            del values
            if error:
                raise error
            return np.array([[1 - probability, probability]])

    return {
        "feat_cols": ["feature"],
        "medians": {"feature": 0.0},
        "scaler": SimpleNamespace(transform=lambda values: values),
        "classifier": _Classifier(),
    }


def _kronos(*, probability=0.8, predictor=None, gate=None):
    from cryptotrader.signals.components.kronos import KronosComponent, KronosSettings

    return KronosComponent(
        KronosSettings(lookback=20, pred_len=50, ohlcv_limit=20),
        gate_loader=lambda _: gate or _gate(probability),
        predictor_loader=lambda _: predictor or _Predictor(),
        feature_computer=_features,
    )


@pytest.mark.asyncio
async def test_kronos_is_a_pure_directional_component_with_gate_and_prediction_failure_boundaries():
    from cryptotrader.signals.component import ComponentExecutionError

    accepted = await _kronos().evaluate(_kronos_context())
    rejected = await _kronos(probability=0.49).evaluate(_kronos_context())
    weak_short = await _kronos(predictor=_Predictor(-0.02)).evaluate(_kronos_context())

    assert (accepted.component_id, accepted.direction) == ("kronos", "long")
    assert "position_scale" not in accepted.details
    assert (rejected.direction, rejected.confidence) == ("neutral", 0.0)
    assert (weak_short.direction, weak_short.confidence) == ("neutral", 0.0)
    with pytest.raises(ComponentExecutionError) as raised:
        await _kronos(predictor=_Predictor(error=RuntimeError("offline"))).evaluate(_kronos_context())
    assert raised.value.stage == "prediction"
    assert "offline" not in str(raised.value)


def test_kronos_factory_reads_only_component_parameters_from_runtime_document():
    from cryptotrader.signals.components.kronos import KronosSettings, create_component

    document = runtime_document(
        signals=signal_config(
            components=(
                SignalComponentConfig(
                    component_id="kronos",
                    enabled=True,
                    weight=1.0,
                    parameters={"lookback": 111, "timeframe": "1h"},
                ),
            )
        )
    )

    component = create_component(document, sink=None)

    assert component.config == KronosSettings(lookback=111, timeframe="1h")


class _Sink:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


class _Agent:
    def __init__(self, agent_id: str, direction: str, *, error: Exception | None = None, is_mock=False):
        self.agent_id, self.direction, self.error, self.is_mock = agent_id, direction, error, is_mock

    async def analyze(self, snapshot):
        if self.error:
            raise self.error
        return AgentAnalysis(
            agent_id=self.agent_id,
            pair=snapshot.pair,
            direction=self.direction,
            confidence=0.7,
            reasoning=f"{self.agent_id} analysis",
            is_mock=self.is_mock,
        )


def _committee_context():
    return context(snapshots={"1h": _snapshot()})


def _committee_agents(**overrides):
    agents = {
        "tech_agent": _Agent("tech", "bullish"),
        "chain_agent": _Agent("chain", "bearish"),
        "news_agent": _Agent("news", "neutral"),
        "macro_agent": _Agent("macro", "bullish"),
    }
    return agents | overrides


@pytest.mark.asyncio
async def test_committee_runs_four_agents_then_internal_debate_and_captures_turns():
    from cryptotrader.signals.components.llm_committee import DebateSettings, LLMCommitteeComponent

    order, sink = [], _Sink()

    async def challenger(agent_id, analysis, others, signal_context, round_number):
        del signal_context
        order.append(("debate", agent_id))
        return analysis, {"round": round_number, "from": agent_id, "to": next(iter(others)), "move": "keep"}

    async def summary(state):
        order.append(("summary", "committee"))
        assert len(state["debate_turns"]) == 4
        return {"direction": "long", "confidence": 0.65, "reasoning": "committee summary"}

    component = LLMCommitteeComponent(
        None,
        agents=_committee_agents(),
        summary=summary,
        challenger=challenger,
        sink=sink,
        default_timeframe="1h",
        ohlcv_limit=100,
        debate=DebateSettings(max_rounds=1, skip_debate=False, consensus_skip_threshold=0.5),
        models=object(),
    )
    result = await component.evaluate(_committee_context())

    assert (result.component_id, result.direction, result.confidence) == ("llm_committee", "long", 0.65)
    assert len(result.details["analyses"]) == 4
    assert len(result.details["debate_turns"]) == 4
    assert order[-1] == ("summary", "committee")
    assert [event.name for event in sink.events].count("agent_analysis_completed") == 4


@pytest.mark.asyncio
async def test_committee_fails_closed_for_agent_or_debate_failures_without_leaking_error_payload():
    from cryptotrader.signals.component import ComponentExecutionError
    from cryptotrader.signals.components.llm_committee import DebateSettings, LLMCommitteeComponent

    marker, sink = "provider-secret", _Sink()
    component = LLMCommitteeComponent(
        None,
        agents=_committee_agents(tech_agent=_Agent("tech", "bullish", error=RuntimeError(marker))),
        summary=lambda _: None,
        sink=sink,
        default_timeframe="1h",
        ohlcv_limit=100,
        debate=DebateSettings(),
        models=object(),
    )
    with pytest.raises(ComponentExecutionError, match="analysis:tech_agent"):
        await component.evaluate(_committee_context())
    assert marker not in repr(sink.events)


@pytest.mark.asyncio
async def test_committee_initial_analyses_are_parallel():
    from cryptotrader.signals.components.llm_committee import DebateSettings, LLMCommitteeComponent

    starts = []

    class _DelayedAgent(_Agent):
        async def analyze(self, snapshot):
            starts.append(asyncio.get_running_loop().time())
            await asyncio.sleep(0.02)
            return await super().analyze(snapshot)

    component = LLMCommitteeComponent(
        None,
        agents={name: _DelayedAgent(name, "bullish") for name in _committee_agents()},
        summary=_neutral_summary,
        sink=_Sink(),
        default_timeframe="1h",
        ohlcv_limit=100,
        debate=DebateSettings(skip_debate=True),
        models=object(),
    )
    await component.evaluate(_committee_context())
    assert len(starts) == 4
    assert max(starts) - min(starts) < 0.01


async def _neutral_summary(_):
    return {"direction": "neutral", "confidence": 0.0, "reasoning": "skip"}
