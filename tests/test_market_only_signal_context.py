"""Signal components consume market evidence without execution-account state."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cryptotrader.pair import Pair
from cryptotrader.signals.models import CandleRequirement, DataRequirements
from tests.factories.runtime_config import market_config


class _MarketOnlyGuard:
    _forbidden = {"exchange_id", "equity", "current_position", "portfolio"}

    def __init__(self, context) -> None:
        for name in (
            "pair",
            "as_of",
            "market_type",
            "current_price",
            "atr",
            "snapshots",
        ):
            setattr(self, name, getattr(context, name))
        self.market_data_source_id = "fixture-market"

    def __getattr__(self, name):
        if name in self._forbidden:
            raise AssertionError(f"signal component accessed execution field {name}")
        raise AttributeError(name)


@pytest.mark.asyncio
async def test_default_market_source_builds_context_from_explicit_public_source_without_network():
    from cryptotrader.market_sources.default import DefaultMarketDataSource
    from tests.test_kronos_component import _snapshot

    class FakeMarketCollector:
        def __init__(self) -> None:
            self.calls = []

        async def collect(self, *args):
            self.calls.append(args)
            return _snapshot().market

    class FakeAggregator:
        def __init__(self) -> None:
            self.calls = []
            self.market = FakeMarketCollector()

        async def collect(self, **kwargs):
            self.calls.append(kwargs)
            return _snapshot()

    aggregator = FakeAggregator()
    source = DefaultMarketDataSource(
        market_config(parameters={"exchange_id": "binance"}),
        aggregator=aggregator,
        clock=lambda: as_of,
    )
    as_of = datetime(2026, 1, 1, tzinfo=UTC)

    context = await source.collect(
        Pair.parse("BTC/USDT:USDT"),
        as_of,
        DataRequirements(
            candles=(CandleRequirement("4h", 20), CandleRequirement("1h", 20)),
            kronos_aux=True,
        ),
    )

    assert context.market_data_source_id == "default"
    assert context.as_of == as_of
    assert tuple(context.snapshots) == ("4h", "1h")
    assert aggregator.calls == [
        {
            "pair": "BTC/USDT:USDT",
            "exchange_id": "binance",
            "timeframe": "4h",
            "limit": 20,
            "backtest_mode": False,
            "kronos_aux": True,
            "kronos_aux_symbol": "BTCUSDT",
        }
    ]
    assert aggregator.market.calls == [("BTC/USDT:USDT", "binance", "1h", 20)]


@pytest.mark.asyncio
async def test_default_market_source_rejects_historical_as_of_before_collection():
    from cryptotrader.market_sources.default import DefaultMarketDataSource

    class FakeAggregator:
        def __init__(self) -> None:
            self.calls = []
            self.market = object()

        async def collect(self, **kwargs):
            self.calls.append(kwargs)
            raise AssertionError("historical request reached live collection")

    aggregator = FakeAggregator()
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    source = DefaultMarketDataSource(
        market_config(parameters={"exchange_id": "binance"}),
        aggregator=aggregator,
        clock=lambda: now,
    )

    with pytest.raises(ValueError, match="live-only"):
        await source.collect(
            Pair.parse("BTC/USDT:USDT"),
            datetime(2026, 1, 1, 11, 55, tzinfo=UTC),
            DataRequirements(candles=(CandleRequirement("1h", 20),)),
        )

    assert aggregator.calls == []


@pytest.mark.asyncio
async def test_kronos_does_not_read_execution_account_fields():
    from tests.test_kronos_component import _component, _context

    result = await _component(probability=0.49).evaluate(_MarketOnlyGuard(_context()))

    assert result.component_id == "kronos"
    assert result.direction == "neutral"


@pytest.mark.asyncio
async def test_llm_committee_debate_does_not_read_execution_account_fields():
    from cryptotrader.signals.components.llm_committee import LLMCommitteeComponent
    from tests.test_llm_committee_component import RecordingSink, _agents, _config, _context, _summary

    debate_calls = []

    async def challenger(agent_id, analysis, others, context, round_number):
        debate_calls.append((agent_id, context.market_data_source_id))
        return analysis, {
            "round": round_number,
            "from": agent_id,
            "to": next(iter(others)),
            "before": {"direction": analysis["direction"], "confidence": analysis["confidence"]},
            "after": {"direction": analysis["direction"], "confidence": analysis["confidence"]},
            "move": "keep",
            "reasoning": "market evidence only",
            "new_findings": "",
            "errored": False,
        }

    component = LLMCommitteeComponent(
        _config(rounds=1),
        agents=_agents(),
        summary=_summary,
        challenger=challenger,
        sink=RecordingSink(),
    )

    result = await component.evaluate(_MarketOnlyGuard(_context()))

    assert result.component_id == "llm_committee"
    assert debate_calls == [
        ("tech_agent", "fixture-market"),
        ("chain_agent", "fixture-market"),
        ("news_agent", "fixture-market"),
        ("macro_agent", "fixture-market"),
    ]
