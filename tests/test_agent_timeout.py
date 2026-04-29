"""Tests for per-agent timeout — SC-006."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from cryptotrader.config import AgentConfig, AgentsConfig, AppConfig, ModelConfig
from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData


def _make_snapshot(pair: str = "BTC/USDT") -> DataSnapshot:
    return DataSnapshot(
        timestamp=None,
        pair=pair,
        market=MarketData(
            pair=pair,
            ohlcv=None,
            ticker={"last": 50000.0},
            funding_rate=0.01,
            orderbook_imbalance=0.5,
            volatility=0.02,
        ),
        onchain=OnchainData(),
        news=NewsSentiment(),
        macro=MacroData(),
    )


def _make_state(pair: str = "BTC/USDT") -> dict:
    return {
        "messages": [],
        "data": {
            "snapshot": _make_snapshot(pair),
            "snapshot_hash": None,
            "prev_snapshot_hash": None,
            "prev_analyses": {},
        },
        "metadata": {
            "pair": pair,
            "engine": "paper",
            "backtest_mode": False,
            "models": {},
            "analysis_model": "",
        },
        "debate_round": 0,
        "max_debate_rounds": 3,
        "divergence_scores": [],
    }


async def _hang_forever(*args, **kwargs):
    await asyncio.sleep(999)


@pytest.mark.asyncio
async def test_agent_timeout_uses_per_agent_config():
    """Agent with timeout_seconds=1 should timeout and return mock result."""
    agents = {
        "tech_agent": AgentConfig(agent_id="tech_agent", timeout_seconds=1),
    }
    cfg = AppConfig(
        agents=AgentsConfig(_agents=agents),
        models=ModelConfig(timeout_seconds=90),
    )

    state = _make_state()
    with (
        patch("cryptotrader.config.load_config", return_value=cfg),
        patch("cryptotrader.agents.base.create_llm"),
        patch("cryptotrader.agents.tech.TechAgent.analyze", new=_hang_forever),
    ):
        from cryptotrader.nodes.agents import _run_agent

        result = await asyncio.wait_for(_run_agent("tech_agent", state), timeout=10)

    analysis = result["data"]["analyses"]["tech_agent"]
    assert analysis["is_mock"] is True
    assert analysis["direction"] == "neutral"


@pytest.mark.asyncio
async def test_agent_timeout_falls_back_to_global():
    """Agent without timeout_seconds uses global models.timeout_seconds."""
    cfg = AppConfig(
        agents=AgentsConfig(),
        models=ModelConfig(timeout_seconds=1),
    )

    state = _make_state()
    with (
        patch("cryptotrader.config.load_config", return_value=cfg),
        patch("cryptotrader.agents.base.create_llm"),
        patch("cryptotrader.agents.tech.TechAgent.analyze", new=_hang_forever),
    ):
        from cryptotrader.nodes.agents import _run_agent

        result = await asyncio.wait_for(_run_agent("tech_agent", state), timeout=10)

    analysis = result["data"]["analyses"]["tech_agent"]
    assert analysis["is_mock"] is True
