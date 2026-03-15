"""LLM parallel call timeout protection tests (task 3.1).

Coverage:
- nodes/agents.py: _run_agent() timeout degrades to is_mock=True analysis
- nodes/debate.py: debate_round() timeout degrades to original analysis (non-blocking)
- return_exceptions=True: exceptions are logged per-item, not silently ignored
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.models import (
    AgentAnalysis,
    DataSnapshot,
    MacroData,
    MarketData,
    NewsSentiment,
    OnchainData,
)

# ── Test helper factories ──


def _make_snapshot(pair: str = "BTC/USDT", price: float = 50000.0) -> DataSnapshot:
    return DataSnapshot(
        timestamp=None,
        pair=pair,
        market=MarketData(
            pair=pair,
            ohlcv=None,
            ticker={"last": price, "baseVolume": 1000},
            funding_rate=0.01,
            orderbook_imbalance=0.5,
            volatility=0.02,
        ),
        onchain=OnchainData(),
        news=NewsSentiment(),
        macro=MacroData(),
    )


def _base_state(pair: str = "BTC/USDT", price: float = 50000.0, **extra_data) -> dict:
    snapshot = _make_snapshot(pair, price)
    data: dict = {
        "snapshot": snapshot,
        "snapshot_summary": {
            "pair": pair,
            "price": price,
            "funding_rate": 0.01,
            "volatility": 0.02,
            "orderbook_imbalance": 0.5,
        },
        "experience": "",
        **extra_data,
    }
    return {
        "messages": [],
        "data": data,
        "metadata": {
            "pair": pair,
            "engine": "paper",
            "models": {},
            "analysis_model": "gpt-4o-mini",
            "debate_model": "gpt-4o-mini",
            "verdict_model": "gpt-4o-mini",
        },
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }


def _make_mock_cfg(timeout_seconds: float = 0.05) -> MagicMock:
    """Build a mock AppConfig with the given timeout_seconds."""
    cfg = MagicMock()
    cfg.models.timeout_seconds = timeout_seconds
    cfg.models.debate = "gpt-4o-mini"
    cfg.models.fallback = "gpt-4o-mini"
    cfg.models.analysis = "gpt-4o-mini"
    return cfg


def _mock_llm_hanging() -> MagicMock:
    """Return a mock LLM whose ainvoke hangs indefinitely."""

    async def _hang(*args, **kwargs):
        await asyncio.sleep(10)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=_hang)
    return llm


def _mock_llm_raising(exc: Exception) -> MagicMock:
    """Return a mock LLM whose ainvoke raises the given exception immediately."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=exc)
    return llm


# ── nodes/agents.py timeout tests ──


@pytest.mark.asyncio
async def test_agent_timeout_degrades_to_mock():
    """When the LLM call times out, _run_agent should degrade to is_mock=True."""
    from cryptotrader.nodes.agents import tech_analyze

    async def _hang(*args, **kwargs):
        await asyncio.sleep(10)  # far exceeds 0.05 s timeout

    with (
        patch("cryptotrader.agents.tech.TechAgent") as mock_agent_cls,
        patch("cryptotrader.config.load_config", return_value=_make_mock_cfg(0.05)),
    ):
        instance = mock_agent_cls.return_value
        instance.analyze = AsyncMock(side_effect=_hang)
        result = await tech_analyze(_base_state())

    analysis = result["data"]["analyses"]["tech_agent"]
    assert analysis["is_mock"] is True, "Timeout should degrade to is_mock=True"


@pytest.mark.asyncio
async def test_agent_timeout_does_not_block_pipeline():
    """One agent timing out must not block another agent from completing normally."""
    from cryptotrader.nodes.agents import chain_analyze, tech_analyze

    async def _hang(*args, **kwargs):
        await asyncio.sleep(10)

    mock_fast_analysis = AgentAnalysis(
        agent_id="chain_agent",
        pair="BTC/USDT",
        direction="neutral",
        confidence=0.5,
        reasoning="Fast",
        is_mock=False,
    )

    with (
        patch("cryptotrader.agents.tech.TechAgent") as mock_tech,
        patch("cryptotrader.agents.chain.ChainAgent") as mock_chain,
        patch("cryptotrader.config.load_config", return_value=_make_mock_cfg(0.05)),
    ):
        mock_tech.return_value.analyze = AsyncMock(side_effect=_hang)
        mock_chain.return_value.analyze = AsyncMock(return_value=mock_fast_analysis)

        tech_result = await tech_analyze(_base_state())
        chain_result = await chain_analyze(_base_state())

    # tech times out -> mock; chain succeeds -> not mock
    assert tech_result["data"]["analyses"]["tech_agent"]["is_mock"] is True
    assert chain_result["data"]["analyses"]["chain_agent"]["is_mock"] is False


@pytest.mark.asyncio
async def test_agent_non_timeout_exception_is_logged(caplog):
    """Non-timeout exceptions (e.g. API errors) must be logged with WARNING, not silently swallowed."""
    import logging

    from cryptotrader.nodes.agents import tech_analyze

    with (
        patch("cryptotrader.agents.tech.TechAgent") as mock_agent_cls,
        patch("cryptotrader.config.load_config", return_value=_make_mock_cfg(60)),
    ):
        instance = mock_agent_cls.return_value
        instance.analyze = AsyncMock(side_effect=RuntimeError("API connection refused"))

        with caplog.at_level(logging.WARNING, logger="cryptotrader.nodes.agents"):
            result = await tech_analyze(_base_state())

    analysis = result["data"]["analyses"]["tech_agent"]
    assert analysis["is_mock"] is True
    assert any(r.levelno >= logging.WARNING for r in caplog.records), "Expected WARNING log entry"


@pytest.mark.asyncio
async def test_agent_timeout_warning_logged(caplog):
    """A timeout should produce a WARNING log containing 'LLM timeout'."""
    import logging

    from cryptotrader.nodes.agents import tech_analyze

    async def _hang(*args, **kwargs):
        await asyncio.sleep(10)

    with (
        patch("cryptotrader.agents.tech.TechAgent") as mock_agent_cls,
        patch("cryptotrader.config.load_config", return_value=_make_mock_cfg(0.05)),
    ):
        instance = mock_agent_cls.return_value
        instance.analyze = AsyncMock(side_effect=_hang)

        with caplog.at_level(logging.WARNING, logger="cryptotrader.nodes.agents"):
            await tech_analyze(_base_state())

    messages = [r.message for r in caplog.records]
    assert any("LLM timeout" in m or "timeout" in m.lower() for m in messages), (
        f"Expected timeout warning log, got: {messages}"
    )


# ── nodes/debate.py timeout tests ──
# Note: debate tests patch "cryptotrader.nodes.debate.create_llm" directly to avoid
# ChatOpenAI instantiation failing due to mock cfg values (e.g. base_url validation).


@pytest.mark.asyncio
async def test_debate_round_timeout_degrades_gracefully():
    """When LLM times out inside debate_round, original analyses are preserved and structure is intact."""
    from cryptotrader.nodes.debate import debate_round

    analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.8, "reasoning": "up"},
        "chain_agent": {"direction": "bearish", "confidence": 0.7, "reasoning": "down"},
    }
    state = _base_state(analyses=analyses)

    with (
        patch("cryptotrader.nodes.debate.create_llm", return_value=_mock_llm_hanging()),
        patch("cryptotrader.config.load_config", return_value=_make_mock_cfg(0.05)),
    ):
        result = await debate_round(state)

    assert "data" in result
    assert "analyses" in result["data"]
    assert result["debate_round"] == 1

    updated = result["data"]["analyses"]
    assert "tech_agent" in updated
    assert "chain_agent" in updated
    # Original analyses preserved on timeout
    assert updated["tech_agent"]["direction"] == "bullish"
    assert updated["chain_agent"]["direction"] == "bearish"


@pytest.mark.asyncio
async def test_debate_round_timeout_warning_logged(caplog):
    """debate_round timeout must produce a WARNING log containing 'timeout'."""
    import logging

    from cryptotrader.nodes.debate import debate_round

    analyses = {"tech_agent": {"direction": "bullish", "confidence": 0.8, "reasoning": "up"}}
    state = _base_state(analyses=analyses)

    with (
        patch("cryptotrader.nodes.debate.create_llm", return_value=_mock_llm_hanging()),
        patch("cryptotrader.config.load_config", return_value=_make_mock_cfg(0.05)),
        caplog.at_level(logging.WARNING, logger="cryptotrader.nodes.debate"),
    ):
        await debate_round(state)

    messages = [r.message for r in caplog.records]
    assert any("timeout" in m.lower() or "LLM timeout" in m for m in messages), (
        f"Expected timeout warning log, got: {messages}"
    )


@pytest.mark.asyncio
async def test_debate_round_non_timeout_exception_logged(caplog):
    """Non-timeout exceptions in debate_round must be logged and original analyses preserved."""
    import logging

    from cryptotrader.nodes.debate import debate_round

    analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.8, "reasoning": "up"},
        "chain_agent": {"direction": "bearish", "confidence": 0.7, "reasoning": "down"},
    }
    state = _base_state(analyses=analyses)

    with (
        patch("cryptotrader.nodes.debate.create_llm", return_value=_mock_llm_raising(ValueError("LLM error"))),
        patch("cryptotrader.config.load_config", return_value=_make_mock_cfg(60)),
        caplog.at_level(logging.WARNING, logger="cryptotrader.nodes.debate"),
    ):
        result = await debate_round(state)

    updated = result["data"]["analyses"]
    assert updated["tech_agent"]["direction"] == "bullish"
    assert updated["chain_agent"]["direction"] == "bearish"
    assert any(r.levelno >= logging.WARNING for r in caplog.records), "Expected WARNING log"


@pytest.mark.asyncio
async def test_debate_round_gather_multiple_exceptions_all_logged(caplog):
    """When gather returns multiple exceptions, each must be individually logged."""
    import logging

    from cryptotrader.nodes.debate import debate_round

    analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.8, "reasoning": "up"},
        "chain_agent": {"direction": "bearish", "confidence": 0.7, "reasoning": "down"},
        "news_agent": {"direction": "neutral", "confidence": 0.5, "reasoning": "uncertain"},
    }
    state = _base_state(analyses=analyses)

    call_count = 0

    async def _fail_twice_then_succeed(*args, **kwargs):
        """Fail on first two calls, succeed on third."""
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise RuntimeError(f"LLM error on call {call_count}")
        from langchain_core.messages import AIMessage

        return AIMessage(
            content='{"direction": "neutral", "confidence": 0.5, "reasoning": "ok",'
            ' "key_factors": [], "risk_flags": [], "new_findings": ""}'
        )

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=_fail_twice_then_succeed)

    with (
        patch("cryptotrader.nodes.debate.create_llm", return_value=mock_llm),
        patch("cryptotrader.config.load_config", return_value=_make_mock_cfg(60)),
        caplog.at_level(logging.WARNING, logger="cryptotrader.nodes.debate"),
    ):
        result = await debate_round(state)

    assert "analyses" in result["data"]
    warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warn_records) >= 2, f"Expected >= 2 WARNING entries, got: {len(warn_records)}"


@pytest.mark.asyncio
async def test_debate_round_partial_timeout_preserves_successful():
    """With one agent timing out and one succeeding, both results must be present in output."""
    from cryptotrader.nodes.debate import debate_round

    analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.8, "reasoning": "original_tech"},
        "chain_agent": {"direction": "bearish", "confidence": 0.7, "reasoning": "original_chain"},
    }
    state = _base_state(analyses=analyses)

    call_count = 0

    from langchain_core.messages import AIMessage

    async def _selective_hang(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await asyncio.sleep(10)  # first agent times out
        return AIMessage(
            content='{"direction": "neutral", "confidence": 0.65, "reasoning": "revised",'
            ' "key_factors": [], "risk_flags": [], "new_findings": ""}'
        )

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=_selective_hang)

    with (
        patch("cryptotrader.nodes.debate.create_llm", return_value=mock_llm),
        patch("cryptotrader.config.load_config", return_value=_make_mock_cfg(0.05)),
    ):
        result = await debate_round(state)

    updated = result["data"]["analyses"]
    assert "tech_agent" in updated
    assert "chain_agent" in updated
    assert result["debate_round"] == 1
