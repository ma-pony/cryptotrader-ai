"""Tests for LangGraph node modules under cryptotrader.nodes.*"""

from __future__ import annotations

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


def _make_snapshot(pair="BTC/USDT", price=50000.0):
    """Build a minimal DataSnapshot for testing."""
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


def _base_state(pair="BTC/USDT", price=50000.0, **extra_data):
    """Build minimal ArenaState dict."""
    snapshot = _make_snapshot(pair, price)
    data = {
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


# ── nodes/agents.py ──


@pytest.mark.asyncio
async def test_run_agent_tech():
    """tech_analyze calls TechAgent and returns analyses dict."""
    from cryptotrader.nodes.agents import tech_analyze

    mock_analysis = AgentAnalysis(
        agent_id="tech_agent",
        pair="BTC/USDT",
        direction="bullish",
        confidence=0.8,
        reasoning="Strong momentum",
        key_factors=["RSI above 70"],
        risk_flags=[],
    )

    with patch("cryptotrader.agents.tech.TechAgent") as mock_agent:
        instance = mock_agent.return_value
        instance.analyze = AsyncMock(return_value=mock_analysis)
        result = await tech_analyze(_base_state())

    analyses = result["data"]["analyses"]
    assert "tech_agent" in analyses
    assert analyses["tech_agent"]["direction"] == "bullish"
    assert analyses["tech_agent"]["confidence"] == 0.8


@pytest.mark.asyncio
async def test_run_agent_uses_model_from_config():
    """Agent uses model name from state metadata."""
    from cryptotrader.nodes.agents import chain_analyze

    state = _base_state()
    state["metadata"]["models"]["chain_agent"] = "claude-3-haiku"

    mock_analysis = AgentAnalysis(
        agent_id="chain_agent",
        pair="BTC/USDT",
        direction="neutral",
        confidence=0.5,
        reasoning="Mixed signals",
    )

    with patch("cryptotrader.agents.chain.ChainAgent") as mock_agent:
        instance = mock_agent.return_value
        instance.analyze = AsyncMock(return_value=mock_analysis)
        await chain_analyze(state)
        mock_agent.assert_called_once_with(model="claude-3-haiku", backtest_mode=False)


@pytest.mark.asyncio
async def test_run_agent_all_four():
    """All four agent nodes produce correct key in analyses dict."""
    from cryptotrader.nodes.agents import chain_analyze, macro_analyze, news_analyze, tech_analyze

    agent_modules = {
        "tech_agent": ("cryptotrader.agents.tech", "TechAgent"),
        "chain_agent": ("cryptotrader.agents.chain", "ChainAgent"),
        "news_agent": ("cryptotrader.agents.news", "NewsAgent"),
        "macro_agent": ("cryptotrader.agents.macro", "MacroAgent"),
    }
    funcs = {
        "tech_agent": tech_analyze,
        "chain_agent": chain_analyze,
        "news_agent": news_analyze,
        "macro_agent": macro_analyze,
    }

    for key, (module, cls_name) in agent_modules.items():
        mock_analysis = AgentAnalysis(
            agent_id=key, pair="BTC/USDT", direction="bearish", confidence=0.6, reasoning="test"
        )
        with patch(f"{module}.{cls_name}") as m:
            m.return_value.analyze = AsyncMock(return_value=mock_analysis)
            result = await funcs[key](_base_state())
            assert key in result["data"]["analyses"]


# ── nodes/data.py ──


@pytest.mark.asyncio
async def test_collect_snapshot_reuses_existing():
    """collect_snapshot reuses pre-provided snapshot in state."""
    from cryptotrader.nodes.data import collect_snapshot

    state = _base_state()
    result = await collect_snapshot(state)
    summary = result["data"]["snapshot_summary"]
    assert summary["pair"] == "BTC/USDT"
    assert summary["price"] == 50000.0


@pytest.mark.asyncio
async def test_verbal_reinforcement():
    """verbal_reinforcement injects experience and calibration."""
    from cryptotrader.nodes.data import verbal_reinforcement

    state = _base_state()

    with (
        patch("cryptotrader.learning.verbal.get_experience", new_callable=AsyncMock, return_value=[]),
        patch("cryptotrader.journal.calibrate.detect_biases", new_callable=AsyncMock, return_value={}),
        patch(
            "cryptotrader.journal.calibrate.generate_per_agent_corrections",
            return_value={"tech_agent": "OVERCONFIDENT"},
        ),
        patch(
            "cryptotrader.journal.calibrate.generate_verdict_calibration",
            return_value="calibrate: reduce overconfidence",
        ),
    ):
        result = await verbal_reinforcement(state)

    assert "historical_cases" in result["data"]
    assert "regime_tags" in result["data"]
    assert result["data"]["agent_corrections"] == {"tech_agent": "OVERCONFIDENT"}
    assert result["data"]["verdict_calibration"] == "calibrate: reduce overconfidence"


# ── nodes/debate.py ──


@pytest.mark.asyncio
async def test_debate_round():
    """debate_round increments round counter and stores revised analyses."""
    from cryptotrader.nodes.debate import debate_round

    analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.8, "reasoning": "up"},
        "chain_agent": {"direction": "bearish", "confidence": 0.7, "reasoning": "down"},
    }
    state = _base_state(analyses=analyses)

    from langchain_core.messages import AIMessage

    mock_ai_msg = AIMessage(
        content='{"direction": "bullish", "confidence": 0.75, "reasoning": "revised", '
        '"key_factors": [], "risk_flags": [], "new_findings": "cross-check"}'
    )

    with patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock, return_value=mock_ai_msg):
        result = await debate_round(state)

    assert result["debate_round"] == 1
    assert "analyses" in result["data"]


@pytest.mark.asyncio
async def test_convergence_router_max_rounds():
    """convergence_router returns 'converged' when max rounds reached."""
    from cryptotrader.nodes.debate import convergence_router

    state = _base_state()
    state["debate_round"] = 3
    state["max_debate_rounds"] = 3
    assert convergence_router(state) == "converged"


@pytest.mark.asyncio
async def test_convergence_router_continue():
    """convergence_router returns 'continue' when not converged."""
    from cryptotrader.nodes.debate import convergence_router

    state = _base_state()
    state["debate_round"] = 1
    state["max_debate_rounds"] = 3
    state["divergence_scores"] = [0.5, 0.4]
    assert convergence_router(state) == "continue"


# ── nodes/verdict.py ──


@pytest.mark.asyncio
async def test_make_verdict_weighted():
    """make_verdict uses weighted fallback when llm_verdict=False."""
    from cryptotrader.nodes.verdict import make_verdict

    analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.9, "reasoning": "strong"},
        "chain_agent": {"direction": "bullish", "confidence": 0.8, "reasoning": "positive"},
        "news_agent": {"direction": "bullish", "confidence": 0.7, "reasoning": "good news"},
        "macro_agent": {"direction": "neutral", "confidence": 0.5, "reasoning": "mixed"},
    }
    state = _base_state(analyses=analyses)
    state["metadata"]["llm_verdict"] = False
    state["divergence_scores"] = [0.1]

    result = await make_verdict(state)
    verdict = result["data"]["verdict"]
    assert verdict["action"] in ("long", "short", "hold")
    assert 0 <= verdict["confidence"] <= 1


@pytest.mark.asyncio
async def test_risk_router_approved():
    """risk_router returns 'approved' when passed=True."""
    from cryptotrader.nodes.verdict import risk_router

    state = _base_state(risk_gate={"passed": True, "rejected_by": None, "reason": None})
    assert risk_router(state) == "approved"


@pytest.mark.asyncio
async def test_risk_router_rejected():
    """risk_router returns 'rejected' when passed=False."""
    from cryptotrader.nodes.verdict import risk_router

    state = _base_state(risk_gate={"passed": False, "rejected_by": "daily_loss", "reason": "over limit"})
    assert risk_router(state) == "rejected"


# ── nodes/execution.py ──


@pytest.mark.asyncio
async def test_place_order_hold_returns_none():
    """place_order returns None for hold verdict."""
    from cryptotrader.nodes.execution import place_order

    state = _base_state(verdict={"action": "hold", "confidence": 0.5, "position_scale": 1.0})
    result = await place_order(state)
    assert result["data"]["order"] is None


@pytest.mark.asyncio
async def test_place_order_long_paper():
    """place_order executes long via PaperExchange."""
    from cryptotrader.nodes.execution import _paper_exchanges, place_order

    state = _base_state(
        verdict={"action": "long", "confidence": 0.8, "position_scale": 1.0},
        portfolio={"total_value": 10000},
    )

    mock_exchange = MagicMock()
    mock_exchange.get_balance = AsyncMock(return_value={"USDT": 10000.0})
    mock_exchange.get_positions = AsyncMock(return_value={})
    mock_exchange.place_order = AsyncMock(return_value={"status": "filled", "filled": 0.02, "price": 50000})

    # Inject mock into per-pair cache
    _paper_exchanges["BTC/USDT"] = mock_exchange

    try:
        result = await place_order(state)
        order = result["data"]["order"]
        assert order is not None
        assert order["side"] == "buy"
        assert order["status"] == "filled"
    finally:
        _paper_exchanges.pop("BTC/USDT", None)


@pytest.mark.asyncio
async def test_place_order_zero_price_returns_none():
    """place_order returns None when price is 0."""
    from cryptotrader.nodes.execution import place_order

    state = _base_state(
        verdict={"action": "long", "confidence": 0.8, "position_scale": 1.0},
    )
    state["data"]["snapshot_summary"]["price"] = 0

    result = await place_order(state)
    assert result["data"]["order"] is None


# ── nodes/journal.py ──


@pytest.mark.asyncio
async def test_journal_trade_produces_hash():
    """journal_trade commits and returns a hash, including fill_price/slippage and full order fields."""
    from cryptotrader.nodes.journal import journal_trade

    state = _base_state(
        analyses={
            "tech_agent": {"direction": "bullish", "confidence": 0.8, "reasoning": "up"},
        },
        verdict={
            "action": "long",
            "confidence": 0.8,
            "position_scale": 1.0,
            "divergence": 0.1,
            "reasoning": "bullish",
            "thesis": "breakout",
            "invalidation": "below 48k",
        },
        risk_gate={"passed": True, "rejected_by": None, "reason": None},
        order={
            "pair": "BTC/USDT",
            "side": "buy",
            "amount": 0.02,
            "price": 50000,
            "status": "filled",
            "order_type": "market",
            "exchange_id": "paper-001",
        },
        fill_price=50050.0,
        slippage=0.001,
    )

    # Capture the commit object passed to store.commit()
    captured = {}

    async def _capture_commit(dc):
        captured["dc"] = dc

    with patch("cryptotrader.journal.store.JournalStore.commit", side_effect=_capture_commit):
        result = await journal_trade(state)

    assert "journal_hash" in result["data"]
    assert len(result["data"]["journal_hash"]) >= 16

    # Verify fill_price and slippage were recorded in the commit
    dc = captured["dc"]
    assert dc.fill_price == 50050.0
    assert dc.slippage == 0.001
    # Verify order fields were preserved (not just pair/side/amount/price)
    assert dc.order is not None
    assert str(dc.order.status) == "filled"
    assert dc.order.order_type == "market"


@pytest.mark.asyncio
async def test_journal_rejection_produces_hash():
    """journal_rejection commits and returns a hash."""
    from cryptotrader.nodes.journal import journal_rejection

    state = _base_state(
        analyses={
            "tech_agent": {"direction": "bullish", "confidence": 0.8, "reasoning": "up"},
        },
        verdict={
            "action": "long",
            "confidence": 0.8,
            "position_scale": 1.0,
            "divergence": 0.1,
            "reasoning": "bullish",
            "thesis": "breakout",
            "invalidation": "below 48k",
        },
        risk_gate={"passed": False, "rejected_by": "daily_loss", "reason": "over limit"},
    )

    mock_notifier = MagicMock()
    mock_notifier.notify = AsyncMock()
    with patch("cryptotrader.nodes.verdict._get_notifier", return_value=mock_notifier):
        result = await journal_rejection(state)

    assert "journal_hash" in result["data"]


# ── data/snapshot.py ──


@pytest.mark.asyncio
async def test_snapshot_aggregator():
    """SnapshotAggregator collects data from all sources."""
    from cryptotrader.data.snapshot import SnapshotAggregator

    mock_market = MarketData(
        pair="BTC/USDT",
        ohlcv=None,
        ticker={"last": 50000},
        funding_rate=0.01,
        orderbook_imbalance=0.5,
        volatility=0.02,
    )
    mock_onchain = OnchainData()
    mock_news = NewsSentiment()
    mock_macro = MacroData()

    with (
        patch.object(SnapshotAggregator, "__init__", lambda self, cfg=None: None),
    ):
        agg = SnapshotAggregator()
        agg.market = MagicMock()
        agg.market.collect = AsyncMock(return_value=mock_market)
        agg.onchain = MagicMock()
        agg.onchain.collect = AsyncMock(return_value=mock_onchain)
        agg.news = MagicMock()
        agg.news.collect = AsyncMock(return_value=mock_news)
        agg.macro = MagicMock()
        agg.macro.collect = AsyncMock(return_value=mock_macro)

        snapshot = await agg.collect("BTC/USDT")

    assert snapshot.pair == "BTC/USDT"
    assert snapshot.market.ticker["last"] == 50000
    agg.market.collect.assert_called_once()
    agg.onchain.collect.assert_called_once()


# ── debate/challenge.py ──


def test_build_challenge_prompt():
    """build_challenge_prompt produces valid prompt string."""
    from cryptotrader.debate.challenge import build_challenge_prompt

    prompt = build_challenge_prompt(
        agent_role="technical",
        pair="ETH/USDT",
        own_analysis={"direction": "bullish", "confidence": 0.8},
        other_analyses={
            "chain_agent": {"direction": "bearish", "confidence": 0.7},
            "macro_agent": {"direction": "neutral", "confidence": 0.5},
        },
    )

    assert "ETH/USDT" in prompt
    assert "technical" in prompt
    assert "CHAIN_AGENT" in prompt
    assert "ANTI-CONVERGENCE" in prompt
    assert "new_findings" in prompt


# ── nodes/verdict.py — deeper coverage ──


@pytest.mark.asyncio
async def test_gather_risk_constraints():
    """_gather_risk_constraints collects portfolio and market constraints."""
    from cryptotrader.nodes.verdict import _gather_risk_constraints

    state = _base_state()

    mock_pm = MagicMock()
    mock_pm.get_portfolio = AsyncMock(
        return_value={"total_value": 100000, "positions": {"BTC/USDT": {"amount": 0.5, "avg_price": 50000}}}
    )
    mock_pm.get_daily_pnl = AsyncMock(return_value=-500.0)
    mock_pm.get_drawdown = AsyncMock(return_value=0.03)

    mock_rsm = MagicMock()
    mock_rsm.is_circuit_breaker_active = AsyncMock(return_value=False)
    mock_rsm.get = AsyncMock(return_value=None)

    with (
        patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm),
    ):
        constraints = await _gather_risk_constraints(state)

    assert "max_position_pct" in constraints
    assert "max_drawdown_pct" in constraints
    assert constraints.get("funding_rate") == 0.01
    assert constraints.get("volatility") == 0.02
    assert constraints.get("drawdown_current") == 0.03
    assert constraints.get("circuit_breaker_active") is False


@pytest.mark.asyncio
async def test_gather_risk_constraints_no_portfolio():
    """_gather_risk_constraints handles missing portfolio gracefully."""
    from cryptotrader.nodes.verdict import _gather_risk_constraints

    state = _base_state()

    mock_pm = MagicMock()
    mock_pm.get_portfolio = AsyncMock(side_effect=Exception("DB unavailable"))

    mock_rsm = MagicMock()
    mock_rsm.is_circuit_breaker_active = AsyncMock(side_effect=Exception("Redis down"))

    with (
        patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm),
    ):
        constraints = await _gather_risk_constraints(state)

    # Should not raise, returns at least config defaults
    assert "max_position_pct" in constraints
    assert "remaining_exposure_pct" not in constraints  # Portfolio failed


@pytest.mark.asyncio
async def test_risk_check_node():
    """risk_check node runs gate checks and returns result dict."""
    from cryptotrader.nodes.verdict import _risk_gate_cache, risk_check

    state = _base_state(
        verdict={
            "action": "long",
            "confidence": 0.7,
            "position_scale": 0.5,
            "divergence": 0.1,
            "reasoning": "bullish",
            "thesis": "breakout",
            "invalidation": "below 48k",
        },
    )

    # Clear cache to ensure fresh gate
    _risk_gate_cache.clear()

    mock_pm = MagicMock()
    mock_pm.get_portfolio = AsyncMock(return_value={"total_value": 10000, "positions": {}})
    mock_pm.get_daily_pnl = AsyncMock(return_value=0.0)
    mock_pm.get_drawdown = AsyncMock(return_value=0.0)
    mock_pm.get_returns = AsyncMock(return_value=[0.01] * 30)

    with patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm):
        result = await risk_check(state)

    rg = result["data"]["risk_gate"]
    assert "passed" in rg
    assert "rejected_by" in rg
    assert "reason" in rg


@pytest.mark.asyncio
async def test_make_verdict_llm():
    """make_verdict with llm_verdict=True calls LLM and returns verdict."""
    from cryptotrader.nodes.verdict import make_verdict

    analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.85, "reasoning": "breakout"},
        "chain_agent": {"direction": "bullish", "confidence": 0.7, "reasoning": "positive funding"},
    }
    state = _base_state(analyses=analyses)
    state["metadata"]["llm_verdict"] = True

    mock_verdict = MagicMock()
    mock_verdict.action = "long"
    mock_verdict.confidence = 0.8
    mock_verdict.position_scale = 0.7
    mock_verdict.divergence = 0.1
    mock_verdict.reasoning = "Strong bullish consensus"
    mock_verdict.thesis = "Breakout confirmed"
    mock_verdict.invalidation = "Below 48k"

    with (
        patch(
            "cryptotrader.debate.verdict.make_verdict_llm",
            new_callable=AsyncMock,
            return_value=mock_verdict,
        ),
        patch(
            "cryptotrader.nodes.verdict._gather_risk_constraints",
            new_callable=AsyncMock,
            return_value={"max_position_pct": 0.1},
        ),
    ):
        result = await make_verdict(state)

    v = result["data"]["verdict"]
    assert v["action"] == "long"
    assert v["confidence"] == 0.8
    assert v["thesis"] == "Breakout confirmed"


# ── nodes/execution.py — deeper coverage ──


@pytest.mark.asyncio
async def test_place_order_short():
    """place_order creates sell order for short verdict."""
    from cryptotrader.nodes.execution import _paper_exchanges, place_order

    state = _base_state(
        verdict={"action": "short", "confidence": 0.7, "position_scale": 0.5},
        portfolio={"total_value": 10000},
    )

    mock_exchange = MagicMock()
    mock_exchange.get_balance = AsyncMock(return_value={"USDT": 10000.0})
    mock_exchange.get_positions = AsyncMock(return_value={})
    mock_exchange.place_order = AsyncMock(return_value={"status": "filled", "filled": 0.01, "price": 50000})

    _paper_exchanges["BTC/USDT"] = mock_exchange
    try:
        result = await place_order(state)
        order = result["data"]["order"]
        assert order["side"] == "sell"
    finally:
        _paper_exchanges.pop("BTC/USDT", None)


@pytest.mark.asyncio
async def test_place_order_partial_fill():
    """place_order handles partially filled orders."""
    from cryptotrader.nodes.execution import _paper_exchanges, place_order

    state = _base_state(
        verdict={"action": "long", "confidence": 0.8, "position_scale": 1.0},
        portfolio={"total_value": 10000},
    )

    mock_exchange = MagicMock()
    mock_exchange.get_balance = AsyncMock(return_value={"USDT": 10000.0})
    mock_exchange.get_positions = AsyncMock(return_value={})
    mock_exchange.place_order = AsyncMock(return_value={"status": "partially_filled", "filled": 0.005, "price": 50100})

    _paper_exchanges["BTC/USDT"] = mock_exchange
    try:
        result = await place_order(state)
        order = result["data"]["order"]
        assert order is not None
        assert order["amount"] == 0.005
        assert order["status"] == "partially_filled"
    finally:
        _paper_exchanges.pop("BTC/USDT", None)


@pytest.mark.asyncio
async def test_place_order_rejected():
    """place_order returns None for rejected order."""
    from cryptotrader.nodes.execution import _paper_exchanges, place_order

    state = _base_state(
        verdict={"action": "long", "confidence": 0.8, "position_scale": 1.0},
        portfolio={"total_value": 10000},
    )

    mock_exchange = MagicMock()
    mock_exchange.get_balance = AsyncMock(return_value={"USDT": 10000.0})
    mock_exchange.get_positions = AsyncMock(return_value={})
    mock_exchange.place_order = AsyncMock(return_value={"status": "rejected"})

    _paper_exchanges["BTC/USDT"] = mock_exchange
    try:
        result = await place_order(state)
        assert result["data"]["order"] is None
    finally:
        _paper_exchanges.pop("BTC/USDT", None)


# ── nodes/data.py — live collection path ──


@pytest.mark.asyncio
async def test_collect_snapshot_live_path():
    """collect_snapshot calls SnapshotAggregator when no pre-provided snapshot."""
    from cryptotrader.nodes.data import collect_snapshot

    state = {
        "messages": [],
        "data": {},
        "metadata": {"pair": "ETH/USDT", "exchange_id": "binance", "timeframe": "1h", "ohlcv_limit": 50},
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }

    mock_snapshot = _make_snapshot("ETH/USDT", 3000)

    with (
        patch("cryptotrader.data.snapshot.SnapshotAggregator") as mock_agg,
        patch("cryptotrader.config.load_config") as mock_cfg,
    ):
        mock_cfg.return_value.providers = MagicMock()
        mock_agg.return_value.collect = AsyncMock(return_value=mock_snapshot)

        result = await collect_snapshot(state)

    assert result["data"]["snapshot_summary"]["pair"] == "ETH/USDT"
    assert result["data"]["snapshot_summary"]["price"] == 3000
    assert "snapshot" in result["data"]
