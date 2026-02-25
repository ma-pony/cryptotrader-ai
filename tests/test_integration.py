"""End-to-end integration tests for the full trading pipeline.

Tests the complete flow: data → agents → debate → verdict → risk gate → execution → journal.
All LLM calls and external APIs are mocked.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from cryptotrader.models import (
    DataSnapshot,
    MacroData,
    MarketData,
    NewsSentiment,
    OnchainData,
    TradeVerdict,
)


# ── Fixtures ──

@pytest.fixture
def sample_ohlcv():
    """100 rows of synthetic OHLCV data."""
    rows = []
    price = 50000.0
    for i in range(100):
        o = price + i * 10
        h = o + 50
        low = o - 50
        c = o + 20
        v = 1000.0
        rows.append([o, h, low, c, v])
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def sample_snapshot(sample_ohlcv):
    return DataSnapshot(
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        market=MarketData(
            pair="BTC/USDT",
            ohlcv=sample_ohlcv,
            ticker={"last": 51000.0, "baseVolume": 5000.0},
            funding_rate=0.0001,
            orderbook_imbalance=0.05,
            volatility=0.02,
        ),
        onchain=OnchainData(
            exchange_netflow=-500.0,
            open_interest=15_000_000_000,
            liquidations_24h={"long_liquidations": 50_000_000, "short_liquidations": 30_000_000},
            defi_tvl=45_000_000_000,
            defi_tvl_change_7d=0.03,
        ),
        news=NewsSentiment(
            headlines=["Bitcoin surges past 50k", "ETF approval expected"],
            sentiment_score=0.6,
            key_events=["ETF approval expected"],
        ),
        macro=MacroData(
            fed_rate=5.25, dxy=104.5, btc_dominance=56.0, fear_greed_index=72,
        ),
    )


def _mock_llm_response(direction: str, confidence: float, reasoning: str = "test"):
    """Create a mock litellm response."""
    content = json.dumps({
        "direction": direction,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_factors": ["factor1"],
        "risk_flags": [],
    })
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Agent Tests ──

class TestAgentParsing:
    """Test agent prompt building and response parsing."""

    @pytest.mark.asyncio
    async def test_base_agent_parse_valid_json(self, sample_snapshot):
        from cryptotrader.agents.tech import TechAgent

        mock_resp = _mock_llm_response("bullish", 0.8, "Strong uptrend")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            agent = TechAgent(model="test-model")
            result = await agent.analyze(sample_snapshot)

        assert result.direction == "bullish"
        assert result.confidence == 0.8
        assert result.agent_id == "tech"

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_range(self, sample_snapshot):
        """LLM returns confidence > 1.0, should be clamped."""
        from cryptotrader.agents.tech import TechAgent

        mock_resp = _mock_llm_response("bullish", 1.5)
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = await TechAgent(model="test").analyze(sample_snapshot)

        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_confidence_clamped_negative(self, sample_snapshot):
        from cryptotrader.agents.tech import TechAgent

        mock_resp = _mock_llm_response("bearish", -0.3)
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = await TechAgent(model="test").analyze(sample_snapshot)

        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_agent_llm_failure_returns_neutral(self, sample_snapshot):
        from cryptotrader.agents.tech import TechAgent

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("API down")):
            result = await TechAgent(model="test").analyze(sample_snapshot)

        assert result.direction == "neutral"
        assert result.confidence == 0.5


# ── Verdict Tests ──

class TestVerdict:

    def test_rules_verdict_clear_bullish(self):
        from cryptotrader.debate.verdict import make_verdict_rules

        analyses = {
            "tech": {"direction": "bullish", "confidence": 0.8},
            "chain": {"direction": "bullish", "confidence": 0.7},
            "news": {"direction": "bullish", "confidence": 0.6},
            "macro": {"direction": "neutral", "confidence": 0.5},
        }
        v = make_verdict_rules(analyses)
        assert v is not None
        assert v.action == "long"

    def test_rules_verdict_ambiguous_returns_none(self):
        """When score is near zero, rules should return None for LLM tiebreak."""
        from cryptotrader.debate.verdict import make_verdict_rules

        analyses = {
            "tech": {"direction": "bullish", "confidence": 0.5},
            "chain": {"direction": "bearish", "confidence": 0.5},
            "news": {"direction": "neutral", "confidence": 0.5},
            "macro": {"direction": "neutral", "confidence": 0.5},
        }
        v = make_verdict_rules(analyses)
        assert v is None  # ambiguous, should defer to LLM


class TestChainAgentPrompt:

    def test_chain_prompt_includes_onchain_data(self, sample_snapshot):
        """ChainAgent prompt should include exchange netflow, whale transfers, DeFi TVL."""
        from cryptotrader.agents.chain import ChainAgent

        agent = ChainAgent(model="test")
        prompt = agent._build_prompt(sample_snapshot, "")

        assert "Exchange netflow" in prompt
        assert "outflow" in prompt  # negative netflow = accumulation
        assert "DeFi TVL" in prompt


class TestMalformedLLMOutput:

    @pytest.mark.asyncio
    async def test_agent_parse_malformed_json(self, sample_snapshot):
        """Agent should gracefully handle non-JSON LLM output."""
        from cryptotrader.agents.tech import TechAgent

        choice = MagicMock()
        choice.message.content = "I think the market is bullish but I can't format JSON"
        resp = MagicMock()
        resp.choices = [choice]

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
            result = await TechAgent(model="test").analyze(sample_snapshot)

        assert result.direction == "neutral"
        assert result.confidence == 0.5


# ── Risk Gate Integration Tests ──

class _FakeRedis:
    """In-memory fake Redis for testing risk checks without a real Redis server."""

    def __init__(self):
        self._store: dict[str, str] = {}

    @property
    def available(self):
        return True

    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        self._store[key] = value

    async def incr(self, key: str):
        val = int(self._store.get(key, "0")) + 1
        self._store[key] = str(val)
        return val

    async def expire(self, key: str, seconds: int):
        pass

    async def set_cooldown(self, pair: str, minutes: int):
        await self.set(f"cooldown:{pair}", "1", ex=minutes * 60)

    async def set_post_loss_cooldown(self, minutes: int):
        await self.set("cooldown:post_loss", "1", ex=minutes * 60)

    async def incr_trade_count(self):
        pass

    async def get_trade_counts(self):
        return 0, 0

    async def set_circuit_breaker(self):
        self._store["circuit_breaker:active"] = "1"

    async def is_circuit_breaker_active(self):
        return "circuit_breaker:active" in self._store

    async def reset_circuit_breaker(self):
        self._store.pop("circuit_breaker:active", None)


class TestRiskGateIntegration:
    """Test the full risk gate with all 11 checks."""

    @pytest.fixture
    def healthy_portfolio(self):
        return {
            "total_value": 10000,
            "positions": {},
            "daily_pnl": 0.0,
            "drawdown": 0.02,
            "returns_60d": [0.01, -0.01, 0.02, -0.005, 0.01] * 12,
            "recent_prices": [100, 101, 100.5, 101.2, 100.8],
            "funding_rate": 0.0001,
            "api_latency_ms": 200,
            "pair": "BTC/USDT",
        }

    @pytest.mark.asyncio
    async def test_all_checks_pass(self, healthy_portfolio):
        from cryptotrader.risk.gate import RiskGate
        from cryptotrader.config import RiskConfig

        gate = RiskGate(RiskConfig(), _FakeRedis())
        verdict = TradeVerdict(action="long", confidence=0.7, position_scale=0.5)
        result = await gate.check(verdict, healthy_portfolio)
        assert result.passed

    @pytest.mark.asyncio
    async def test_circuit_breaker_persists_across_calls(self):
        from cryptotrader.risk.gate import RiskGate
        from cryptotrader.config import RiskConfig

        fake_redis = _FakeRedis()
        gate = RiskGate(RiskConfig(), fake_redis)
        verdict = TradeVerdict(action="long", confidence=0.7, position_scale=0.5)

        bad = {
            "total_value": 10000, "positions": {}, "daily_pnl": -500,
            "drawdown": 0.0, "returns_60d": [], "recent_prices": [100],
            "funding_rate": 0.0, "api_latency_ms": 100, "pair": "BTC/USDT",
        }
        r1 = await gate.check(verdict, bad)
        assert not r1.passed

        ok = {
            "total_value": 10000, "positions": {}, "daily_pnl": 0,
            "drawdown": 0.0, "returns_60d": [], "recent_prices": [100],
            "funding_rate": 0.0, "api_latency_ms": 100, "pair": "ETH/USDT",
        }
        r2 = await gate.check(verdict, ok)
        assert not r2.passed
        assert r2.rejected_by == "daily_loss_limit"


# ── Execution Tests ──

class TestPaperExchange:

    @pytest.mark.asyncio
    async def test_buy_order_fills(self):
        from cryptotrader.execution.simulator import PaperExchange
        from cryptotrader.models import Order

        ex = PaperExchange()
        order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
        result = await ex.place_order(order)

        assert result["status"] == "filled"
        assert result["amount"] == 0.1
        bal = await ex.get_balance()
        assert "BTC" in bal
        assert bal["BTC"] == 0.1

    @pytest.mark.asyncio
    async def test_insufficient_balance_rejected(self):
        from cryptotrader.execution.simulator import PaperExchange
        from cryptotrader.models import Order

        ex = PaperExchange()  # starts with 10000 USDT
        order = Order(pair="BTC/USDT", side="buy", amount=1.0, price=50000)  # costs 50k
        result = await ex.place_order(order)

        assert result["status"] == "failed"
        assert "Insufficient" in result["reason"]

    @pytest.mark.asyncio
    async def test_sell_without_holdings_rejected(self):
        from cryptotrader.execution.simulator import PaperExchange
        from cryptotrader.models import Order

        ex = PaperExchange()
        order = Order(pair="BTC/USDT", side="sell", amount=0.1, price=50000)
        result = await ex.place_order(order)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_buy_then_sell_roundtrip(self):
        from cryptotrader.execution.simulator import PaperExchange
        from cryptotrader.models import Order

        ex = PaperExchange()
        buy = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
        await ex.place_order(buy)

        sell = Order(pair="BTC/USDT", side="sell", amount=0.1, price=51000)
        result = await ex.place_order(sell)

        assert result["status"] == "filled"
        bal = await ex.get_balance()
        assert bal.get("BTC", 0) == 0


# ── Backtest Engine Tests ──

class TestBacktestEngine:

    def test_apply_costs_buy(self):
        from cryptotrader.backtest.engine import BacktestEngine

        engine = BacktestEngine("BTC/USDT", "2024-01-01", "2024-02-01", slippage_bps=5, fee_bps=10)
        fill = engine._apply_costs(50000, "buy")
        # buy: price + slip + fee = 50000 + 25 + 50 = 50075
        assert fill == pytest.approx(50075.0)

    def test_apply_costs_sell(self):
        from cryptotrader.backtest.engine import BacktestEngine

        engine = BacktestEngine("BTC/USDT", "2024-01-01", "2024-02-01", slippage_bps=5, fee_bps=10)
        fill = engine._apply_costs(50000, "sell")
        # sell: price - slip - fee = 50000 - 25 - 50 = 49925
        assert fill == pytest.approx(49925.0)

    def test_simple_signal_sma_crossover(self):
        from cryptotrader.backtest.engine import BacktestEngine

        engine = BacktestEngine("BTC/USDT", "2024-01-01", "2024-02-01")
        # Build uptrending window: 60 candles with rising prices
        window = [[i * 3600000, 100 + i, 105 + i, 95 + i, 100 + i, 1000] for i in range(60)]
        signal = engine._simple_signal(window)
        assert signal in ("long", "short", "hold")

    def test_simple_signal_too_few_candles(self):
        from cryptotrader.backtest.engine import BacktestEngine

        engine = BacktestEngine("BTC/USDT", "2024-01-01", "2024-02-01")
        window = [[i * 3600000, 100, 105, 95, 100, 1000] for i in range(10)]
        assert engine._simple_signal(window) == "hold"

    def test_compute_result_metrics(self):
        from cryptotrader.backtest.engine import BacktestEngine

        engine = BacktestEngine("BTC/USDT", "2024-01-01", "2024-02-01", initial_capital=10000)
        engine.capital = 10000
        curve = [10000, 10100, 10050, 10200, 10150, 10300]
        trades = [
            {"side": "buy", "price": 50000, "amount": 0.02, "ts": 1},
            {"side": "close_long", "price": 51000, "pnl": 20, "ts": 2},
            {"side": "sell", "price": 51000, "amount": 0.02, "ts": 3},
            {"side": "close_short", "price": 50500, "pnl": 10, "ts": 4},
        ]
        result = engine._compute_result(10300, curve, trades)
        assert result.total_return == pytest.approx(0.03)
        assert result.win_rate == 1.0  # both pnl trades are positive
        assert result.max_drawdown <= 0  # drawdown is negative or zero
        assert result.sharpe_ratio != 0  # should compute something


# ── Config Tests ──

class TestConfigLoading:

    def test_default_config_loads(self):
        from cryptotrader.config import AppConfig

        config = AppConfig()
        assert config.engine == "paper"
        assert config.risk.position.max_single_pct == 0.10
        assert config.risk.loss.max_daily_loss_pct == 0.03
        assert config.scheduler.enabled is False

    def test_risk_config_nested(self):
        from cryptotrader.config import RiskConfig

        rc = RiskConfig()
        assert rc.cooldown.same_pair_minutes == 60
        assert rc.cooldown.post_loss_minutes == 120
        assert rc.volatility.flash_crash_threshold == 0.05
        assert rc.rate_limit.max_trades_per_hour == 6

    def test_scheduler_config_defaults(self):
        from cryptotrader.config import SchedulerConfig

        sc = SchedulerConfig()
        assert sc.pairs == ["BTC/USDT", "ETH/USDT"]
        assert sc.interval_minutes == 240
        assert sc.exchange_id == "binance"


# ── Journal Tests ──

class TestJournalInMemory:

    @pytest.mark.asyncio
    async def test_commit_and_retrieve(self):
        from cryptotrader.journal.store import JournalStore
        from cryptotrader.journal.commit import build_commit

        store = JournalStore(None)  # in-memory
        commit = build_commit(
            pair="BTC/USDT",
            snapshot_summary={"price": 50000, "volatility": 0.02},
            analyses={"tech": {"agent_id": "tech", "pair": "BTC/USDT", "direction": "bullish", "confidence": 0.8, "reasoning": "test"}},
            debate_rounds=1,
            divergence=0.15,
            verdict=None,
            risk_gate=None,
            order=None,
            parent_hash=None,
        )
        await store.commit(commit)
        logs = await store.log(limit=10)
        assert len(logs) == 1
        assert logs[0].pair == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_show_by_hash(self):
        from cryptotrader.journal.store import JournalStore
        from cryptotrader.journal.commit import build_commit

        store = JournalStore(None)
        commit = build_commit(
            pair="ETH/USDT",
            snapshot_summary={"price": 3000},
            analyses={},
            debate_rounds=0,
            divergence=0.0,
            verdict=None,
            risk_gate=None,
            order=None,
            parent_hash=None,
        )
        await store.commit(commit)
        found = await store.show(commit.hash)
        assert found is not None
        assert found.pair == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_update_pnl(self):
        from cryptotrader.journal.store import JournalStore
        from cryptotrader.journal.commit import build_commit

        store = JournalStore(None)
        commit = build_commit(
            pair="BTC/USDT",
            snapshot_summary={"price": 50000},
            analyses={},
            debate_rounds=0,
            divergence=0.0,
            verdict=None,
            risk_gate=None,
            order=None,
            parent_hash=None,
        )
        await store.commit(commit)
        await store.update_pnl(commit.hash, 150.0, "Good entry timing")
        found = await store.show(commit.hash)
        assert found.pnl == 150.0
        assert found.retrospective == "Good entry timing"


# ── Verbal Learning Tests ──

class TestVerbalLearning:

    @pytest.mark.asyncio
    async def test_no_history_returns_empty(self):
        from cryptotrader.learning.verbal import get_experience
        from cryptotrader.journal.store import JournalStore

        store = JournalStore(None)
        result = await get_experience(store, {"funding_rate": 0.0001, "volatility": 0.02})
        assert result == ""

    @pytest.mark.asyncio
    async def test_with_history_returns_text(self):
        from cryptotrader.learning.verbal import get_experience
        from cryptotrader.journal.store import JournalStore
        from cryptotrader.journal.commit import build_commit
        from cryptotrader.models import TradeVerdict

        store = JournalStore(None)
        commit = build_commit(
            pair="BTC/USDT",
            snapshot_summary={"funding_rate": 0.0001, "volatility": 0.02},
            analyses={},
            debate_rounds=0,
            divergence=0.0,
            verdict=TradeVerdict(action="long", confidence=0.7, position_scale=0.5),
            risk_gate=None,
            order=None,
            parent_hash=None,
        )
        await store.commit(commit)
        await store.update_pnl(commit.hash, 200.0, "Caught the trend early")

        result = await get_experience(store, {"funding_rate": 0.0001, "volatility": 0.02})
        assert "BTC/USDT" in result
        assert "Caught the trend early" in result


# ── Full Pipeline Integration Test ──

class TestFullPipeline:
    """End-to-end: snapshot → agents → verdict → risk gate → execution.

    All LLM calls mocked, no external dependencies.
    """

    @pytest.fixture
    def bullish_llm_response(self):
        return _mock_llm_response("bullish", 0.75, "Strong momentum")

    @pytest.fixture
    def verdict_llm_response(self):
        content = json.dumps({
            "action": "long",
            "confidence": 0.7,
            "position_scale": 0.5,
            "reasoning": "Consensus bullish across agents",
        })
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    @pytest.mark.asyncio
    async def test_full_pipeline_long_signal(
        self, sample_snapshot, bullish_llm_response, verdict_llm_response
    ):
        """Full graph: 4 agents all bullish → verdict long → risk passes → order placed."""
        from cryptotrader.graph import build_lite_graph, _risk_gate_cache

        # Clear cached risk gate to avoid cross-test contamination
        _risk_gate_cache.clear()

        # Mock all LLM calls: agents get bullish, verdict gets long
        call_count = {"n": 0}
        async def mock_acompletion(*args, **kwargs):
            call_count["n"] += 1
            # First 4 calls are agents, 5th would be verdict
            if call_count["n"] <= 4:
                return bullish_llm_response
            return verdict_llm_response

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=mock_acompletion):
            graph = build_lite_graph()
            initial = {
                "messages": [],
                "data": {"snapshot": sample_snapshot},
                "metadata": {
                    "pair": "BTC/USDT",
                    "engine": "paper",
                    "analysis_model": "test-model",
                    "llm_verdict": True,
                    "verdict_model": "test-model",
                },
                "debate_round": 0,
                "max_debate_rounds": 2,
                "divergence_scores": [],
            }
            result = await graph.ainvoke(initial)

        # All 4 agents should have produced analyses
        analyses = result["data"].get("analyses", {})
        assert len(analyses) == 4
        for agent_id in ("tech_agent", "chain_agent", "news_agent", "macro_agent"):
            assert agent_id in analyses
            assert analyses[agent_id]["direction"] == "bullish"

        # Verdict should exist
        verdict = result["data"].get("verdict", {})
        assert verdict["action"] == "long"
        assert verdict["confidence"] > 0
