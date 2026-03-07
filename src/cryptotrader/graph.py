"""LangGraph main orchestration — build_trading_graph() per ARCHITECTURE.md section 3.2."""

from __future__ import annotations

import logging
import operator
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


def merge_dicts(a: dict, b: dict) -> dict:
    result = {**a}
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


class ArenaState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, Any], merge_dicts]
    metadata: Annotated[dict[str, Any], merge_dicts]
    debate_round: int
    max_debate_rounds: int
    divergence_scores: list[float]


# ── Node functions ──


async def collect_snapshot(state: ArenaState) -> dict:
    # Skip collection if snapshot already provided (e.g. backtest)
    if state.get("data", {}).get("snapshot"):
        snapshot = state["data"]["snapshot"]
        summary = {
            "pair": snapshot.pair,
            "price": snapshot.market.ticker.get("last", 0),
            "funding_rate": snapshot.market.funding_rate,
            "volatility": snapshot.market.volatility,
            "orderbook_imbalance": snapshot.market.orderbook_imbalance,
        }
        return {"data": {"snapshot_summary": summary}}

    from cryptotrader.config import load_config
    from cryptotrader.data.snapshot import SnapshotAggregator

    pair = state["metadata"]["pair"]
    exchange_id = state["metadata"].get("exchange_id", "binance")
    timeframe = state["metadata"].get("timeframe", "1h")
    limit = state["metadata"].get("ohlcv_limit", 100)

    providers_cfg = load_config().providers
    agg = SnapshotAggregator(providers_cfg)
    snapshot = await agg.collect(pair, exchange_id, timeframe, limit)
    summary = {
        "pair": pair,
        "price": snapshot.market.ticker.get("last", 0),
        "funding_rate": snapshot.market.funding_rate,
        "volatility": snapshot.market.volatility,
        "orderbook_imbalance": snapshot.market.orderbook_imbalance,
    }
    return {"data": {"snapshot": snapshot, "snapshot_summary": summary}}


async def verbal_reinforcement(state: ArenaState) -> dict:
    from cryptotrader.journal.calibrate import detect_biases, generate_bias_correction, generate_verdict_calibration
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.learning.verbal import get_experience

    db_url = state["metadata"].get("database_url")
    store = JournalStore(db_url)
    summary = state["data"].get("snapshot_summary", {})
    experience = await get_experience(store, summary)

    # Phase 4D: Detect biases and generate corrections
    bias_correction = ""
    verdict_calibration = ""
    try:
        biases = await detect_biases(store, days=30)
        bias_correction = generate_bias_correction(biases)
        verdict_calibration = generate_verdict_calibration(biases)
    except Exception:
        logger.debug("Bias detection failed, continuing without calibration")

    if bias_correction:
        experience = f"{experience}\n\n{bias_correction}" if experience else bias_correction

    return {"data": {"experience": experience, "verdict_calibration": verdict_calibration}}


async def _run_agent(agent_type: str, state: ArenaState) -> dict:
    from cryptotrader.agents.chain import ChainAgent
    from cryptotrader.agents.macro import MacroAgent
    from cryptotrader.agents.news import NewsAgent
    from cryptotrader.agents.tech import TechAgent

    agents = {
        "tech_agent": lambda m: TechAgent(model=m),
        "chain_agent": lambda m: ChainAgent(model=m),
        "news_agent": lambda m: NewsAgent(model=m),
        "macro_agent": lambda m: MacroAgent(model=m),
    }
    # Per-agent model: metadata.models.tech_agent, fallback to analysis_model
    models_cfg = state["metadata"].get("models", {})
    model = models_cfg.get(agent_type, state["metadata"].get("analysis_model", "gpt-4o-mini"))
    agent = agents[agent_type](model)
    snapshot = state["data"]["snapshot"]
    experience = state["data"].get("experience", "")
    analysis = await agent.analyze(snapshot, experience)
    result = {
        "direction": analysis.direction,
        "confidence": analysis.confidence,
        "reasoning": analysis.reasoning,
        "key_factors": analysis.key_factors,
        "risk_flags": analysis.risk_flags,
    }
    # Pass through all extra fields from data_points (regime, strength, crowding, etc.)
    result.update(analysis.data_points)
    return {"data": {"analyses": {agent_type: result}}}


async def tech_analyze(state: ArenaState) -> dict:
    return await _run_agent("tech_agent", state)


async def chain_analyze(state: ArenaState) -> dict:
    return await _run_agent("chain_agent", state)


async def news_analyze(state: ArenaState) -> dict:
    return await _run_agent("news_agent", state)


async def macro_analyze(state: ArenaState) -> dict:
    return await _run_agent("macro_agent", state)


_DEBATE_ROLES = {
    "tech_agent": "technical analysis",
    "chain_agent": "on-chain and derivatives analysis",
    "news_agent": "news and sentiment analysis",
    "macro_agent": "macroeconomic analysis",
}

DEBATE_SYSTEM = """You are a {role} specialist in a multi-agent trading debate.

Rules:
- Base arguments ONLY on data. Cite specific numbers for every claim.
- HOLD your position when your data supports it — do NOT converge toward majority opinion.
- Only change your view if another agent presented a specific data point you hadn't considered.
- If you change, state exactly which data point changed your mind.
- Look for cross-domain contradictions (e.g., bullish technicals + bearish on-chain = important signal).

Output JSON: {{"direction": "bullish|bearish|neutral", "confidence": 0.0-1.0, "reasoning": "...",
"key_factors": [...], "risk_flags": [...], "new_findings": "cross-domain insight from other agents' data"}}"""


async def debate_round(state: ArenaState) -> dict:
    import json

    import litellm

    from cryptotrader.debate.challenge import build_challenge_prompt

    analyses = state["data"].get("analyses", {})
    model = state["metadata"].get("debate_model", "gpt-4o-mini")
    updated: dict[str, Any] = {}

    for agent_id, analysis in analyses.items():
        others = {k: v for k, v in analyses.items() if k != agent_id}
        prompt = build_challenge_prompt(agent_id, state["metadata"]["pair"], analysis, others)
        role_label = _DEBATE_ROLES.get(agent_id, agent_id)
        system = DEBATE_SYSTEM.format(role=role_label)
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
            )
            text = resp.choices[0].message.content
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"No JSON in debate response for {agent_id}")
            data = json.loads(text[start : end + 1])
            # Start from original analysis to preserve data_points fields,
            # then overlay only the fields the debate LLM updated
            merged = dict(analysis)
            merged.update(
                {
                    "direction": data.get("direction", analysis["direction"]),
                    "confidence": float(data.get("confidence", analysis["confidence"])),
                    "reasoning": data.get("reasoning", analysis["reasoning"]),
                    "key_factors": data.get("key_factors", analysis.get("key_factors", [])),
                    "risk_flags": data.get("risk_flags", analysis.get("risk_flags", [])),
                    "new_findings": data.get("new_findings", ""),
                }
            )
            updated[agent_id] = merged
        except Exception as e:
            logger.warning("Debate round LLM call failed for %s: %s", agent_id, e)
            updated[agent_id] = analysis

    return {
        "data": {"analyses": updated},
        "debate_round": state["debate_round"] + 1,
    }


async def check_stability(state: ArenaState) -> dict:
    from cryptotrader.debate.convergence import compute_divergence

    analyses = state["data"].get("analyses", {})
    divergence = compute_divergence(analyses)
    scores = list(state.get("divergence_scores") or [])
    scores.append(divergence)
    return {"divergence_scores": scores}


def convergence_router(state: ArenaState) -> str:
    from cryptotrader.debate.convergence import check_convergence

    scores = state.get("divergence_scores") or []
    if state["debate_round"] >= state["max_debate_rounds"]:
        return "converged"
    threshold = state["metadata"].get("convergence_threshold", 0.1)
    if len(scores) >= 2 and check_convergence(scores[:-1], scores[-1], threshold=threshold):
        return "converged"
    return "continue"


async def _gather_risk_constraints(state: ArenaState) -> dict:
    """Collect current risk constraints for injection into verdict prompt."""
    from cryptotrader.config import load_config
    from cryptotrader.portfolio.manager import PortfolioManager
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    constraints: dict[str, Any] = {
        "max_position_pct": config.risk.position.max_single_pct,
        "max_drawdown_pct": config.risk.loss.max_drawdown_pct,
    }

    # Portfolio state
    db_url = state["metadata"].get("database_url")
    pm = PortfolioManager(db_url)
    try:
        pm_data = await pm.get_portfolio()
        daily_pnl = await pm.get_daily_pnl()
        drawdown = await pm.get_drawdown()
        total = pm_data.get("total_value", 0)
        if total > 0:
            positions = pm_data.get("positions", {})
            current_exposure = sum(abs(p.get("amount", 0) * p.get("avg_price", 0)) for p in positions.values())
            max_exp = config.risk.position.max_total_exposure_pct
            constraints["remaining_exposure_pct"] = max(0.0, max_exp - current_exposure / total)
            daily_loss_budget = config.risk.loss.max_daily_loss_pct
            if daily_pnl < 0:
                constraints["daily_loss_remaining_pct"] = max(0.0, daily_loss_budget - abs(daily_pnl / total))
            else:
                constraints["daily_loss_remaining_pct"] = daily_loss_budget
            constraints["drawdown_current"] = drawdown
    except Exception:
        pass

    # Redis-based state (cooldowns, circuit breaker)
    redis_url = state["metadata"].get("redis_url")
    redis_state = RedisStateManager(redis_url)
    try:
        constraints["circuit_breaker_active"] = await redis_state.is_circuit_breaker_active()
        # Check cooldown for current pair
        pair = state["metadata"].get("pair", "")
        cooldown_val = await redis_state.get(f"cooldown:{pair}")
        if cooldown_val:
            constraints["cooldown_pairs"] = [pair]
    except Exception:
        pass

    # Market conditions from snapshot
    summary = state["data"].get("snapshot_summary", {})
    if summary.get("funding_rate") is not None:
        constraints["funding_rate"] = summary["funding_rate"]
    if summary.get("volatility") is not None:
        constraints["volatility"] = summary["volatility"]

    return constraints


async def make_verdict(state: ArenaState) -> dict:
    from cryptotrader.debate.verdict import make_verdict_llm, make_verdict_weighted

    analyses = state["data"].get("analyses", {})
    use_llm_verdict = state["metadata"].get("llm_verdict", True)

    if use_llm_verdict:
        model = state["metadata"].get("verdict_model", state["metadata"].get("debate_model", "gpt-4o-mini"))
        # Gather risk constraints and calibration data to inject into verdict prompt
        constraints = await _gather_risk_constraints(state)
        calibration = state["data"].get("verdict_calibration", "")
        verdict = await make_verdict_llm(analyses, model=model, constraints=constraints, calibration=calibration)
    else:
        scores = state.get("divergence_scores") or [0.0]
        threshold = state["metadata"].get("divergence_hold_threshold", 0.7)
        verdict = make_verdict_weighted(analyses, scores[-1], threshold)

    return {
        "data": {
            "verdict": {
                "action": verdict.action,
                "confidence": verdict.confidence,
                "position_scale": verdict.position_scale,
                "divergence": verdict.divergence,
                "reasoning": verdict.reasoning,
                "thesis": verdict.thesis,
                "invalidation": verdict.invalidation,
            }
        }
    }


# Module-level cache for RiskGate to preserve circuit breaker state across invocations
_risk_gate_cache: dict[str, Any] = {}

# Per-pair PaperExchange cache to prevent cross-pair balance contamination
_paper_exchanges: dict[str, Any] = {}

# Lazy notifier — reads config once
_notifier_instance: Any = None


def _get_notifier(state: ArenaState) -> Any:
    global _notifier_instance
    if _notifier_instance is None:
        from cryptotrader.config import load_config
        from cryptotrader.notifications import Notifier

        cfg = load_config().notifications
        _notifier_instance = Notifier(cfg.webhook_url, cfg.enabled, cfg.events)
    return _notifier_instance


async def risk_check(state: ArenaState) -> dict:
    from cryptotrader.config import load_config
    from cryptotrader.models import TradeVerdict
    from cryptotrader.portfolio.manager import PortfolioManager
    from cryptotrader.risk.gate import RiskGate
    from cryptotrader.risk.state import RedisStateManager

    redis_url = state["metadata"].get("redis_url")
    cache_key = redis_url or "_default"
    if cache_key not in _risk_gate_cache:
        config = load_config()
        redis_state = RedisStateManager(redis_url)
        _risk_gate_cache[cache_key] = RiskGate(config.risk, redis_state)
    gate = _risk_gate_cache[cache_key]

    vd = state["data"]["verdict"]
    verdict = TradeVerdict(**vd)

    # Extract recent prices and returns from snapshot OHLCV for CVaR/volatility checks
    # Normalize to daily returns regardless of candle interval
    snapshot = state["data"].get("snapshot")
    recent_prices = []
    returns_daily = []
    if snapshot and hasattr(snapshot, "market") and snapshot.market.ohlcv is not None:
        closes = snapshot.market.ohlcv["close"].dropna().tolist()
        recent_prices = closes[-60:]
        if len(closes) >= 2:
            bar_returns = [
                (closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes)) if closes[i - 1] > 0
            ]
            # Aggregate sub-daily bars into daily returns
            timeframe = state["metadata"].get("timeframe", "1h")
            _tf_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
            bars_per_day = int(86_400_000 / _tf_ms.get(timeframe, 3_600_000))
            if bars_per_day > 1 and len(bar_returns) >= bars_per_day:
                returns_daily = []
                for j in range(0, len(bar_returns) - bars_per_day + 1, bars_per_day):
                    chunk = bar_returns[j : j + bars_per_day]
                    daily = 1.0
                    for r in chunk:
                        daily *= 1 + r
                    returns_daily.append(daily - 1)
            else:
                returns_daily = bar_returns

    # Load real portfolio state if database is available
    db_url = state["metadata"].get("database_url")
    pm = PortfolioManager(db_url)
    try:
        pm_data = await pm.get_portfolio()
        daily_pnl = await pm.get_daily_pnl()
        drawdown = await pm.get_drawdown()
        pm_returns = await pm.get_returns()
    except Exception:
        pm_data = {"total_value": 0, "positions": {}}
        daily_pnl = 0.0
        drawdown = 0.0
        pm_returns = []

    # Use PortfolioManager data if it has real positions, else fall back to defaults
    has_real_portfolio = pm_data.get("total_value", 0) > 0
    portfolio = state["data"].get(
        "portfolio",
        {
            "total_value": pm_data["total_value"] if has_real_portfolio else 10000,
            "positions": pm_data.get("positions", {}),
            "daily_pnl": daily_pnl,
            "drawdown": drawdown,
            "returns_60d": pm_returns if pm_returns else returns_daily,
            "recent_prices": recent_prices,
            "funding_rate": state["data"].get("snapshot_summary", {}).get("funding_rate", 0),
            "api_latency_ms": 100,
            "pair": state["metadata"]["pair"],
        },
    )
    result = await gate.check(verdict, portfolio)

    # Fire circuit_breaker notification if triggered
    if not result.passed and result.rejected_by == "daily_loss_limit" and "Circuit breaker" in (result.reason or ""):
        try:
            notifier = _get_notifier(state)
            await notifier.notify(
                "circuit_breaker",
                {
                    "pair": state["metadata"]["pair"],
                    "reason": result.reason,
                },
            )
        except Exception:
            pass

    return {
        "data": {
            "risk_gate": {
                "passed": result.passed,
                "rejected_by": result.rejected_by,
                "reason": result.reason,
            }
        }
    }


def risk_router(state: ArenaState) -> str:
    rg = state["data"].get("risk_gate", {})
    return "approved" if rg.get("passed", False) else "rejected"


def _get_exchange(state: ArenaState, pair: str):
    """Get exchange instance (paper or live) for the given pair."""
    from cryptotrader.execution.exchange import LiveExchange
    from cryptotrader.execution.simulator import PaperExchange

    engine = state["metadata"].get("engine", "paper")
    if engine == "paper":
        if pair not in _paper_exchanges:
            _paper_exchanges[pair] = PaperExchange()
        return _paper_exchanges[pair], None
    else:
        cfg = state["metadata"].get("exchange_config", {})
        live_exchange = LiveExchange(
            cfg.get("exchange_id", "binance"),
            cfg.get("api_key", ""),
            cfg.get("secret", ""),
        )
        return live_exchange, live_exchange


async def _update_trade_tracking(state: ArenaState, pair: str):
    """Update trade count and cooldown after successful order."""
    redis_url = state["metadata"].get("redis_url")
    cache_key = redis_url or "_default"
    if cache_key in _risk_gate_cache:
        try:
            rsm = _risk_gate_cache[cache_key].redis_state
            await rsm.incr_trade_count()
            from cryptotrader.config import load_config

            cooldown_min = load_config().risk.cooldown.same_pair_minutes
            await rsm.set_cooldown(pair, cooldown_min)
        except Exception:
            pass


async def _update_portfolio(state: ArenaState, order, filled_amount: float, filled_price: float):
    """Update portfolio after successful trade."""
    pair = order.pair
    db_url = state["metadata"].get("database_url")
    try:
        from cryptotrader.portfolio.manager import PortfolioManager

        pm = PortfolioManager(db_url)
        portfolio = await pm.get_portfolio()
        existing = portfolio.get("positions", {}).get(pair, {})
        old_amount = existing.get("amount", 0.0)
        old_price = existing.get("avg_price", 0.0)

        if order.side == "buy":
            new_amount = old_amount + filled_amount
            new_price = (
                ((old_amount * old_price) + (filled_amount * filled_price)) / new_amount
                if new_amount > 0
                else filled_price
            )
        else:
            new_amount = old_amount - filled_amount
            new_price = old_price if new_amount > 0 else 0.0

        await pm.update_position("default", pair, new_amount, new_price)
        total = sum(p["amount"] * p["avg_price"] for p in (await pm.get_portfolio()).get("positions", {}).values())
        await pm.snapshot("default", total)
    except Exception:
        logger.warning("Portfolio write-back failed for %s", pair, exc_info=True)


async def place_order(state: ArenaState) -> dict:
    from cryptotrader.models import Order

    verdict = state["data"]["verdict"]
    if verdict["action"] == "hold":
        return {"data": {"order": None}}

    pair = state["metadata"]["pair"]
    price = (
        state["data"]
        .get(
            "snapshot_summary",
        )
        .get("price", 0)
    )
    scale = verdict.get("position_scale", 1.0)
    total = state["data"].get("portfolio", {}).get("total_value", 10000)
    if price <= 0:
        return {"data": {"order": None}}
    max_single_pct = state["metadata"].get("max_single_pct", 0.1)
    amount = (total * max_single_pct * scale) / price

    order = Order(
        pair=pair,
        side="buy" if verdict["action"] == "long" else "sell",
        amount=amount,
        price=price,
    )

    exchange, live_exchange = _get_exchange(state, pair)

    try:
        result = await exchange.place_order(order)
    finally:
        if live_exchange is not None:
            await live_exchange.close()

    status = result.get("status", "")
    if status not in ("filled", "partially_filled"):
        return {"data": {"order": None}}

    # Use actual filled amount/price from exchange result when available
    filled_amount = result.get("filled", order.amount) if status == "partially_filled" else order.amount
    filled_price = result.get("price", order.price)

    await _update_trade_tracking(state, pair)

    order_data = {
        "pair": order.pair,
        "side": order.side,
        "amount": filled_amount,
        "price": filled_price,
        "status": status,
    }

    await _update_portfolio(state, order, filled_amount, filled_price)

    # Fire-and-forget notification
    try:
        notifier = _get_notifier(state)
        await notifier.notify("trade", {"pair": order.pair, "order": order_data})
    except Exception:
        pass

    return {"data": {"order": order_data}}


async def journal_trade(state: ArenaState) -> dict:
    """Journal a successful trade — mirrors journal_rejection but includes order."""
    from cryptotrader.journal.commit import build_commit
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.models import AgentAnalysis, GateResult, Order, TradeVerdict

    db_url = state["metadata"].get("database_url")
    store = JournalStore(db_url)

    raw_analyses = state["data"].get("analyses", {})
    analyses = {}
    for k, v in raw_analyses.items():
        if isinstance(v, dict):
            analyses[k] = AgentAnalysis(
                agent_id=k,
                pair=state["metadata"]["pair"],
                direction=v.get("direction", "neutral"),
                confidence=v.get("confidence", 0.5),
                reasoning=v.get("reasoning", ""),
                key_factors=v.get("key_factors", []),
                risk_flags=v.get("risk_flags", []),
            )
        else:
            analyses[k] = v

    raw_verdict = state["data"].get("verdict")
    verdict = None
    if raw_verdict and isinstance(raw_verdict, dict):
        verdict = TradeVerdict(**{k: v for k, v in raw_verdict.items() if k in TradeVerdict.__dataclass_fields__})

    raw_gate = state["data"].get("risk_gate")
    risk_gate = None
    if raw_gate and isinstance(raw_gate, dict):
        risk_gate = GateResult(**{k: v for k, v in raw_gate.items() if k in GateResult.__dataclass_fields__})

    raw_order = state["data"].get("order")
    order = None
    if raw_order and isinstance(raw_order, dict):
        order = Order(
            pair=raw_order.get("pair", ""),
            side=raw_order.get("side", "buy"),
            amount=raw_order.get("amount", 0),
            price=raw_order.get("price", 0),
        )

    commit = build_commit(
        pair=state["metadata"]["pair"],
        snapshot_summary=state["data"].get("snapshot_summary", {}),
        analyses=analyses,
        debate_rounds=state.get("debate_round", 0),
        divergence=state.get("divergence_scores", [0.0])[-1],
        verdict=verdict,
        risk_gate=risk_gate,
        order=order,
        parent_hash=None,
    )
    await store.commit(commit)
    return {"data": {"journal_hash": commit.hash}}


async def journal_rejection(state: ArenaState) -> dict:
    from cryptotrader.journal.commit import build_commit
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.models import AgentAnalysis, GateResult, TradeVerdict

    db_url = state["metadata"].get("database_url")
    store = JournalStore(db_url)

    # Reconstruct dataclass instances from graph dicts
    raw_analyses = state["data"].get("analyses", {})
    analyses = {}
    for k, v in raw_analyses.items():
        if isinstance(v, dict):
            analyses[k] = AgentAnalysis(
                agent_id=k,
                pair=state["metadata"]["pair"],
                direction=v.get("direction", "neutral"),
                confidence=v.get("confidence", 0.5),
                reasoning=v.get("reasoning", ""),
                key_factors=v.get("key_factors", []),
                risk_flags=v.get("risk_flags", []),
            )
        else:
            analyses[k] = v

    raw_verdict = state["data"].get("verdict")
    verdict = None
    if raw_verdict and isinstance(raw_verdict, dict):
        verdict = TradeVerdict(**{k: v for k, v in raw_verdict.items() if k in TradeVerdict.__dataclass_fields__})

    raw_gate = state["data"].get("risk_gate")
    risk_gate = None
    if raw_gate and isinstance(raw_gate, dict):
        risk_gate = GateResult(**{k: v for k, v in raw_gate.items() if k in GateResult.__dataclass_fields__})

    commit = build_commit(
        pair=state["metadata"]["pair"],
        snapshot_summary=state["data"].get("snapshot_summary", {}),
        analyses=analyses,
        debate_rounds=state.get("debate_round", 0),
        divergence=state.get("divergence_scores", [0.0])[-1],
        verdict=verdict,
        risk_gate=risk_gate,
        order=None,
        parent_hash=None,
    )
    await store.commit(commit)

    # Fire-and-forget rejection notification
    try:
        notifier = _get_notifier(state)
        raw_gate = state["data"].get("risk_gate", {})
        await notifier.notify(
            "rejection",
            {
                "pair": state["metadata"]["pair"],
                "rejected_by": raw_gate.get("rejected_by"),
                "reason": raw_gate.get("reason"),
            },
        )
    except Exception:
        pass

    return {"data": {"journal_hash": commit.hash}}


# ── Graph builder ──


def build_trading_graph(config: dict | None = None) -> Any:
    return _build_full_graph(config)


def build_lite_graph(config: dict | None = None) -> Any:
    """Lightweight graph for backtesting: skip debate, go straight to verdict."""
    graph = StateGraph(ArenaState)

    graph.add_node("collect_data", collect_snapshot)
    graph.add_node("inject_experience", verbal_reinforcement)
    graph.add_node("tech_agent", tech_analyze)
    graph.add_node("chain_agent", chain_analyze)
    graph.add_node("news_agent", news_analyze)
    graph.add_node("macro_agent", macro_analyze)
    graph.add_node("verdict", make_verdict)

    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "inject_experience")
    graph.add_edge("inject_experience", "tech_agent")
    graph.add_edge("inject_experience", "chain_agent")
    graph.add_edge("inject_experience", "news_agent")
    graph.add_edge("inject_experience", "macro_agent")
    graph.add_edge("tech_agent", "verdict")
    graph.add_edge("chain_agent", "verdict")
    graph.add_edge("news_agent", "verdict")
    graph.add_edge("macro_agent", "verdict")
    graph.add_edge("verdict", END)

    return graph.compile()


# ── Bull/Bear debate nodes ──


async def bull_bear_debate(state: ArenaState) -> dict:
    from cryptotrader.debate.researchers import run_debate

    analyses = state["data"].get("analyses", {})
    model = state["metadata"].get("debate_model", "gpt-4o-mini")
    rounds = state["metadata"].get("debate_rounds", 2)
    debate = await run_debate(analyses, rounds=rounds, model=model)
    return {"data": {"debate": debate}}


async def judge_verdict(state: ArenaState) -> dict:
    from cryptotrader.debate.researchers import judge_debate

    debate = state["data"]["debate"]
    pair = state["metadata"]["pair"]
    model = state["metadata"].get("verdict_model", state["metadata"].get("debate_model", "gpt-4o-mini"))
    result = await judge_debate(debate, pair, model=model)
    return {
        "data": {
            "verdict": {
                "action": result["action"],
                "confidence": result["confidence"],
                "position_scale": result["confidence"],
                "divergence": 0.0,
                "reasoning": result["reasoning"],
            }
        }
    }


def build_debate_graph(config: dict | None = None) -> Any:
    """Lite graph + bull/bear adversarial debate before verdict."""
    graph = StateGraph(ArenaState)

    graph.add_node("collect_data", collect_snapshot)
    graph.add_node("inject_experience", verbal_reinforcement)
    graph.add_node("tech_agent", tech_analyze)
    graph.add_node("chain_agent", chain_analyze)
    graph.add_node("news_agent", news_analyze)
    graph.add_node("macro_agent", macro_analyze)
    graph.add_node("debate", bull_bear_debate)
    graph.add_node("verdict", judge_verdict)

    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "inject_experience")
    graph.add_edge("inject_experience", "tech_agent")
    graph.add_edge("inject_experience", "chain_agent")
    graph.add_edge("inject_experience", "news_agent")
    graph.add_edge("inject_experience", "macro_agent")
    graph.add_edge("tech_agent", "debate")
    graph.add_edge("chain_agent", "debate")
    graph.add_edge("news_agent", "debate")
    graph.add_edge("macro_agent", "debate")
    graph.add_edge("debate", "verdict")
    graph.add_edge("verdict", END)

    return graph.compile()


def _build_full_graph(config: dict | None = None) -> Any:
    """Full pipeline: agents → 2 fixed debate rounds → AI verdict → risk gate → execute.

    Phase 4C: Fixed 2 debate rounds instead of convergence-seeking loop.
    Agents are encouraged to maintain disagreement when data supports it.
    """
    graph = StateGraph(ArenaState)

    graph.add_node("collect_data", collect_snapshot)
    graph.add_node("inject_experience", verbal_reinforcement)
    graph.add_node("tech_agent", tech_analyze)
    graph.add_node("chain_agent", chain_analyze)
    graph.add_node("news_agent", news_analyze)
    graph.add_node("macro_agent", macro_analyze)
    graph.add_node("debate_round_1", debate_round)
    graph.add_node("debate_round_2", debate_round)
    graph.add_node("verdict", make_verdict)
    graph.add_node("risk_gate", risk_check)
    graph.add_node("execute", place_order)
    graph.add_node("record_trade", journal_trade)
    graph.add_node("record_rejection", journal_rejection)

    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "inject_experience")
    graph.add_edge("inject_experience", "tech_agent")
    graph.add_edge("inject_experience", "chain_agent")
    graph.add_edge("inject_experience", "news_agent")
    graph.add_edge("inject_experience", "macro_agent")
    # Fan-in to first debate round
    graph.add_edge("tech_agent", "debate_round_1")
    graph.add_edge("chain_agent", "debate_round_1")
    graph.add_edge("news_agent", "debate_round_1")
    graph.add_edge("macro_agent", "debate_round_1")
    # Fixed 2 rounds — no convergence-seeking
    graph.add_edge("debate_round_1", "debate_round_2")
    graph.add_edge("debate_round_2", "verdict")
    graph.add_edge("verdict", "risk_gate")
    graph.add_conditional_edges(
        "risk_gate",
        risk_router,
        {
            "approved": "execute",
            "rejected": "record_rejection",
        },
    )
    graph.add_edge("execute", "record_trade")
    graph.add_edge("record_trade", END)
    graph.add_edge("record_rejection", END)

    return graph.compile()


def build_supervisor_graph_v2(config: dict | None = None) -> Any:
    """Build graph using LangChain official supervisor pattern.

    Alternative to build_trading_graph() using progressive disclosure.
    """
    from cryptotrader.graph_supervisor import build_supervisor_graph

    return build_supervisor_graph(config)
