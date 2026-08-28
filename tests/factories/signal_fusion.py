"""信号融合测试共用的完整领域对象工厂。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from cryptotrader.decision.models import CycleRequest, TargetPosition, TradePlan
from cryptotrader.pair import Pair
from cryptotrader.profiles.models import ComponentWeight, SignalProfile
from cryptotrader.signals.models import ComponentSignal, PositionSnapshot, SignalContext


def position(side="flat", amount=0.0, size_ratio=0.0, **overrides) -> PositionSnapshot:
    values = {"side": side, "amount": amount, "size_ratio": size_ratio}
    return PositionSnapshot(**(values | overrides))


def context(
    price=100.0,
    equity=10_000.0,
    position=None,
    atr=5.0,
    market_type="swap",
    **overrides,
) -> SignalContext:
    values = {
        "pair": Pair.parse("BTC/USDT:USDT" if market_type == "swap" else "BTC/USDT"),
        "as_of": datetime(2026, 1, 1, tzinfo=UTC),
        "mode": "paper",
        "exchange_id": "okx",
        "market_type": market_type,
        "equity": equity,
        "current_price": price,
        "atr": atr,
        "current_position": position or PositionSnapshot("flat", 0.0, 0.0),
        "snapshots": {},
        "portfolio": {
            "total_value": equity,
            "cash": equity,
            "free_cash": equity,
            "positions": {},
        },
    }
    return SignalContext(**(values | overrides))


def profile(
    *components,
    kronos=0.6,
    llm=0.4,
    revision=1,
    neutral_threshold=0.2,
    max_target_ratio=1.0,
    atr_stop_multiplier=2.0,
    reward_ratio=2.0,
    hitl=False,
) -> SignalProfile:
    configured = components or (
        ComponentWeight("kronos", kronos > 0.0, kronos),
        ComponentWeight("llm_committee", llm > 0.0, llm),
    )
    return SignalProfile(
        revision=revision,
        components=configured,
        neutral_threshold=neutral_threshold,
        max_target_ratio=max_target_ratio,
        atr_stop_multiplier=atr_stop_multiplier,
        reward_ratio=reward_ratio,
        hitl_required=hitl,
    )


def signal(component_id="kronos", direction="long", confidence=0.8, **overrides) -> ComponentSignal:
    values = {
        "component_id": component_id,
        "direction": direction,
        "confidence": confidence,
        "reasoning": "fixture",
    }
    return ComponentSignal(**(values | overrides))


def trade_plan(target: TargetPosition, **overrides) -> TradePlan:
    from cryptotrader.signals.fusion import FusedSignal

    values = {
        "target": target,
        "stop_loss": None,
        "take_profit": None,
        "component_signals": (),
        "fused_signal": FusedSignal(score=0.0, contributions=(), reasoning="fixture"),
    }
    return TradePlan(**(values | overrides))


def request(pair="BTC/USDT:USDT", mode="paper", exchange_id="okx", as_of=None) -> CycleRequest:
    return CycleRequest(Pair.parse(pair), mode, exchange_id, as_of)


def risk_request(current=None, target=None, **context_overrides):
    from cryptotrader.risk.models import RiskRequest

    target = target or TargetPosition("long", 0.5)
    signal_context = context(position=current or position(), **context_overrides)
    return RiskRequest(context=signal_context, plan=trade_plan(target))


def cycle_record(**overrides):
    from cryptotrader.journal.models import TradingCycleRecord

    values = {
        "cycle_id": str(uuid4()),
        "created_at": datetime.now(UTC),
        "pair": "BTC/USDT:USDT",
        "status": "no_change",
        "profile_revision": 1,
        "profile_snapshot": {
            "revision": 1,
            "components": [
                {"component_id": "kronos", "enabled": True, "weight": 0.6},
                {"component_id": "llm_committee", "enabled": True, "weight": 0.4},
            ],
            "neutral_threshold": 0.2,
            "max_target_ratio": 1.0,
            "atr_stop_multiplier": 2.0,
            "reward_ratio": 2.0,
            "hitl_required": False,
            "updated_at": None,
        },
        "context_summary": {
            "available": True,
            "pair": "BTC/USDT:USDT",
            "as_of": "2026-01-01T00:00:00+00:00",
            "mode": "paper",
            "exchange_id": "okx",
            "market_type": "swap",
            "equity": 10_000.0,
            "current_price": 100.0,
            "atr": 5.0,
            "current_position": {
                "side": "flat",
                "amount": 0.0,
                "size_ratio": 0.0,
                "avg_price": None,
                "unrealized_pnl": 0.0,
            },
            "portfolio": {},
        },
        "component_signals": (),
        "component_error": None,
        "fused_signal": None,
        "target_position": None,
        "trade_plan": None,
        "hitl_result": None,
        "risk_result": None,
        "execution_result": None,
    }
    values.update(overrides)
    if "profile_snapshot" not in overrides:
        values["profile_snapshot"] = {
            **values["profile_snapshot"],
            "revision": values["profile_revision"],
        }
    return TradingCycleRecord(**values)
