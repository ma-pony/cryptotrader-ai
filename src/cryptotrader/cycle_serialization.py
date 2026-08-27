"""TradingCycle 领域对象与 JSON 审计快照之间的显式转换。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from cryptotrader.decision.models import CycleRequest, TargetPosition, TradePlan
from cryptotrader.pair import Pair
from cryptotrader.signals.fusion import ComponentContribution, FusedSignal
from cryptotrader.signals.models import ComponentSignal, PositionSnapshot, SignalContext

if TYPE_CHECKING:
    from collections.abc import Mapping


def json_value(value: Any) -> Any:
    """把组件 details 等扩展字段收敛为可持久化 JSON 值。"""
    return json.loads(json.dumps(value, default=str))


def component_signal_payload(signal: ComponentSignal) -> dict[str, Any]:
    return {
        "component_id": signal.component_id,
        "direction": signal.direction,
        "confidence": signal.confidence,
        "reasoning": signal.reasoning,
        "details": json_value(signal.details),
    }


def component_signal_from_payload(payload: Mapping[str, Any]) -> ComponentSignal:
    return ComponentSignal(
        component_id=str(payload["component_id"]),
        direction=cast("Any", payload["direction"]),
        confidence=float(payload["confidence"]),
        reasoning=str(payload["reasoning"]),
        details=cast("Mapping[str, Any]", payload.get("details") or {}),
    )


def fused_signal_payload(fused: FusedSignal) -> dict[str, Any]:
    return {
        "score": fused.score,
        "reasoning": fused.reasoning,
        "contributions": [
            {
                "component_id": item.component_id,
                "weight": item.weight,
                "signed_score": item.signed_score,
                "weighted_score": item.weighted_score,
            }
            for item in fused.contributions
        ],
    }


def fused_signal_from_payload(payload: Mapping[str, Any]) -> FusedSignal:
    return FusedSignal(
        score=float(payload["score"]),
        reasoning=str(payload["reasoning"]),
        contributions=tuple(ComponentContribution(**item) for item in payload["contributions"]),
    )


def target_payload(target: TargetPosition) -> dict[str, Any]:
    return {"side": target.side, "size_ratio": target.size_ratio}


def trade_plan_payload(plan: TradePlan) -> dict[str, Any]:
    return {
        "target": target_payload(plan.target),
        "stop_loss": plan.stop_loss,
        "take_profit": plan.take_profit,
        "component_signals": [component_signal_payload(item) for item in plan.component_signals],
        "fused_signal": fused_signal_payload(plan.fused_signal),
    }


def trade_plan_from_payload(payload: Mapping[str, Any]) -> TradePlan:
    target = payload["target"]
    return TradePlan(
        target=TargetPosition(str(target["side"]), float(target["size_ratio"])),  # type: ignore[arg-type]
        stop_loss=float(payload["stop_loss"]) if payload.get("stop_loss") is not None else None,
        take_profit=float(payload["take_profit"]) if payload.get("take_profit") is not None else None,
        component_signals=tuple(component_signal_from_payload(item) for item in payload["component_signals"]),
        fused_signal=fused_signal_from_payload(payload["fused_signal"]),
    )


def cycle_request_payload(request: CycleRequest) -> dict[str, Any]:
    return {
        "pair": request.pair.canonical(),
        "mode": request.mode,
        "exchange_id": request.exchange_id,
        "as_of": request.as_of.isoformat() if request.as_of is not None else None,
    }


def cycle_request_from_payload(payload: Mapping[str, Any]) -> CycleRequest:
    raw_as_of = payload.get("as_of")
    return CycleRequest(
        pair=Pair.parse(str(payload["pair"])),
        mode=cast("Any", payload["mode"]),
        exchange_id=str(payload.get("exchange_id") or ""),
        as_of=datetime.fromisoformat(str(raw_as_of)) if raw_as_of else None,
    )


def position_payload(position: PositionSnapshot) -> dict[str, Any]:
    return {
        "side": position.side,
        "amount": position.amount,
        "size_ratio": position.size_ratio,
        "avg_price": position.avg_price,
        "unrealized_pnl": position.unrealized_pnl,
    }


def signal_context_payload(context: SignalContext) -> dict[str, Any]:
    return {
        "pair": context.pair.canonical(),
        "as_of": context.as_of.isoformat(),
        "mode": context.mode,
        "exchange_id": context.exchange_id,
        "market_type": context.market_type,
        "equity": context.equity,
        "current_price": context.current_price,
        "atr": context.atr,
        "current_position": position_payload(context.current_position),
        "portfolio": json_value(context.portfolio),
    }


def signal_context_from_payload(payload: Mapping[str, Any]) -> SignalContext:
    raw_position = payload["current_position"]
    return SignalContext(
        pair=Pair.parse(str(payload["pair"])),
        as_of=datetime.fromisoformat(str(payload["as_of"])),
        mode=cast("Any", payload["mode"]),
        exchange_id=str(payload.get("exchange_id") or ""),
        market_type=cast("Any", payload["market_type"]),
        equity=float(payload["equity"]),
        current_price=float(payload["current_price"]),
        atr=float(payload["atr"]),
        current_position=PositionSnapshot(
            side=cast("Any", raw_position["side"]),
            amount=float(raw_position["amount"]),
            size_ratio=float(raw_position["size_ratio"]),
            avg_price=(float(raw_position["avg_price"]) if raw_position.get("avg_price") is not None else None),
            unrealized_pnl=float(raw_position.get("unrealized_pnl") or 0.0),
        ),
        snapshots={},
        portfolio=cast("Mapping[str, Any]", payload.get("portfolio") or {}),
    )
