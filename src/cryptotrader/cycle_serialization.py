"""TradingCycle 领域对象与 JSON 审计快照之间的显式转换。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Any, cast

from cryptotrader.decision.models import TargetPosition, TradePlan
from cryptotrader.pair import Pair
from cryptotrader.profiles.models import ComponentWeight, SignalProfile
from cryptotrader.signals.fusion import ComponentContribution, FusedSignal
from cryptotrader.signals.models import ComponentSignal, SignalContext


def json_value(value: Any) -> Any:
    """把组件 details 等扩展字段收敛为可持久化 JSON 值。"""

    def encode(item):
        if is_dataclass(item):
            return {field.name: getattr(item, field.name) for field in fields(item)}
        if isinstance(item, Mapping):
            return dict(item)
        if isinstance(item, set | frozenset):
            return list(item)
        return str(item)

    return json.loads(json.dumps(value, default=encode))


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


def signal_profile_payload(profile: SignalProfile) -> dict[str, Any]:
    return {
        "revision": profile.revision,
        "components": [
            {
                "component_id": item.component_id,
                "enabled": item.enabled,
                "weight": item.weight,
            }
            for item in profile.components
        ],
        "neutral_threshold": profile.neutral_threshold,
        "max_target_ratio": profile.max_target_ratio,
        "atr_stop_multiplier": profile.atr_stop_multiplier,
        "reward_ratio": profile.reward_ratio,
        "hitl_required": profile.hitl_required,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at is not None else None,
    }


def signal_profile_from_payload(payload: Mapping[str, Any]) -> SignalProfile:
    raw_updated_at = payload.get("updated_at")
    return SignalProfile(
        revision=int(payload["revision"]),
        components=tuple(ComponentWeight(**item) for item in payload["components"]),
        neutral_threshold=float(payload["neutral_threshold"]),
        max_target_ratio=float(payload["max_target_ratio"]),
        atr_stop_multiplier=float(payload["atr_stop_multiplier"]),
        reward_ratio=float(payload["reward_ratio"]),
        hitl_required=bool(payload["hitl_required"]),
        updated_at=datetime.fromisoformat(str(raw_updated_at)) if raw_updated_at else None,
    )


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


def signal_context_payload(context: SignalContext) -> dict[str, Any]:
    return {
        "available": True,
        "pair": context.pair.canonical(),
        "as_of": context.as_of.isoformat(),
        "market_data_source_id": context.market_data_source_id,
        "market_type": context.market_type,
        "current_price": context.current_price,
        "atr": context.atr,
    }


def signal_context_from_payload(payload: Mapping[str, Any]) -> SignalContext:
    return SignalContext(
        pair=Pair.parse(str(payload["pair"])),
        as_of=datetime.fromisoformat(str(payload["as_of"])),
        market_data_source_id=str(payload["market_data_source_id"]),
        market_type=cast("Any", payload["market_type"]),
        current_price=float(payload["current_price"]),
        atr=float(payload["atr"]),
        snapshots={},
    )
