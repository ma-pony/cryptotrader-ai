"""Run real market, Kronos, and committee inference without entering execution."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from cryptotrader.cycle_events import NullCycleEventSink
from cryptotrader.decision.engine import DecisionEngine
from cryptotrader.market_sources.registry import MarketSourceRegistry
from cryptotrader.pair import Pair
from cryptotrader.runtime_config.repository import LLM_GATEWAY_CREDENTIAL_REF, RuntimeConfigRepository
from cryptotrader.runtime_config.secrets import CredentialVault
from cryptotrader.signals.fusion import WeightedSignalFusion
from cryptotrader.signals.models import DataRequirements
from cryptotrader.signals.registry import SignalComponentRegistry
from cryptotrader.signals.runner import ComponentRunner

REQUIRED_COMPONENT_IDS = frozenset({"kronos", "llm_committee"})


def _safe_value(value: Any, key: str = "") -> Any:
    normalized = key.lower().replace("-", "_")
    if any(token in normalized for token in ("secret", "token", "key", "passphrase", "authorization", "credential")):
        return "[redacted]"
    if isinstance(value, Mapping):
        return {str(item_key): _safe_value(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    return value


def parse_signal_canary_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a non-executing real signal canary")
    parser.add_argument("--pair", required=True)
    parser.set_defaults(execute=False)
    return parser.parse_args(argv)


def _component_summary(signal, model_ids: dict[str, str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "component_id": signal.component_id,
        "direction": signal.direction,
        "confidence": signal.confidence,
    }
    if signal.component_id == "llm_committee":
        details = signal.details
        analyses = details.get("analyses", {})
        turns = details.get("debate_turns", ())
        if details.get("debate_skipped") or not turns:
            raise RuntimeError("signal canary requires an observed internal committee debate")
        summary.update({"agent_ids": sorted(analyses), "debate_turn_count": len(turns), "model_ids": model_ids})
    return summary


async def run_signal_canary(pair_text: str) -> dict[str, Any]:
    from cryptotrader.bootstrap import BootstrapSettings

    settings = BootstrapSettings.from_environment()
    repository = RuntimeConfigRepository(settings.database_url, CredentialVault(settings.config_master_key))
    snapshot = await repository.get_existing()
    if not snapshot.operational:
        raise RuntimeError("active runtime configuration is required")
    gateway_key = (await repository.reveal_token(LLM_GATEWAY_CREDENTIAL_REF)).token
    events = NullCycleEventSink()
    registry = SignalComponentRegistry.discover(snapshot.document, events, llm_gateway_key=gateway_key)
    profile = snapshot.document.signals.to_profile(snapshot.revision)
    components = registry.enabled(profile)
    ids = frozenset(component.id for component in components)
    if not ids >= REQUIRED_COMPONENT_IDS:
        raise RuntimeError("Kronos and llm_committee must both be enabled for signal canary")
    market_registry = MarketSourceRegistry.discover(snapshot.document.market_data)
    source = market_registry.require(snapshot.document.market_data.source_id)
    requirements = DataRequirements.merge(*(component.requirements() for component in components))
    context = await source.collect(Pair.parse(pair_text), datetime.now(UTC), requirements)
    signals = await ComponentRunner(events).run(components, context)
    fused = WeightedSignalFusion().fuse(signals, profile.components)
    target = DecisionEngine().target_for(fused, profile)
    models = snapshot.document.llm.models
    model_ids = {
        "tech_agent": models.tech_agent or models.analysis,
        "chain_agent": models.chain_agent or models.analysis,
        "news_agent": models.news_agent or models.analysis,
        "macro_agent": models.macro_agent or models.analysis,
        "debate": models.debate or models.fallback,
        "committee_summary": models.committee_summary or models.debate or models.fallback,
    }
    return _safe_value(
        {
            "status": "completed",
            "mode": "signal_only_no_execution",
            "config_revision": snapshot.revision,
            "market_data_source_id": context.market_data_source_id,
            "components": [_component_summary(signal, model_ids) for signal in signals],
            "fused_score": fused.score,
            "target_position": {"side": target.side, "size_ratio": target.size_ratio},
        }
    )


def main(argv: list[str] | None = None) -> int:
    options = parse_signal_canary_args(argv)
    try:
        result = asyncio.run(run_signal_canary(options.pair))
    except Exception as error:
        result = {"status": "failed", "error_type": type(error).__name__}
    print(json.dumps(_safe_value(result), ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
