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
from cryptotrader.decision.analysis import SignalAnalysisService
from cryptotrader.decision.engine import DecisionEngine
from cryptotrader.market_sources.registry import MarketSourceRegistry
from cryptotrader.pair import Pair
from cryptotrader.runtime_config.repository import LLM_GATEWAY_CREDENTIAL_REF, RuntimeConfigRepository
from cryptotrader.runtime_config.secrets import CredentialVault
from cryptotrader.signals.fusion import WeightedSignalFusion
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


def strict_llm_factory(config, gateway_key: str, response_observer=None):
    """Bind the database gateway while prohibiting transparent model fallback."""
    from cryptotrader.agents.base import create_runtime_llm_factory

    runtime_factory = create_runtime_llm_factory(config, api_key=gateway_key, response_observer=response_observer)

    def invoke(**kwargs):
        kwargs["with_fallback"] = False
        return runtime_factory(**kwargs)

    return invoke


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
    if signal.component_id == "kronos":
        if signal.details.get("predictor_executed") is not True:
            raise RuntimeError("signal canary requires observed Kronos predictor execution")
        summary["predictor_executed"] = True
    return summary


async def run_signal_canary(pair_text: str) -> dict[str, Any]:
    from cryptotrader.bootstrap import BootstrapSettings

    settings = BootstrapSettings.from_environment()
    repository = RuntimeConfigRepository(settings.database_url, CredentialVault(settings.config_master_key))
    snapshot = await repository.get_existing()
    if not snapshot.operational:
        raise RuntimeError("active runtime configuration is required")
    gateway_key = (await repository.reveal_token(LLM_GATEWAY_CREDENTIAL_REF)).token
    observed_models: dict[str, set[str]] = {}

    def observe(role: str, model: str) -> None:
        observed_models.setdefault(role, set()).add(model)

    events = NullCycleEventSink()
    registry = SignalComponentRegistry.discover(
        snapshot.document,
        events,
        llm_gateway_key=gateway_key,
        llm_factory_builder=lambda config: strict_llm_factory(config, gateway_key, observe),
    )
    profile = snapshot.document.signals.to_profile(snapshot.revision)
    components = registry.enabled(profile)
    ids = frozenset(component.id for component in components)
    if not ids >= REQUIRED_COMPONENT_IDS:
        raise RuntimeError("Kronos and llm_committee must both be enabled for signal canary")
    market_registry = MarketSourceRegistry.discover(snapshot.document.market_data)
    source = market_registry.require(snapshot.document.market_data.source_id)
    analysis = SignalAnalysisService(
        market_source=source,
        registry=registry,
        runner=ComponentRunner(events),
        fusion=WeightedSignalFusion(),
        decisions=DecisionEngine(),
    )
    outcome = await analysis.analyze(Pair.parse(pair_text), snapshot, datetime.now(UTC))
    if (
        outcome.failure is not None
        or outcome.context is None
        or outcome.fused_signal is None
        or outcome.target_position is None
    ):
        raise RuntimeError("signal canary analysis did not complete")
    context = outcome.context
    signals = outcome.component_signals
    required_roles = {"tech_agent", "chain_agent", "news_agent", "macro_agent", "debate", "committee_summary"}
    if set(observed_models) != required_roles or any(not models for models in observed_models.values()):
        raise RuntimeError("signal canary requires actual response model metadata for every committee role")
    fused = outcome.fused_signal
    target = outcome.target_position
    model_ids = {role: sorted(models) for role, models in observed_models.items()}
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
