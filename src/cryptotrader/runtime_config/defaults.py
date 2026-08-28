"""Code-owned minimal runtime document used for first-time setup."""

from __future__ import annotations

from cryptotrader.runtime_config.models import (
    ExecutionConfig,
    MarketDataConfig,
    RuntimeConfigDocument,
    SignalComponentConfig,
    SignalConfig,
    SystemConfig,
)


def minimal_runtime_document() -> RuntimeConfigDocument:
    """Return the inactive setup document with no executable capital configured."""

    return RuntimeConfigDocument(
        system=SystemConfig(active=False),
        market_data=MarketDataConfig(source_id="default"),
        signals=SignalConfig(
            components=(
                SignalComponentConfig(
                    component_id="kronos",
                    enabled=True,
                    weight=0.6,
                    parameters={
                        "gate_path": "artifacts/kronos/gate_v21.pkl",
                        "model_name": "NeoQuasar/Kronos-base",
                        "tokenizer_name": "NeoQuasar/Kronos-Tokenizer-base",
                    },
                ),
                SignalComponentConfig(component_id="llm_committee", enabled=True, weight=0.4),
            ),
            neutral_threshold=0.2,
            max_target_ratio=1.0,
            atr_stop_multiplier=2.0,
            reward_ratio=2.0,
            hitl_required=False,
        ),
        execution=ExecutionConfig(),
    )
