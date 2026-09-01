# ruff: noqa: RUF001 -- Chinese user-facing messages use Chinese punctuation.
"""Allowlisted replay inputs. Transport, credentials and capital authorizations are never reusable."""

from cryptotrader.decision.service import configuration_summary
from cryptotrader.runtime_config.models import (
    ExecutionConfig,
    LlmConfig,
    MarketDataConfig,
    RiskConfig,
    RuntimeConfigDocument,
    RuntimeConfigSnapshot,
    SignalConfig,
)


def safe_snapshot(snapshot):
    summary = configuration_summary(snapshot)
    return {
        "version": 1,
        "revision": snapshot.revision,
        "updated_at": snapshot.updated_at.isoformat(),
        "market_data": summary["market_data"],
        "signals": summary["signals"],
        "risk": snapshot.document.risk.model_dump(mode="json"),
        "llm": snapshot.document.llm.model_dump(mode="json", exclude={"base_url"}),
    }


def restore_snapshot(value, transport_snapshot):
    from datetime import datetime

    if value.get("version") != 1:
        raise ValueError("历史记录缺少完整安全配置，不能复用；请选择当前已保存配置")
    signals = {**value["signals"], "components": []}
    for component in value["signals"]["components"]:
        identity = component.get("model_identity", {})
        parameters = dict(component["parameters"])
        if "vocabulary" in identity:
            parameters["tokenizer_name"] = identity["vocabulary"]
        signals["components"].append(
            {key: component[key] for key in ("component_id", "enabled", "weight")} | {"parameters": parameters}
        )
    # Only the current transport endpoint is bound transiently for explicit execution.
    llm = LlmConfig.model_validate(value["llm"]).model_copy(
        update={"base_url": transport_snapshot.document.llm.base_url}
    )
    document = RuntimeConfigDocument(
        market_data=MarketDataConfig.model_validate(value["market_data"]),
        signals=SignalConfig.model_validate(signals),
        risk=RiskConfig.model_validate(value["risk"]),
        llm=llm,
        execution=ExecutionConfig(),
    )
    return RuntimeConfigSnapshot(value["revision"], document, datetime.fromisoformat(value["updated_at"]))
