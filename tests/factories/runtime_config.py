"""Runtime configuration domain factories."""

from __future__ import annotations

from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
from cryptotrader.runtime_config.models import (
    ExecutionConfig,
    MarketDataConfig,
    RuntimeConfigDocument,
    SignalComponentConfig,
    SignalConfig,
)
from cryptotrader.venues.models import VenueConnection

INSTALLED_SIGNALS = {"kronos", "llm_committee"}
INSTALLED_ADAPTERS = {"paper", "okx", "bybit"}
INSTALLED_MARKET_SOURCES = {"default", "fixture-market"}


def connection(
    connection_id: str = "paper-local",
    environment: str = "paper",
    **overrides,
) -> VenueConnection:
    values = {
        "id": connection_id,
        "label": connection_id,
        "adapter_id": "paper" if environment == "paper" else "bybit" if environment == "testnet" else "okx",
        "environment": environment,
        "enabled": True,
        "credential_ref": "live-credentials" if environment == "live" else None,
        "leverage": 1,
        "margin_mode": "cross" if environment == "paper" else "isolated",
        "canary_only": False,
        "parameters": {"initial_equity": "10000"} if environment == "paper" else {},
    }
    return VenueConnection(**(values | overrides))


def allocation(connection_id: str = "paper-local", weight: float = 1.0, **overrides) -> ConnectionAllocation:
    return ConnectionAllocation(connection_id=connection_id, enabled=True, weight=weight, **overrides)


def book(
    book_id: str = "simulation",
    capital_scope: str = "simulated",
    *allocations: ConnectionAllocation,
    **overrides,
) -> ExecutionBook:
    values = {
        "id": book_id,
        "label": book_id,
        "capital_scope": capital_scope,
        "enabled": True,
        "hitl_required": False,
        "allocations": allocations or (allocation(),),
    }
    return ExecutionBook(**(values | overrides))


def signal_config(**overrides) -> SignalConfig:
    values = {
        "components": (
            SignalComponentConfig(component_id="kronos", enabled=True, weight=0.6),
            SignalComponentConfig(component_id="llm_committee", enabled=True, weight=0.4),
        ),
        "neutral_threshold": 0.2,
        "max_target_ratio": 1.0,
        "atr_stop_multiplier": 2.0,
        "reward_ratio": 2.0,
    }
    return SignalConfig(**(values | overrides))


def market_config(**overrides) -> MarketDataConfig:
    return MarketDataConfig(**({"source_id": "default"} | overrides))


def runtime_document(
    *,
    connections: tuple[VenueConnection, ...] = (),
    books: tuple[ExecutionBook, ...] = (),
    **overrides,
) -> RuntimeConfigDocument:
    values = {
        "market_data": market_config(),
        "signals": signal_config(),
        "execution": ExecutionConfig(connections=connections, books=books, pairs=("BTC/USDT", "BTC/USDT:USDT")),
    }
    return RuntimeConfigDocument(**(values | overrides))


def runtime_document_with_weights(first: float, second: float) -> RuntimeConfigDocument:
    first_connection = connection("paper-a")
    second_connection = connection("paper-b")
    return runtime_document(
        connections=(first_connection, second_connection),
        books=(book("simulation", "simulated", allocation("paper-a", first), allocation("paper-b", second)),),
    )


def active_document(**overrides) -> RuntimeConfigDocument:
    from cryptotrader.runtime_config.models import InfrastructureConfig

    values = {
        "connections": (connection(),),
        "books": (book(),),
        "infrastructure": InfrastructureConfig(redis_url="redis://runtime-test:6379/0"),
    }
    return runtime_document(
        **(values | overrides),
    )
