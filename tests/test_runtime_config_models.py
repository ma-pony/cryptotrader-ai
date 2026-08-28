"""Strict runtime configuration document contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tests.factories.runtime_config import (
    INSTALLED_ADAPTERS,
    INSTALLED_MARKET_SOURCES,
    INSTALLED_SIGNALS,
    active_document,
    allocation,
    book,
    connection,
    runtime_document,
    runtime_document_with_weights,
)

NOW = datetime(2026, 8, 28, tzinfo=UTC)


def test_minimal_document_requires_setup_and_contains_no_connection():
    from cryptotrader.runtime_config.defaults import minimal_runtime_document
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot

    snapshot = RuntimeConfigSnapshot(1, minimal_runtime_document(), NOW)

    assert snapshot.setup_required is True
    assert snapshot.document.execution.connections == ()
    assert snapshot.document.execution.books == ()


def test_document_rejects_connection_in_two_enabled_books():
    from cryptotrader.runtime_config.models import validate_runtime_document

    document = runtime_document(
        connections=(connection("paper-local", "paper"),),
        books=(
            book("simulation-a", "simulated", allocation("paper-local", 1.0)),
            book("simulation-b", "simulated", allocation("paper-local", 1.0)),
        ),
    )

    with pytest.raises(ValueError, match="one enabled book"):
        validate_runtime_document(document, INSTALLED_SIGNALS, {"paper"}, INSTALLED_MARKET_SOURCES)


@pytest.mark.parametrize(
    ("environment", "scope"),
    [("paper", "real"), ("demo", "real"), ("testnet", "real"), ("live", "simulated")],
)
def test_document_rejects_environment_scope_mismatch(environment, scope):
    from cryptotrader.runtime_config.models import validate_runtime_document

    with pytest.raises(ValueError, match="capital_scope"):
        validate_runtime_document(
            runtime_document(
                connections=(connection("venue", environment),),
                books=(book("book", scope, allocation("venue", 1.0)),),
            ),
            INSTALLED_SIGNALS,
            INSTALLED_ADAPTERS,
            INSTALLED_MARKET_SOURCES,
        )


def test_document_requires_enabled_allocation_weights_to_equal_one():
    from cryptotrader.runtime_config.models import validate_runtime_document

    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        validate_runtime_document(
            runtime_document_with_weights(0.4, 0.5),
            INSTALLED_SIGNALS,
            INSTALLED_ADAPTERS,
            INSTALLED_MARKET_SOURCES,
        )


def test_document_rejects_unknown_enabled_connection_adapter():
    from cryptotrader.runtime_config.models import validate_runtime_document

    document = runtime_document(connections=(connection(adapter_id="missing"),))

    with pytest.raises(ValueError, match="uninstalled adapter"):
        validate_runtime_document(document, INSTALLED_SIGNALS, {"paper"}, INSTALLED_MARKET_SOURCES)


def test_active_document_requires_an_enabled_book():
    from cryptotrader.runtime_config.models import SystemConfig, validate_runtime_document

    document = runtime_document(system=SystemConfig(active=True))

    with pytest.raises(ValueError, match="enabled execution book"):
        validate_runtime_document(document, INSTALLED_SIGNALS, INSTALLED_ADAPTERS, INSTALLED_MARKET_SOURCES)


def test_active_document_requires_credential_for_enabled_non_paper_connection():
    from cryptotrader.runtime_config.models import SystemConfig, validate_runtime_document

    document = runtime_document(
        system=SystemConfig(active=True),
        connections=(connection("okx-demo", "demo", adapter_id="okx"),),
        books=(book("simulation", "simulated", allocation("okx-demo", 1.0)),),
    )

    with pytest.raises(ValueError, match="credential_ref"):
        validate_runtime_document(document, INSTALLED_SIGNALS, INSTALLED_ADAPTERS, INSTALLED_MARKET_SOURCES)


def test_active_document_requires_an_installed_market_source():
    from cryptotrader.runtime_config.models import MarketDataConfig, SystemConfig, validate_runtime_document

    document = runtime_document(
        system=SystemConfig(active=True),
        market_data=MarketDataConfig(source_id="missing"),
        connections=(connection(),),
        books=(book(),),
    )

    with pytest.raises(ValueError, match="uninstalled market source"):
        validate_runtime_document(document, INSTALLED_SIGNALS, INSTALLED_ADAPTERS, INSTALLED_MARKET_SOURCES)


def test_active_document_accepts_an_installed_market_source():
    from cryptotrader.runtime_config.models import validate_runtime_document

    validate_runtime_document(active_document(), INSTALLED_SIGNALS, INSTALLED_ADAPTERS, INSTALLED_MARKET_SOURCES)


def test_document_is_frozen_and_rejects_unknown_top_level_fields():
    from cryptotrader.runtime_config.models import RuntimeConfigDocument

    document = runtime_document()

    with pytest.raises(ValidationError, match="extra"):
        RuntimeConfigDocument.model_validate(document.model_dump() | {"engine": "paper"})
    with pytest.raises(ValidationError, match="frozen"):
        document.system = None


def test_document_recursively_freezes_market_and_signal_parameters():
    from cryptotrader.runtime_config.models import MarketDataConfig, SignalComponentConfig

    document = runtime_document(
        market_data=MarketDataConfig(parameters={"nested": {"values": ["market"]}}),
        signals=runtime_document().signals.model_copy(
            update={
                "components": (
                    SignalComponentConfig(
                        component_id="kronos",
                        enabled=True,
                        weight=0.6,
                        parameters={"nested": {"values": ["signal"]}},
                    ),
                    SignalComponentConfig(component_id="llm_committee", enabled=True, weight=0.4),
                )
            }
        ),
    )

    with pytest.raises(TypeError):
        document.market_data.parameters["nested"]["values"] = ()
    with pytest.raises(TypeError):
        document.signals.components[0].parameters["nested"]["values"] = ()


@pytest.mark.parametrize("enabled", ["false", 0, 1])
def test_venue_connection_rejects_non_boolean_enabled(enabled):
    from cryptotrader.venues.models import VenueConnection

    with pytest.raises(ValueError, match="enabled"):
        VenueConnection("paper", "Paper", "paper", "paper", enabled, None, 1, "isolated")


@pytest.mark.parametrize("enabled", ["false", 0, 1])
def test_connection_allocation_rejects_non_boolean_enabled(enabled):
    from cryptotrader.execution.models import ConnectionAllocation

    with pytest.raises(ValueError, match="enabled"):
        ConnectionAllocation("paper", enabled, 1.0)


@pytest.mark.parametrize("enabled", ["false", 0, 1])
def test_execution_book_rejects_non_boolean_enabled(enabled):
    from cryptotrader.execution.models import ExecutionBook

    with pytest.raises(ValueError, match="enabled"):
        ExecutionBook("simulation", "Simulation", "simulated", enabled, False, ())


def test_execution_models_reject_invalid_critical_scalar_types():
    from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
    from cryptotrader.venues.models import VenueConnection

    with pytest.raises(ValueError, match="id"):
        VenueConnection(1, "Paper", "paper", "paper", True, None, 1, "isolated")
    with pytest.raises(ValueError, match="leverage"):
        VenueConnection("paper", "Paper", "paper", "paper", True, None, True, "isolated")
    with pytest.raises(ValueError, match="weight"):
        ConnectionAllocation("paper", True, 1)
    with pytest.raises(ValueError, match="allocations"):
        ExecutionBook("simulation", "Simulation", "simulated", True, False, [])
