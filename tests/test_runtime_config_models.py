"""Strict runtime configuration document contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tests.factories.runtime_config import (
    INSTALLED_ADAPTERS,
    INSTALLED_SIGNALS,
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
        validate_runtime_document(document, INSTALLED_SIGNALS, {"paper"})


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
        )


def test_document_requires_enabled_allocation_weights_to_equal_one():
    from cryptotrader.runtime_config.models import validate_runtime_document

    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        validate_runtime_document(runtime_document_with_weights(0.4, 0.5), INSTALLED_SIGNALS, INSTALLED_ADAPTERS)


def test_document_rejects_unknown_enabled_connection_adapter():
    from cryptotrader.runtime_config.models import validate_runtime_document

    document = runtime_document(connections=(connection(adapter_id="missing"),))

    with pytest.raises(ValueError, match="uninstalled adapter"):
        validate_runtime_document(document, INSTALLED_SIGNALS, {"paper"})


def test_active_document_requires_an_enabled_book():
    from cryptotrader.runtime_config.models import SystemConfig, validate_runtime_document

    document = runtime_document(system=SystemConfig(active=True))

    with pytest.raises(ValueError, match="enabled execution book"):
        validate_runtime_document(document, INSTALLED_SIGNALS, INSTALLED_ADAPTERS)


def test_active_document_requires_credential_for_enabled_non_paper_connection():
    from cryptotrader.runtime_config.models import SystemConfig, validate_runtime_document

    document = runtime_document(
        system=SystemConfig(active=True),
        connections=(connection("okx-live", "live", adapter_id="okx"),),
        books=(book("live", "real", allocation("okx-live", 1.0)),),
    )

    with pytest.raises(ValueError, match="credential_ref"):
        validate_runtime_document(document, INSTALLED_SIGNALS, INSTALLED_ADAPTERS)


def test_active_document_requires_a_resolvable_market_source():
    from cryptotrader.runtime_config.models import MarketDataConfig, SystemConfig, validate_runtime_document

    document = runtime_document(
        system=SystemConfig(active=True),
        market_data=MarketDataConfig(source_id=""),
        connections=(connection(),),
        books=(book(),),
    )

    with pytest.raises(ValueError, match="market source"):
        validate_runtime_document(document, INSTALLED_SIGNALS, INSTALLED_ADAPTERS)


def test_document_is_frozen_and_rejects_unknown_top_level_fields():
    from cryptotrader.runtime_config.models import RuntimeConfigDocument

    document = runtime_document()

    with pytest.raises(ValidationError, match="extra"):
        RuntimeConfigDocument.model_validate(document.model_dump() | {"engine": "paper"})
    with pytest.raises(ValidationError, match="frozen"):
        document.system = None
