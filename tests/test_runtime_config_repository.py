"""Encrypted runtime configuration persistence and revision CAS contracts."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import asdict

import pytest
from sqlalchemy import text

from tests.factories.runtime_config import active_document, runtime_document


def credential_payload(marker: str = "credential-marker"):
    from cryptotrader.runtime_config.secrets import CredentialPayload

    return CredentialPayload(api_key=f"{marker}-key", secret=f"{marker}-secret", passphrase=f"{marker}-phrase")


@pytest.fixture
def repository(tmp_path):
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository
    from cryptotrader.runtime_config.secrets import CredentialVault

    vault = CredentialVault(base64.urlsafe_b64encode(b"r" * 32).decode())
    return RuntimeConfigRepository(f"sqlite+aiosqlite:///{tmp_path / 'runtime-config.db'}", vault)


async def test_get_or_create_is_stable_and_round_trips_the_exact_document(repository):
    from cryptotrader.runtime_config.models import MarketDataConfig

    first = await repository.get_or_create()
    document = runtime_document(
        market_data=MarketDataConfig(
            source_id="fixture-market",
            parameters={"nested": {"symbols": ["BTC/USDT"], "window": 42}},
        )
    )

    saved = await repository.replace(first.revision, document)
    loaded = await repository.get_or_create()

    assert loaded == saved
    assert loaded.document == document
    assert loaded.document.model_dump(mode="json") == document.model_dump(mode="json")


async def test_get_existing_never_creates_schema_or_default_row(tmp_path):
    from cryptotrader.db import get_async_session
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository, RuntimeConfigUnavailable
    from cryptotrader.runtime_config.secrets import CredentialVault

    url = f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}"
    repo = RuntimeConfigRepository(url, CredentialVault(base64.urlsafe_b64encode(b"s" * 32).decode()))
    with pytest.raises(RuntimeConfigUnavailable):
        await repo.get_existing()
    session = await get_async_session(url)
    try:
        names = await session.scalars(text("SELECT name FROM sqlite_master WHERE type='table'"))
        assert set(names) == set()
    finally:
        await session.close()


async def test_get_existing_preserves_empty_runtime_tables(repository):
    from cryptotrader.db import get_async_session
    from cryptotrader.runtime_config.repository import RuntimeConfigUnavailable

    await repository.ensure_tables()
    session = await get_async_session(repository.database_url)
    try:
        names = set(await session.scalars(text("SELECT name FROM sqlite_master WHERE type='table'")))
        before = tuple(
            await session.execute(
                text("SELECT (SELECT count(*) FROM runtime_config), (SELECT count(*) FROM runtime_credentials)")
            )
        )
    finally:
        await session.close()

    with pytest.raises(RuntimeConfigUnavailable):
        await repository.get_existing()

    session = await get_async_session(repository.database_url)
    try:
        after = tuple(
            await session.execute(
                text("SELECT (SELECT count(*) FROM runtime_config), (SELECT count(*) FROM runtime_credentials)")
            )
        )
    finally:
        await session.close()

    assert {"runtime_config", "runtime_credentials"} <= names
    assert before == ((0, 0),)
    assert after == before


async def test_get_existing_preserves_existing_document_and_credentials(repository):
    from cryptotrader.db import get_async_session
    from cryptotrader.runtime_config.repository import LLM_GATEWAY_CREDENTIAL_REF
    from cryptotrader.runtime_config.secrets import TokenPayload

    first = await repository.get_or_create()
    configured = await repository.put_token(
        first.revision,
        LLM_GATEWAY_CREDENTIAL_REF,
        TokenPayload(token="staging-token"),
    )
    before_document = configured.document.model_dump(mode="json")
    session = await get_async_session(repository.database_url)
    try:
        before_rows = tuple(
            await session.execute(
                text(
                    "SELECT "
                    "(SELECT revision FROM runtime_config WHERE id = 'global'), "
                    "(SELECT document FROM runtime_config WHERE id = 'global'), "
                    "(SELECT count(*) FROM runtime_config), "
                    "(SELECT count(*) FROM runtime_credentials), "
                    "(SELECT encrypted_payload FROM runtime_credentials "
                    "WHERE credential_ref = :credential_ref)"
                ),
                {"credential_ref": LLM_GATEWAY_CREDENTIAL_REF},
            )
        )
    finally:
        await session.close()

    loaded = await repository.get_existing()

    session = await get_async_session(repository.database_url)
    try:
        after_rows = tuple(
            await session.execute(
                text(
                    "SELECT "
                    "(SELECT revision FROM runtime_config WHERE id = 'global'), "
                    "(SELECT document FROM runtime_config WHERE id = 'global'), "
                    "(SELECT count(*) FROM runtime_config), "
                    "(SELECT count(*) FROM runtime_credentials), "
                    "(SELECT encrypted_payload FROM runtime_credentials "
                    "WHERE credential_ref = :credential_ref)"
                ),
                {"credential_ref": LLM_GATEWAY_CREDENTIAL_REF},
            )
        )
    finally:
        await session.close()

    assert loaded.revision == configured.revision
    assert loaded.document.model_dump(mode="json") == before_document
    assert (await repository.reveal_token(LLM_GATEWAY_CREDENTIAL_REF)).token == "staging-token"
    assert after_rows == before_rows


async def test_replace_is_compare_and_swap_and_increments_revision(repository):
    from cryptotrader.runtime_config.repository import RevisionConflict

    current = await repository.get_or_create()
    saved = await repository.replace(current.revision, active_document())

    assert saved.revision == current.revision + 1
    with pytest.raises(RevisionConflict) as error:
        await repository.replace(current.revision, active_document())
    assert (error.value.expected, error.value.actual) == (current.revision, saved.revision)


async def test_credential_update_and_revision_change_are_one_transaction(repository):
    before = await repository.get_or_create()
    expected = credential_payload()

    after = await repository.put_credentials(before.revision, "okx-demo", expected)

    assert after.revision == before.revision + 1
    assert (await repository.credential_state("okx-demo")).configured is True
    assert await repository.reveal_credentials("okx-demo") == expected


async def test_token_update_is_cas_protected_and_never_uses_venue_payloads(repository):
    from cryptotrader.runtime_config.repository import (
        LLM_GATEWAY_CREDENTIAL_REF,
        RevisionConflict,
    )
    from cryptotrader.runtime_config.secrets import TokenPayload

    before = await repository.get_or_create()
    saved = await repository.put_token(
        before.revision,
        LLM_GATEWAY_CREDENTIAL_REF,
        TokenPayload(token="gateway-only-token"),
    )

    assert saved.revision == before.revision + 1
    assert (await repository.token_state(LLM_GATEWAY_CREDENTIAL_REF)).configured is True
    assert (await repository.reveal_token(LLM_GATEWAY_CREDENTIAL_REF)).token == "gateway-only-token"
    with pytest.raises(RevisionConflict):
        await repository.put_token(
            before.revision,
            LLM_GATEWAY_CREDENTIAL_REF,
            TokenPayload(token="replacement-token"),
        )


async def test_stale_credential_insert_rolls_back(repository):
    from cryptotrader.runtime_config.repository import CredentialNotConfigured, RevisionConflict

    before = await repository.get_or_create()
    current = await repository.replace(before.revision, before.document)

    with pytest.raises(RevisionConflict):
        await repository.put_credentials(before.revision, "stale-new", credential_payload("stale-new-marker"))

    assert (await repository.credential_state("stale-new")).configured is False
    with pytest.raises(CredentialNotConfigured):
        await repository.reveal_credentials("stale-new")
    assert (await repository.get_or_create()).revision == current.revision


async def test_stale_credential_update_rolls_back(repository):
    from cryptotrader.runtime_config.repository import RevisionConflict

    first = await repository.get_or_create()
    original = credential_payload("original-marker")
    configured = await repository.put_credentials(first.revision, "okx-demo", original)
    current = await repository.replace(configured.revision, configured.document)

    with pytest.raises(RevisionConflict):
        await repository.put_credentials(configured.revision, "okx-demo", credential_payload("stale-update-marker"))

    assert await repository.reveal_credentials("okx-demo") == original
    assert (await repository.get_or_create()).revision == current.revision


async def test_concurrent_same_reference_creation_has_one_winner_and_one_revision_conflict(repository):
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot
    from cryptotrader.runtime_config.repository import RevisionConflict

    before = await repository.get_or_create()
    first = credential_payload("concurrent-first")
    second = credential_payload("concurrent-second")

    results = await asyncio.gather(
        repository.put_credentials(before.revision, "shared-ref", first),
        repository.put_credentials(before.revision, "shared-ref", second),
        return_exceptions=True,
    )

    assert sum(isinstance(result, RuntimeConfigSnapshot) for result in results) == 1
    assert sum(isinstance(result, RevisionConflict) for result in results) == 1
    conflict = next(result for result in results if isinstance(result, RevisionConflict))
    assert (conflict.expected, conflict.actual) == (before.revision, before.revision + 1)
    assert await repository.reveal_credentials("shared-ref") in {first, second}


async def test_repository_api_objects_never_serialize_secrets(repository):
    before = await repository.get_or_create()
    marker = "visible-marker"
    await repository.put_credentials(before.revision, "bybit-testnet", credential_payload(marker))

    state = await repository.credential_state("bybit-testnet")
    serialized_state = json.dumps(asdict(state), default=str)

    assert marker not in serialized_state
    assert set(asdict(state)) == {"credential_ref", "configured", "updated_at"}


async def test_missing_credentials_return_state_and_raise_redacted_error(repository):
    from cryptotrader.runtime_config.repository import CredentialNotConfigured

    state = await repository.credential_state("missing-ref")

    assert state.configured is False
    assert state.updated_at is None
    with pytest.raises(CredentialNotConfigured, match="missing-ref") as error:
        await repository.reveal_credentials("missing-ref")
    assert "api_key" not in str(error.value)
