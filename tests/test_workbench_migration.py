"""Explicit credential cutover against an isolated temporary SQLite database."""

import base64
import json

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import select


async def test_old_credentials_migrate_without_changing_config_or_tokens(tmp_path):
    from cryptotrader.db import get_async_session
    from cryptotrader.migrations.workbench import migrate_venue_credentials, migrate_workbench_schema
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository, _RuntimeCredentialRow
    from cryptotrader.runtime_config.secrets import CredentialVault, TokenPayload
    from tests.factories.runtime_config import connection, runtime_document

    key = b"m" * 32
    vault = CredentialVault(base64.urlsafe_b64encode(key).decode())
    document = runtime_document()
    document = document.model_copy(
        update={
            "execution": document.execution.model_copy(
                update={
                    "connections": (
                        connection(
                            "okx", environment="demo", adapter_id="okx", credential_ref="old-ref", parameters={}
                        ),
                    ),
                    "books": (),
                }
            )
        }
    )
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'migration.db'}"
    await migrate_workbench_schema(database_url)
    repository = RuntimeConfigRepository(database_url, vault, default_factory=lambda: document)
    snapshot = await repository.get_or_create()
    await repository.put_token(snapshot.revision, "llm-gateway", TokenPayload(token="independent-token"))
    snapshot = await repository.get_or_create()
    nonce = b"n" * 12
    legacy_values = {
        "api_key": "original-key",  # pragma: allowlist secret
        "secret": "original-secret",  # pragma: allowlist secret
        "passphrase": "original-phrase",  # pragma: allowlist secret
    }
    old = (
        b"\x01"
        + nonce
        + AESGCM(key).encrypt(
            nonce,
            json.dumps(legacy_values).encode(),
            b"old-ref",
        )
    )
    with pytest.raises(ValueError, match="unsupported credential envelope"):
        vault.open("old-ref", old)
    session = await get_async_session(repository.database_url)
    async with session:
        session.add(
            _RuntimeCredentialRow(credential_ref="old-ref", encrypted_payload=old, updated_at=snapshot.updated_at)
        )
        await session.commit()
    backup = tmp_path / "credentials-backup.json"
    assert await migrate_venue_credentials(repository.database_url, vault, backup) == 1
    assert await migrate_venue_credentials(repository.database_url, vault, backup) == 0
    assert "original-key" not in backup.read_text()
    assert json.loads(backup.read_text())["credentials"][0]["encrypted_payload"] == base64.b64encode(old).decode()
    assert (await repository.get_or_create()) == snapshot
    assert (await repository.reveal_credentials("old-ref")).values["api_key"].get_secret_value() == "original-key"
    assert (await repository.reveal_token("llm-gateway")).token == "independent-token"
    session = await get_async_session(repository.database_url)
    async with session:
        row = await session.scalar(
            select(_RuntimeCredentialRow).where(_RuntimeCredentialRow.credential_ref == "old-ref")
        )
        assert b"original-key" not in row.encrypted_payload
