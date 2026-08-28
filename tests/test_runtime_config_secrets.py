"""AES-GCM credential envelope and redaction contracts."""

from __future__ import annotations

import base64

import pytest
from cryptography.exceptions import InvalidTag


def test_vault_round_trip_uses_unique_nonce_and_reference_as_aad():
    from cryptotrader.runtime_config.secrets import CredentialPayload, CredentialVault

    vault = CredentialVault(base64.urlsafe_b64encode(b"k" * 32).decode())
    payload = CredentialPayload(api_key="key", secret="secret", passphrase="phrase")

    first = vault.seal("okx-demo", payload)
    second = vault.seal("okx-demo", payload)

    assert first != second
    assert b"secret" not in first
    assert vault.open("okx-demo", first) == payload
    with pytest.raises(InvalidTag):
        vault.open("other-ref", first)


@pytest.mark.parametrize("encoded_key", ["not-a-valid-key", "%%%"])
def test_invalid_master_key_fails_closed(encoded_key):
    from cryptotrader.runtime_config.secrets import CredentialVault

    with pytest.raises(ValueError, match="32-byte"):
        CredentialVault(encoded_key)


def test_payload_text_representations_are_always_redacted():
    from cryptotrader.runtime_config.secrets import CredentialPayload

    payload = CredentialPayload(api_key="repr-key-marker", secret="repr-secret-marker", passphrase=None)

    assert repr(payload) == "CredentialPayload(**redacted**)"
    assert str(payload) == "CredentialPayload(**redacted**)"


def test_vault_rejects_unsupported_envelope_without_echoing_it():
    from cryptotrader.runtime_config.secrets import CredentialPayload, CredentialVault

    vault = CredentialVault(base64.urlsafe_b64encode(b"k" * 32).decode())
    payload = CredentialPayload(api_key="key", secret="secret")
    envelope = vault.seal("okx-demo", payload)

    with pytest.raises(ValueError, match="unsupported credential envelope") as error:
        vault.open("okx-demo", b"\x02" + envelope[1:])

    assert envelope.hex() not in str(error.value)
