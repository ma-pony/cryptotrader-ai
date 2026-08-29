"""AES-GCM credential envelope and redaction contracts."""

from __future__ import annotations

import base64
import json
import string

import pytest
from cryptography.exceptions import InvalidTag
from pydantic import ValidationError

from cryptotrader.runtime_config.secrets import CredentialPayload, CredentialVault

VALID_MASTER_KEY = base64.urlsafe_b64encode(b"k" * 32).decode()
STANDARD_MASTER_KEY = base64.b64encode(b"\xfb" * 32).decode()
STANDARD_PLUS_KEY = STANDARD_MASTER_KEY.replace("/", "_")
STANDARD_SLASH_KEY = STANDARD_MASTER_KEY.replace("+", "-")
_URLSAFE_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits + "-_"
_LAST_DATA_INDEX = _URLSAFE_ALPHABET.index(VALID_MASTER_KEY[-2])
NON_CANONICAL_MASTER_KEY = VALID_MASTER_KEY[:-2] + _URLSAFE_ALPHABET[_LAST_DATA_INDEX + 1] + "="


def test_vault_round_trip_uses_unique_nonce_and_reference_as_aad():
    vault = CredentialVault(VALID_MASTER_KEY)
    payload = CredentialPayload(api_key="key", secret="secret", passphrase="phrase")

    first = vault.seal("okx-demo", payload)
    second = vault.seal("okx-demo", payload)

    assert first != second
    assert b"secret" not in first
    assert vault.open("okx-demo", first) == payload
    with pytest.raises(InvalidTag):
        vault.open("other-ref", first)


def test_token_payload_is_redacted_and_cannot_be_used_as_venue_credentials():
    from cryptotrader.runtime_config.secrets import TokenPayload

    token = TokenPayload(token="gateway-test-token")

    assert "gateway-test-token" not in repr(token)
    assert token.token == "gateway-test-token"


@pytest.mark.parametrize(
    "encoded_key",
    [
        "not-a-valid-key",
        "%%%",
        STANDARD_PLUS_KEY,
        STANDARD_SLASH_KEY,
        VALID_MASTER_KEY.rstrip("="),
        VALID_MASTER_KEY + "=",
        NON_CANONICAL_MASTER_KEY,
    ],
)
def test_invalid_master_key_fails_closed(encoded_key):
    from cryptotrader.runtime_config.secrets import CredentialVault

    with pytest.raises(ValueError, match="32-byte"):
        CredentialVault(encoded_key)


def test_documented_urlsafe_master_key_is_accepted():
    from cryptotrader.runtime_config.secrets import CredentialVault

    assert isinstance(CredentialVault(VALID_MASTER_KEY), CredentialVault)


def test_structured_payload_validation_errors_never_retain_input_values():
    from cryptotrader.runtime_config.secrets import CredentialPayload

    sentinel = "structured-validation-sentinel"

    with pytest.raises(ValidationError) as error:
        CredentialPayload(
            api_key=sentinel.encode(),
            secret=[sentinel],
            passphrase={"value": sentinel},
            unexpected=sentinel,
        )

    structured = json.dumps(error.value.errors(), default=str)
    assert sentinel not in structured
    assert sentinel not in str(error.value)


def test_payload_text_representations_are_always_redacted():
    from cryptotrader.runtime_config.secrets import CredentialPayload

    payload = CredentialPayload(api_key="repr-key-marker", secret="repr-secret-marker", passphrase=None)

    assert repr(payload) == "CredentialPayload(**redacted**)"
    assert str(payload) == "CredentialPayload(**redacted**)"


def test_vault_rejects_unsupported_envelope_without_echoing_it():
    from cryptotrader.runtime_config.secrets import CredentialPayload, CredentialVault

    vault = CredentialVault(VALID_MASTER_KEY)
    payload = CredentialPayload(api_key="key", secret="secret")
    envelope = vault.seal("okx-demo", payload)

    with pytest.raises(ValueError, match="unsupported credential envelope") as error:
        vault.open("okx-demo", b"\x02" + envelope[1:])

    assert envelope.hex() not in str(error.value)
