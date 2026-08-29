"""Encrypted credential payloads with redacted text representations."""

from __future__ import annotations

import base64
import binascii
import os
import string
from collections.abc import Mapping
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

_URLSAFE_ALPHABET = frozenset(string.ascii_letters + string.digits + "-_")
_CREDENTIAL_FIELDS = frozenset({"api_key", "secret", "passphrase"})


class CredentialPayload(BaseModel):
    """Credential material that must never be rendered into logs or API state."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, strict=True)

    api_key: str
    secret: str
    passphrase: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _remove_values_from_invalid_input(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return None
        valid = (
            set(value) <= _CREDENTIAL_FIELDS
            and "api_key" in value
            and "secret" in value
            and isinstance(value["api_key"], str)
            and isinstance(value["secret"], str)
            and ("passphrase" not in value or value["passphrase"] is None or isinstance(value["passphrase"], str))
        )
        if valid:
            return value
        return {key if key in _CREDENTIAL_FIELDS else "unexpected": None for key in value}

    def __repr__(self) -> str:
        return "CredentialPayload(**redacted**)"

    def __str__(self) -> str:
        return self.__repr__()


class TokenPayload(BaseModel):
    """One opaque runtime secret, kept distinct from venue credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, strict=True)

    token: str

    def __repr__(self) -> str:
        return "TokenPayload(**redacted**)"

    def __str__(self) -> str:
        return self.__repr__()


class CredentialVault:
    """Seal credential payloads in a versioned AES-GCM envelope."""

    VERSION = b"\x01"
    _NONCE_BYTES = 12
    _MIN_ENVELOPE_BYTES = 30
    _KEY_ERROR = "CONFIG_MASTER_KEY must encode a 32-byte key"

    def __init__(self, encoded_key: str) -> None:
        if (
            not isinstance(encoded_key, str)
            or len(encoded_key) != 44
            or encoded_key[-1:] != "="
            or not set(encoded_key[:-1]) <= _URLSAFE_ALPHABET
        ):
            raise ValueError(self._KEY_ERROR)
        try:
            encoded = encoded_key.encode("ascii")
            key = base64.urlsafe_b64decode(encoded)
        except (UnicodeEncodeError, ValueError, binascii.Error):
            raise ValueError(self._KEY_ERROR) from None
        if len(key) != 32 or base64.urlsafe_b64encode(key).decode() != encoded_key:
            raise ValueError(self._KEY_ERROR)
        self._cipher = AESGCM(key)

    def seal(self, credential_ref: str, payload: CredentialPayload) -> bytes:
        nonce = os.urandom(self._NONCE_BYTES)
        plaintext = payload.model_dump_json().encode()
        ciphertext = self._cipher.encrypt(nonce, plaintext, credential_ref.encode())
        return self.VERSION + nonce + ciphertext

    def open(self, credential_ref: str, envelope: bytes) -> CredentialPayload:
        envelope = bytes(envelope)
        if envelope[:1] != self.VERSION or len(envelope) < self._MIN_ENVELOPE_BYTES:
            raise ValueError("unsupported credential envelope")
        plaintext = self._cipher.decrypt(
            envelope[1 : 1 + self._NONCE_BYTES],
            envelope[1 + self._NONCE_BYTES :],
            credential_ref.encode(),
        )
        try:
            return CredentialPayload.model_validate_json(plaintext)
        except (ValidationError, ValueError):
            raise ValueError("invalid credential payload") from None

    def seal_token(self, credential_ref: str, payload: TokenPayload) -> bytes:
        nonce = os.urandom(self._NONCE_BYTES)
        plaintext = payload.model_dump_json().encode()
        ciphertext = self._cipher.encrypt(nonce, plaintext, credential_ref.encode())
        return self.VERSION + nonce + ciphertext

    def open_token(self, credential_ref: str, envelope: bytes) -> TokenPayload:
        envelope = bytes(envelope)
        if envelope[:1] != self.VERSION or len(envelope) < self._MIN_ENVELOPE_BYTES:
            raise ValueError("unsupported credential envelope")
        plaintext = self._cipher.decrypt(
            envelope[1 : 1 + self._NONCE_BYTES],
            envelope[1 + self._NONCE_BYTES :],
            credential_ref.encode(),
        )
        try:
            return TokenPayload.model_validate_json(plaintext)
        except (ValidationError, ValueError):
            raise ValueError("invalid token payload") from None
