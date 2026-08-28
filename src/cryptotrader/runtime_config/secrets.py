"""Encrypted credential payloads with redacted text representations."""

from __future__ import annotations

import base64
import binascii
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, ValidationError


class CredentialPayload(BaseModel):
    """Credential material that must never be rendered into logs or API state."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, strict=True)

    api_key: str
    secret: str
    passphrase: str | None = None

    def __repr__(self) -> str:
        return "CredentialPayload(**redacted**)"

    def __str__(self) -> str:
        return self.__repr__()


class CredentialVault:
    """Seal credential payloads in a versioned AES-GCM envelope."""

    VERSION = b"\x01"
    _NONCE_BYTES = 12
    _MIN_ENVELOPE_BYTES = 30
    _KEY_ERROR = "CONFIG_MASTER_KEY must encode a 32-byte key"

    def __init__(self, encoded_key: str) -> None:
        try:
            encoded = encoded_key.encode("ascii")
            padding = b"=" * (-len(encoded) % 4)
            key = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
        except (AttributeError, UnicodeEncodeError, ValueError, binascii.Error):
            raise ValueError(self._KEY_ERROR) from None
        if len(key) != 32:
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
