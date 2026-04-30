"""Transient adapter for the str→Pair migration (spec 013, Phase 3a).

This module exists only during the cascade. Once every callsite under
``src/cryptotrader/nodes/`` and ``src/cryptotrader/execution/`` consumes
``Pair`` directly (Phase 3b/3c), task T026 removes this module.

Do NOT add new callers. New code MUST take ``Pair`` directly.
"""

from __future__ import annotations

from cryptotrader.pair import Pair

__all__ = ["from_pair", "to_pair"]


def to_pair(value: Pair | str) -> Pair:
    """Coerce a value to ``Pair``. Idempotent on ``Pair`` instances."""
    if isinstance(value, Pair):
        return value
    return Pair.parse(value)


def from_pair(value: Pair | str) -> str:
    """Coerce a value to canonical ccxt unified str. Idempotent on str."""
    if isinstance(value, Pair):
        return value.canonical()
    # Round-trip through Pair to validate even when input is already str.
    return Pair.parse(value).canonical()
