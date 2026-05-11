"""Tests for spec 021 E1 stale-cycle-lock SIGKILL recovery."""

from __future__ import annotations

import os

import pytest

from cryptotrader.risk.state import RedisStateManager, _is_owner_dead


def test_is_owner_dead_legacy_uuid_returns_false():
    """Bare-uuid format (pre-spec-021) is treated as unknown PID → safe default."""
    assert _is_owner_dead("ae04397b8a244420deadbeefcafef00d") is False


def test_is_owner_dead_live_pid_returns_false():
    """Owner_id encoding current PID → process is alive → not stealable."""
    owner = f"{os.getpid()}:abc123"
    assert _is_owner_dead(owner) is False


def test_is_owner_dead_dead_pid_returns_true():
    """PID 99999999 ~certainly doesn't exist on Linux/macOS → stealable."""
    owner = "99999999:abc123"
    assert _is_owner_dead(owner) is True


def test_is_owner_dead_invalid_pid_returns_false():
    """Malformed PID → safe default (don't steal)."""
    assert _is_owner_dead("not-an-int:xxx") is False
    assert _is_owner_dead("-1:xxx") is False


def test_is_owner_dead_bytes_input():
    """Redis returns bytes; helper must decode."""
    assert _is_owner_dead(b"99999999:abc") is True
    assert _is_owner_dead(f"{os.getpid()}:abc".encode()) is False


@pytest.mark.asyncio
async def test_try_acquire_lock_stale_dead_pid_steals():
    """Memory path: existing lock with dead PID → next acquire steals it."""
    rs = RedisStateManager(redis_url=None)  # in-memory
    # Seed a stale lock as if a dead process held it
    await rs.try_acquire_lock("cycle_lock:test", owner_id="99999999:stale", ttl=600)
    # New owner should be able to acquire (memory path detects dead PID)
    acquired = await rs.try_acquire_lock(
        "cycle_lock:test", owner_id=f"{os.getpid()}:new", ttl=600
    )
    assert acquired is True


@pytest.mark.asyncio
async def test_try_acquire_lock_live_owner_blocks():
    """Memory path: live PID owner blocks new acquisition."""
    rs = RedisStateManager(redis_url=None)
    await rs.try_acquire_lock("cycle_lock:test", owner_id=f"{os.getpid()}:a", ttl=600)
    acquired = await rs.try_acquire_lock(
        "cycle_lock:test", owner_id=f"{os.getpid()}:b", ttl=600
    )
    assert acquired is False


@pytest.mark.asyncio
async def test_try_acquire_lock_legacy_owner_blocks():
    """Memory path: legacy bare-uuid owner is treated as alive (safe default)."""
    rs = RedisStateManager(redis_url=None)
    await rs.try_acquire_lock("cycle_lock:test", owner_id="legacy-bare-uuid", ttl=600)
    acquired = await rs.try_acquire_lock(
        "cycle_lock:test", owner_id=f"{os.getpid()}:b", ttl=600
    )
    assert acquired is False
