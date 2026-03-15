"""Shared async database session factory.

Engine and sessionmaker are cached per (database_url, event_loop) to avoid
asyncpg 'another operation in progress' errors when the same URL is accessed
from different event loops (e.g. main app vs Streamlit dashboard).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

logger = logging.getLogger(__name__)

# (url, loop_id) -> (engine, sessionmaker)
_engines: dict[tuple[str, int], tuple] = {}
_lock_map: dict[int, asyncio.Lock] = {}


def _get_lock() -> asyncio.Lock:
    """Get or create a Lock for the current event loop."""
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if loop_id not in _lock_map:
        _lock_map[loop_id] = asyncio.Lock()
    return _lock_map[loop_id]


def _cache_key(database_url: str) -> tuple[str, int]:
    return (database_url, id(asyncio.get_running_loop()))


async def _init_engine(database_url: str) -> None:
    """Create and cache engine + sessionmaker for a URL on the current loop."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(database_url, pool_size=5, max_overflow=10)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    _engines[_cache_key(database_url)] = (engine, sm)


async def get_async_session(database_url: str) -> AsyncSession:
    """Get an async session, creating engine on first call per URL per loop."""
    key = _cache_key(database_url)
    if key not in _engines:
        lock = _get_lock()
        async with lock:
            if key not in _engines:
                await _init_engine(database_url)
    _, sm = _engines[key]
    return sm()


async def get_engine(database_url: str) -> AsyncEngine:
    """Get engine for a URL, initialising it if needed."""
    key = _cache_key(database_url)
    if key not in _engines:
        lock = _get_lock()
        async with lock:
            if key not in _engines:
                await _init_engine(database_url)
    engine, _ = _engines[key]
    return engine
