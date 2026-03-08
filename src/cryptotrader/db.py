"""Shared async database session factory."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# url -> (engine, sessionmaker)
_engines: dict[str, tuple] = {}
_lock = asyncio.Lock()


async def _init_engine(database_url: str) -> None:
    """Create and cache engine + sessionmaker for a URL (must be called under _lock)."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(database_url, pool_size=5, max_overflow=10)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    _engines[database_url] = (engine, sm)


async def get_async_session(database_url: str) -> AsyncSession:
    """Get an async session, creating engine on first call per URL."""
    if database_url not in _engines:
        async with _lock:
            if database_url not in _engines:  # double-checked locking
                await _init_engine(database_url)
    _, sm = _engines[database_url]
    return sm()


async def get_engine(database_url: str):
    """Get engine for a URL, initialising it if needed."""
    if database_url not in _engines:
        async with _lock:
            if database_url not in _engines:  # double-checked locking
                await _init_engine(database_url)
    engine, _ = _engines[database_url]
    return engine
