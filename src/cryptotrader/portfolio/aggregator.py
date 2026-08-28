"""Concurrent, fail-closed execution-book portfolio reads."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING

from cryptotrader.portfolio.models import BookPortfolioSnapshot, ConnectionPortfolioSnapshot

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cryptotrader.execution.models import ExecutionBook
    from cryptotrader.pair import Pair
    from cryptotrader.venues.protocol import VenueSession


class PortfolioReadError(RuntimeError):
    """A requested book connection could not provide a valid portfolio."""


class PortfolioAggregator:
    """Read exactly one book's enabled connections and aggregate their values."""

    async def read(
        self,
        book: ExecutionBook,
        sessions: Mapping[str, VenueSession],
        pair: Pair,
    ) -> BookPortfolioSnapshot:
        enabled_allocations = tuple(allocation for allocation in book.allocations if allocation.enabled)
        ordered_sessions: list[tuple[str, VenueSession]] = []
        for allocation in enabled_allocations:
            connection_id = allocation.connection_id
            session = sessions.get(connection_id)
            if session is None:
                raise PortfolioReadError(f"failed to read portfolio for connection {connection_id}: session missing")
            if session.connection_id != connection_id:
                raise PortfolioReadError(
                    f"failed to read portfolio for connection {connection_id}: session connection_id mismatch"
                )
            ordered_sessions.append((connection_id, session))

        tasks = [
            asyncio.create_task(self._read_connection(connection_id, session, pair))
            for connection_id, session in ordered_sessions
        ]
        try:
            connections = tuple(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return BookPortfolioSnapshot(
            book_id=book.id,
            capital_scope=book.capital_scope,
            total_equity=sum((connection.equity for connection in connections), Decimal("0")),
            total_signed_notional=sum(
                (connection.position.signed_notional for connection in connections),
                Decimal("0"),
            ),
            connections=connections,
        )

    @staticmethod
    async def _read_connection(
        connection_id: str,
        session: VenueSession,
        pair: Pair,
    ) -> ConnectionPortfolioSnapshot:
        try:
            snapshot = await session.fetch_portfolio(pair)
        except Exception as exc:
            raise PortfolioReadError(f"failed to read portfolio for connection {connection_id}") from exc
        if not isinstance(snapshot, ConnectionPortfolioSnapshot):
            raise PortfolioReadError(f"failed to read portfolio for connection {connection_id}: invalid snapshot type")
        if snapshot.connection_id != connection_id:
            raise PortfolioReadError(
                f"failed to read portfolio for connection {connection_id}: snapshot connection_id mismatch"
            )
        return snapshot
