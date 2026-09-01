"""Concurrent, fail-closed execution-book portfolio reads."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING

from cryptotrader.accounts.models import AccountSnapshot
from cryptotrader.portfolio.models import BookPortfolioSnapshot, ConnectionPortfolioSnapshot
from cryptotrader.venues.models import ConnectionPosition

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cryptotrader.execution.models import ExecutionBook
    from cryptotrader.pair import Pair
    from cryptotrader.venues.protocol import VenueSession


class PortfolioReadError(RuntimeError):
    """A requested book connection could not provide a valid portfolio."""


class PortfolioAggregator:
    """Read exactly one book's enabled connections and aggregate their values."""

    def __init__(self, account_store=None):
        self.account_store = account_store

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
            total_equity=None
            if any(c.equity is None for c in connections)
            else sum((connection.equity for connection in connections), Decimal("0")),
            total_signed_notional=sum(
                (connection.position.signed_notional for connection in connections),
                Decimal("0"),
            ),
            connections=connections,
        )

    async def _read_connection(
        self,
        connection_id: str,
        session: VenueSession,
        pair: Pair,
    ) -> ConnectionPortfolioSnapshot:
        read_failed = False
        try:
            snapshot = await session.fetch_account()
        except Exception:
            read_failed = True
        if read_failed:
            raise PortfolioReadError(f"failed to read portfolio for connection {connection_id}")
        if not isinstance(snapshot, AccountSnapshot):
            raise PortfolioReadError(f"failed to read portfolio for connection {connection_id}: invalid snapshot type")
        if snapshot.connection_id != connection_id:
            raise PortfolioReadError(
                f"failed to read portfolio for connection {connection_id}: snapshot connection_id mismatch"
            )
        if self.account_store is not None:
            await self.account_store.ingest(snapshot)
        currency = pair.settle or pair.quote
        equity = snapshot.equity.amount if snapshot.equity.currency == currency else None
        positions = [
            item
            for item in snapshot.positions
            if item.instrument.pair == pair and item.instrument.market_type == pair.market_type
        ]
        if len(positions) > 1:
            raise PortfolioReadError(f"failed to read portfolio for connection {connection_id}: ambiguous position")
        if positions:
            position = positions[0]
            if position.signed_notional.currency != currency or position.signed_notional.amount is None:
                quote = await session.fetch_quote(pair)
                notional = position.signed_amount * quote.last
            else:
                notional = position.signed_notional.amount
            projected = ConnectionPosition(pair, position.signed_amount, notional, position.entry_price)
        else:
            projected = ConnectionPosition(pair, Decimal("0"), Decimal("0"), None)
        return ConnectionPortfolioSnapshot(
            connection_id,
            equity,
            {item.currency: item.amount for item in snapshot.balances if item.amount is not None},
            projected,
            snapshot,
        )
