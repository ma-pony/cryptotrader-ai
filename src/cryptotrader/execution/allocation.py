"""Deterministic execution-book target allocation."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from cryptotrader.decision.models import TargetPosition
from cryptotrader.execution.models import ConnectionTarget, ExecutionBook
from cryptotrader.portfolio.models import BookPortfolioSnapshot

if TYPE_CHECKING:
    from cryptotrader.execution.models import ConnectionAllocation


class WeightedAllocationPolicy:
    """Apply configured weights without renormalization or redistribution."""

    id = "weighted"

    def allocate(
        self,
        target: TargetPosition,
        book: ExecutionBook,
        portfolio: BookPortfolioSnapshot,
    ) -> tuple[ConnectionTarget, ...]:
        if not isinstance(target, TargetPosition):
            raise ValueError("target must be a TargetPosition")
        return self.allocate_exposure(Decimal(str(target.signed_ratio)), book, portfolio)

    def allocate_exposure(
        self,
        target_exposure: Decimal,
        book: ExecutionBook,
        portfolio: BookPortfolioSnapshot,
    ) -> tuple[ConnectionTarget, ...]:
        """Allocate an already-validated Decimal book exposure without a float round-trip."""
        if (
            not isinstance(target_exposure, Decimal)
            or not target_exposure.is_finite()
            or not Decimal("-1") <= target_exposure <= Decimal("1")
        ):
            raise ValueError("target_exposure must be a finite Decimal in [-1, 1]")
        if not isinstance(book, ExecutionBook):
            raise ValueError("book must be an ExecutionBook")
        if not isinstance(portfolio, BookPortfolioSnapshot):
            raise ValueError("portfolio must be a BookPortfolioSnapshot")
        if portfolio.book_id != book.id:
            raise ValueError("portfolio book_id must match the execution book")
        if portfolio.capital_scope != book.capital_scope:
            raise ValueError("portfolio capital_scope must match the execution book")

        enabled_allocations = tuple(allocation for allocation in book.allocations if allocation.enabled)
        expected_connections = tuple(allocation.connection_id for allocation in enabled_allocations)
        actual_connections = tuple(connection.connection_id for connection in portfolio.connections)
        if actual_connections != expected_connections:
            raise ValueError("portfolio connections must match enabled book allocations in configured order")

        return tuple(
            self._connection_target(book, portfolio, allocation, target_exposure) for allocation in enabled_allocations
        )

    @staticmethod
    def _connection_target(
        book: ExecutionBook,
        portfolio: BookPortfolioSnapshot,
        allocation: ConnectionAllocation,
        target_exposure: Decimal,
    ) -> ConnectionTarget:
        weight = Decimal(str(allocation.weight))
        if portfolio.total_equity is None and target_exposure != 0:
            raise ValueError("unknown equity only permits a flat target")
        target_signed_notional = (
            Decimal("0") if target_exposure == 0 else portfolio.total_equity * target_exposure * weight
        )
        return ConnectionTarget(
            book_id=book.id,
            connection_id=allocation.connection_id,
            weight=weight,
            book_equity=portfolio.total_equity,
            target_exposure=target_exposure,
            target_signed_notional=target_signed_notional,
        )
