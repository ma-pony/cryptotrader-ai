"""Concurrent execution of one already-approved book proposal."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING

from cryptotrader.execution.models import (
    BookExecutionProposal,
    BookExecutionResult,
    ConnectionExecutionResult,
)
from cryptotrader.venues.protocol import VenueOperationError

if TYPE_CHECKING:
    from cryptotrader.execution.service import VenueExecutionService


class ExecutionCoordinator:
    """Run only the plans present in one ready proposal without reallocation."""

    def __init__(self, services: Mapping[str, VenueExecutionService]) -> None:
        if not isinstance(services, Mapping):
            raise TypeError("services must be a mapping")
        normalized = dict(services)
        for connection_id, service in normalized.items():
            if type(connection_id) is not str or not connection_id.strip():
                raise ValueError("service connection IDs must be non-empty strings")
            if getattr(service, "connection_id", None) != connection_id or not callable(
                getattr(service, "execute", None)
            ):
                raise ValueError("each service must match its connection ID and expose execute")
        self._services = normalized

    async def execute(self, proposal: BookExecutionProposal) -> BookExecutionResult:
        if not isinstance(proposal, BookExecutionProposal):
            raise TypeError("proposal must be a BookExecutionProposal")
        if not proposal.ready:
            raise ValueError("proposal must be ready before execution")

        services = []
        for plan in proposal.connection_plans:
            try:
                services.append(self._services[plan.connection_id])
            except KeyError:
                raise KeyError(f"missing execution service for connection {plan.connection_id}") from None

        outcomes = await asyncio.gather(
            *(service.execute(plan) for service, plan in zip(services, proposal.connection_plans, strict=True)),
            return_exceptions=True,
        )
        results: list[ConnectionExecutionResult] = []
        for plan, outcome in zip(proposal.connection_plans, outcomes, strict=True):
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, VenueOperationError):
                results.append(
                    ConnectionExecutionResult.failed(
                        plan,
                        "execute",
                        requires_attention=True,
                        trace=("execute",),
                        execution_quote=plan.quote,
                    )
                )
                continue
            if isinstance(outcome, BaseException):
                raise outcome
            if not isinstance(outcome, ConnectionExecutionResult):
                raise TypeError("venue execution service must return ConnectionExecutionResult")
            results.append(outcome)

        result_tuple = tuple(results)
        status = BookExecutionResult.expected_status(proposal, result_tuple)
        return BookExecutionResult(
            proposal,
            result_tuple,
            status,
            any(result.requires_attention for result in result_tuple),
            False,
        )
