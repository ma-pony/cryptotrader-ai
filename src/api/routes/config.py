"""数据库 RuntimeConfig 的整文档 CAS 与严格脱敏响应。"""

from __future__ import annotations

import asyncio
from datetime import datetime  # noqa: TC003 - Pydantic resolves this response field at runtime.
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from cryptotrader.runtime_config.models import (
    HitlConfig,
    InfrastructureConfig,
    LlmConfig,
    MarketDataConfig,
    NotificationConfig,
    RiskConfig,
    RuntimeConfigDocument,
    SchedulerConfig,
    SignalConfig,
    SystemConfig,
    TriggerConfig,
    validate_runtime_document,
)
from cryptotrader.runtime_config.repository import CredentialState, RevisionConflict
from cryptotrader.venues.models import (  # noqa: TC001 - Pydantic resolves these fields at runtime.
    ConnectionEnvironment,
    MarginMode,
    VenueConnection,
)

router = APIRouter(prefix="/api/config", tags=["config"])


class ConnectionAllocationOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    enabled: bool
    weight: float


class ExecutionBookOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    capital_scope: str
    enabled: bool
    hitl_required: bool
    allocations: list[ConnectionAllocationOut]


class VenueConnectionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    adapter_id: str
    environment: ConnectionEnvironment
    enabled: bool
    credential_ref: str | None
    credential_configured: bool
    credential_updated_at: datetime | None
    leverage: int
    margin_mode: MarginMode
    parameters: dict[str, Any]


class ExecutionConfigOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connections: list[VenueConnectionOut]
    books: list[ExecutionBookOut]
    allocation_policy: str


class RuntimeDocumentOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: SystemConfig
    market_data: MarketDataConfig
    llm: LlmConfig
    signals: SignalConfig
    risk: RiskConfig
    execution: ExecutionConfigOut
    hitl: HitlConfig
    scheduler: SchedulerConfig
    triggers: TriggerConfig
    notifications: NotificationConfig
    infrastructure: InfrastructureConfig


class RuntimeConfigOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int
    updated_at: datetime
    setup_required: bool
    document: RuntimeDocumentOut


class PutRuntimeConfigIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    expected_revision: int
    document: RuntimeConfigDocument


def require_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or getattr(runtime, "repository", None) is None:
        raise HTTPException(status_code=503, detail="Runtime configuration is unavailable")
    return runtime


def validate_document(runtime, document: RuntimeConfigDocument) -> None:
    try:
        validate_runtime_document(
            document,
            set(runtime.signal_registry.installed_ids()),
            set(runtime.venue_registry.installed_ids()),
            set(runtime.market_registry.installed_ids()),
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Runtime configuration is invalid") from error


async def connection_out(repository, connection: VenueConnection) -> VenueConnectionOut:
    state = (
        await repository.credential_state(connection.credential_ref)
        if connection.credential_ref is not None
        else CredentialState("", False, None)
    )
    return VenueConnectionOut(
        id=connection.id,
        label=connection.label,
        adapter_id=connection.adapter_id,
        environment=connection.environment,
        enabled=connection.enabled,
        credential_ref=connection.credential_ref,
        credential_configured=state.configured,
        credential_updated_at=state.updated_at,
        leverage=connection.leverage,
        margin_mode=connection.margin_mode,
        parameters=dict(connection.parameters),
    )


async def config_out(repository, snapshot) -> RuntimeConfigOut:
    document = snapshot.document
    connections = await asyncio.gather(
        *(connection_out(repository, connection) for connection in document.execution.connections)
    )
    books = [
        ExecutionBookOut(
            id=book.id,
            label=book.label,
            capital_scope=book.capital_scope,
            enabled=book.enabled,
            hitl_required=book.hitl_required,
            allocations=[
                ConnectionAllocationOut(
                    connection_id=allocation.connection_id,
                    enabled=allocation.enabled,
                    weight=allocation.weight,
                )
                for allocation in book.allocations
            ],
        )
        for book in document.execution.books
    ]
    return RuntimeConfigOut(
        revision=snapshot.revision,
        updated_at=snapshot.updated_at,
        setup_required=snapshot.setup_required,
        document=RuntimeDocumentOut(
            system=document.system,
            market_data=document.market_data,
            llm=document.llm,
            signals=document.signals,
            risk=document.risk,
            execution=ExecutionConfigOut(
                connections=list(connections),
                books=books,
                allocation_policy=document.execution.allocation_policy,
            ),
            hitl=document.hitl,
            scheduler=document.scheduler,
            triggers=document.triggers,
            notifications=document.notifications,
            infrastructure=document.infrastructure,
        ),
    )


async def replace_document(runtime, expected_revision: int, document: RuntimeConfigDocument):
    validate_document(runtime, document)
    try:
        return await runtime.repository.replace(expected_revision, document)
    except RevisionConflict as error:
        raise HTTPException(status_code=409, detail="Runtime configuration changed; reload and retry") from error


@router.get("", response_model=RuntimeConfigOut)
async def get_config(request: Request) -> RuntimeConfigOut:
    runtime = require_runtime(request)
    snapshot = await runtime.repository.get_or_create()
    return await config_out(runtime.repository, snapshot)


@router.put("", response_model=RuntimeConfigOut)
async def put_config(body: PutRuntimeConfigIn, request: Request) -> RuntimeConfigOut:
    runtime = require_runtime(request)
    snapshot = await replace_document(runtime, body.expected_revision, body.document)
    return await config_out(runtime.repository, snapshot)
