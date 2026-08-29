"""平台连接、凭据与只读连通性检查 API。"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime  # noqa: TC003 - Pydantic resolves this response field at runtime.
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from api.routes.config import (
    VenueConnectionOut,
    connection_out,
    ensure_expected_revision,
    replace_document,
    require_runtime,
    server_credential_ref,
)
from cryptotrader.runtime_config.repository import CredentialNotConfigured, RevisionConflict
from cryptotrader.runtime_config.secrets import (  # noqa: TC001 - Pydantic resolves this request field at runtime.
    CredentialPayload,
)
from cryptotrader.venues.models import (
    ConnectionEnvironment,
    MarginMode,
    VenueCapabilities,
    VenueConnection,
)

router = APIRouter(prefix="/api/venue-connections", tags=["venue-connections"])


class CreateConnectionIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    expected_revision: int
    id: str
    label: str
    adapter_id: str
    environment: ConnectionEnvironment
    enabled: bool
    leverage: int
    margin_mode: MarginMode
    parameters: dict[str, Any]


class UpdateConnectionIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    expected_revision: int
    label: str
    adapter_id: str
    environment: ConnectionEnvironment
    enabled: bool
    leverage: int
    margin_mode: MarginMode
    parameters: dict[str, Any]


class ConnectionMutationOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int
    connection: VenueConnectionOut


class PutCredentialsIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    expected_revision: int
    credentials: CredentialPayload


class CredentialStateOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configured: bool
    updated_at: datetime | None


class CredentialMutationOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int
    credential: CredentialStateOut


class VenueCapabilitiesOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_types: list[str]
    native_protection: bool
    hedge_mode: bool
    reduce_only: bool
    supported_order_types: list[str]


class ConnectionHealthOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    healthy: bool
    environment: ConnectionEnvironment
    capabilities: VenueCapabilitiesOut
    credential_configured: bool


def _connection_from_create(body: CreateConnectionIn) -> VenueConnection:
    return _build_connection(body.id, body, server_credential_ref(body.id, body.environment))


def _build_connection(
    connection_id: str,
    body: CreateConnectionIn | UpdateConnectionIn,
    credential_ref: str | None,
) -> VenueConnection:
    try:
        return VenueConnection(
            id=connection_id,
            label=body.label,
            adapter_id=body.adapter_id,
            environment=body.environment,
            enabled=body.enabled,
            credential_ref=credential_ref,
            leverage=body.leverage,
            margin_mode=body.margin_mode,
            parameters=body.parameters,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Venue connection is invalid") from error


def _document_with_connections(snapshot, connections: tuple[VenueConnection, ...]):
    execution = snapshot.document.execution.model_copy(update={"connections": connections})
    return snapshot.document.model_copy(update={"execution": execution})


def _find_connection(snapshot, connection_id: str) -> VenueConnection:
    connection = next(
        (item for item in snapshot.document.execution.connections if item.id == connection_id),
        None,
    )
    if connection is None:
        raise HTTPException(status_code=404, detail="Venue connection not found")
    return connection


def _is_referenced_by_enabled_book(snapshot, connection_id: str) -> bool:
    return any(
        book.enabled
        and any(allocation.enabled and allocation.connection_id == connection_id for allocation in book.allocations)
        for book in snapshot.document.execution.books
    )


@router.post("", response_model=ConnectionMutationOut, status_code=status.HTTP_201_CREATED)
async def create_connection(body: CreateConnectionIn, request: Request) -> ConnectionMutationOut:
    runtime = require_runtime(request)
    snapshot = await runtime.repository.get_or_create()
    ensure_expected_revision(snapshot, body.expected_revision)
    if any(item.id == body.id for item in snapshot.document.execution.connections):
        raise HTTPException(status_code=422, detail="Venue connection already exists")
    connection = _connection_from_create(body)
    document = _document_with_connections(snapshot, (*snapshot.document.execution.connections, connection))
    saved = await replace_document(runtime, snapshot, body.expected_revision, document)
    return ConnectionMutationOut(
        revision=saved.revision,
        connection=await connection_out(runtime.repository, connection),
    )


@router.put("/{connection_id}", response_model=ConnectionMutationOut)
async def update_connection(
    connection_id: str,
    body: UpdateConnectionIn,
    request: Request,
) -> ConnectionMutationOut:
    runtime = require_runtime(request)
    snapshot = await runtime.repository.get_or_create()
    ensure_expected_revision(snapshot, body.expected_revision)
    current = _find_connection(snapshot, connection_id)
    if body.environment != current.environment:
        raise HTTPException(status_code=422, detail="Connection environment cannot be changed")
    if not body.enabled and _is_referenced_by_enabled_book(snapshot, connection_id):
        raise HTTPException(status_code=422, detail="Enabled book still references this connection")
    replacement = _build_connection(connection_id, body, current.credential_ref)
    connections = tuple(
        replacement if item.id == connection_id else item for item in snapshot.document.execution.connections
    )
    document = _document_with_connections(snapshot, connections)
    saved = await replace_document(runtime, snapshot, body.expected_revision, document)
    return ConnectionMutationOut(
        revision=saved.revision,
        connection=await connection_out(runtime.repository, replacement),
    )


@router.put("/{connection_id}/credentials", response_model=CredentialMutationOut)
async def put_credentials(
    connection_id: str,
    body: PutCredentialsIn,
    request: Request,
) -> CredentialMutationOut:
    runtime = require_runtime(request)
    snapshot = await runtime.repository.get_or_create()
    ensure_expected_revision(snapshot, body.expected_revision)
    connection = _find_connection(snapshot, connection_id)
    if connection.environment == "paper" or connection.credential_ref is None:
        raise HTTPException(status_code=422, detail="Connection does not accept credentials")
    try:
        saved = await runtime.repository.put_credentials(
            body.expected_revision,
            connection.credential_ref,
            body.credentials,
        )
    except RevisionConflict as error:
        raise HTTPException(status_code=409, detail="Runtime configuration changed; reload and retry") from error
    except Exception:
        raise HTTPException(status_code=503, detail="Credential storage is unavailable") from None
    state = await runtime.repository.credential_state(connection.credential_ref)
    return CredentialMutationOut(
        revision=saved.revision,
        credential=CredentialStateOut(
            configured=state.configured,
            updated_at=state.updated_at,
        ),
    )


def _capabilities_out(capabilities: VenueCapabilities) -> VenueCapabilitiesOut:
    return VenueCapabilitiesOut(
        market_types=sorted(capabilities.market_types),
        native_protection=capabilities.native_protection,
        hedge_mode=capabilities.hedge_mode,
        reduce_only=capabilities.reduce_only,
        supported_order_types=sorted(capabilities.supported_order_types),
    )


async def _close_session(session) -> None:
    close_task = asyncio.create_task(session.close())
    try:
        await asyncio.shield(close_task)
    except asyncio.CancelledError as cancellation:
        current = asyncio.current_task()
        if current is not None:
            while current.cancelling():
                current.uncancel()
        with suppress(Exception):
            await asyncio.shield(close_task)
        raise cancellation


@router.post("/{connection_id}/test", response_model=ConnectionHealthOut)
async def test_connection(connection_id: str, request: Request) -> ConnectionHealthOut:
    runtime = require_runtime(request)
    snapshot = await runtime.repository.get_or_create()
    connection = _find_connection(snapshot, connection_id)
    credential_state = None
    session = None
    try:
        credentials = None
        if connection.credential_ref is not None:
            credential_state = await runtime.repository.credential_state(connection.credential_ref)
            if not credential_state.configured:
                raise CredentialNotConfigured(connection.credential_ref)
            credentials = await runtime.repository.reveal_credentials(connection.credential_ref)
        adapter = runtime.venue_registry.require(connection.adapter_id)
        session = await adapter.connect(connection, credentials)
        capabilities = session.capabilities
    except CredentialNotConfigured:
        raise HTTPException(status_code=503, detail="Connection credentials are not configured") from None
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Venue connection test failed") from None
    finally:
        if session is not None:
            try:
                await _close_session(session)
            except Exception:
                raise HTTPException(status_code=502, detail="Venue connection test failed") from None
    return ConnectionHealthOut(
        connection_id=connection.id,
        healthy=True,
        environment=connection.environment,
        capabilities=_capabilities_out(capabilities),
        credential_configured=bool(credential_state and credential_state.configured),
    )
