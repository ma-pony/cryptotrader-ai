"""平台连接、凭据与只读连通性检查 API。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, SecretStr

from api.routes.config import (
    VenueConnectionOut,
    application_barrier,
    apply_document,
    connection_out,
    ensure_expected_revision,
    publish_pending_snapshot,
    require_runtime,
    server_credential_ref,
)
from cryptotrader.configuration.catalog import CredentialValidationError, validate_venue_credentials
from cryptotrader.runtime_config.models import ExecutionConfig
from cryptotrader.runtime_config.repository import CredentialNotConfigured, RevisionConflict
from cryptotrader.venues.models import (
    ConnectionEnvironment,
    MarginMode,
    VenueCapabilities,
    VenueConnection,
)
from cryptotrader.venues.protocol import VenueOperationError

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
    canary_only: bool
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
    canary_only: bool
    parameters: dict[str, Any]
    confirm_stop: bool = False


class ConnectionMutationOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int
    connection: VenueConnectionOut


class ConnectionRemovedOut(BaseModel):
    revision: int
    connection_id: str


@router.delete("/{connection_id}", response_model=ConnectionRemovedOut)
async def remove_connection(connection_id: str, request: Request, expected_revision: int = Query(ge=1)):
    runtime = require_runtime(request)
    current = await runtime.repository.get_existing()
    ensure_expected_revision(current, expected_revision)
    _find_connection(current, connection_id)
    execution = current.document.execution
    document = current.document.model_copy(
        update={
            "execution": execution.model_copy(
                update={
                    "connections": tuple(c for c in execution.connections if c.id != connection_id),
                    "books": tuple(
                        replace(b, allocations=tuple(a for a in b.allocations if a.connection_id != connection_id))
                        for b in execution.books
                    ),
                }
            )
        }
    )
    saved = await apply_document(
        runtime,
        current,
        expected_revision,
        document,
        getattr(request.app.state, "refresh_runtime_owners", None),
        getattr(request.app.state, "clear_runtime_owners", None),
    )
    return ConnectionRemovedOut(revision=saved.revision, connection_id=connection_id)


class PutCredentialsIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    expected_revision: int
    values: dict[str, SecretStr]


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
    account_reads: list[str]
    exit_operations: list[str]
    history_initial_days: int | None
    unknown_fields: list[str]


class ConnectionHealthOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    healthy: bool
    environment: ConnectionEnvironment
    capabilities: VenueCapabilitiesOut | None
    credential_configured: bool
    checked_at: datetime
    error_code: str | None


def _connection_from_create(body: CreateConnectionIn) -> VenueConnection:
    try:
        credential_ref = server_credential_ref(body.id, body.adapter_id, body.environment)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Venue connection is invalid") from None
    return _build_connection(body.id, body, credential_ref)


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
            canary_only=body.canary_only,
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
    saved = await apply_document(
        runtime,
        snapshot,
        body.expected_revision,
        document,
        getattr(request.app.state, "refresh_runtime_owners", None),
        getattr(request.app.state, "clear_runtime_owners", None),
    )
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
    if not body.enabled and _is_referenced_by_enabled_book(snapshot, connection_id) and not body.confirm_stop:
        raise HTTPException(status_code=422, detail="Enabled book still references this connection")
    replacement = _build_connection(connection_id, body, current.credential_ref)
    connections = tuple(
        replacement if item.id == connection_id else item for item in snapshot.document.execution.connections
    )
    document = _document_with_connections(snapshot, connections)
    if not body.enabled and body.confirm_stop:
        document = document.model_copy(
            update={
                "execution": document.execution.model_copy(
                    update={
                        "books": tuple(
                            replace(book, enabled=False)
                            if any(a.enabled and a.connection_id == connection_id for a in book.allocations)
                            else book
                            for book in document.execution.books
                        ),
                    }
                )
            }
        )
    saved = await apply_document(
        runtime,
        snapshot,
        body.expected_revision,
        document,
        getattr(request.app.state, "refresh_runtime_owners", None),
        getattr(request.app.state, "clear_runtime_owners", None),
    )
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
    if connection.credential_ref is None:
        raise HTTPException(status_code=422, detail="Connection does not accept credentials")
    try:
        payload = validate_venue_credentials(
            connection.adapter_id,
            connection.environment,
            {key: value.get_secret_value() for key, value in body.values.items()},
        )
    except CredentialValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors) from None
    async with application_barrier(runtime):
        try:
            saved = await runtime.repository.put_credentials(
                body.expected_revision,
                connection.credential_ref,
                payload,
            )
        except RevisionConflict as error:
            raise HTTPException(status_code=409, detail="Runtime configuration changed; reload and retry") from error
        except Exception:
            raise HTTPException(status_code=503, detail="Credential storage is unavailable") from None
        await publish_pending_snapshot(
            runtime,
            saved,
            getattr(request.app.state, "refresh_runtime_owners", None),
            clear_owners=getattr(request.app.state, "clear_runtime_owners", None),
        )
    state = await runtime.repository.credential_state(connection.credential_ref)
    return CredentialMutationOut(
        revision=saved.revision,
        credential=CredentialStateOut(
            configured=state.configured,
            updated_at=state.updated_at,
        ),
    )


@router.delete("/{connection_id}/credentials", response_model=CredentialMutationOut)
async def delete_credentials(
    connection_id: str,
    request: Request,
    expected_revision: int = Query(),
) -> CredentialMutationOut:
    runtime = require_runtime(request)
    snapshot = await runtime.repository.get_or_create()
    ensure_expected_revision(snapshot, expected_revision)
    connection = _find_connection(snapshot, connection_id)
    if connection.credential_ref is None:
        raise HTTPException(status_code=422, detail="Connection does not accept credentials")
    async with application_barrier(runtime):
        try:
            saved = await runtime.repository.delete_credentials(expected_revision, connection.credential_ref)
        except RevisionConflict as error:
            raise HTTPException(status_code=409, detail="Runtime configuration changed; reload and retry") from error
        except Exception:
            raise HTTPException(status_code=503, detail="Credential storage is unavailable") from None
        await publish_pending_snapshot(
            runtime,
            saved,
            getattr(request.app.state, "refresh_runtime_owners", None),
            clear_owners=getattr(request.app.state, "clear_runtime_owners", None),
        )
    return CredentialMutationOut(
        revision=saved.revision,
        credential=CredentialStateOut(configured=False, updated_at=None),
    )


def _capabilities_out(capabilities: VenueCapabilities) -> VenueCapabilitiesOut:
    return VenueCapabilitiesOut(
        market_types=sorted(capabilities.market_types),
        native_protection=capabilities.native_protection,
        hedge_mode=capabilities.hedge_mode,
        reduce_only=capabilities.reduce_only,
        supported_order_types=sorted(capabilities.supported_order_types),
        account_reads=sorted(capabilities.account_reads),
        exit_operations=sorted(capabilities.exit_operations),
        history_initial_days=capabilities.history_initial_days,
        unknown_fields=sorted(capabilities.unknown_fields),
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


def _check_fingerprint(connection: VenueConnection, credential_state) -> str:
    payload = {
        "connection": ExecutionConfig(connections=(connection,)).model_dump(mode="json")["connections"][0],
        "credential_updated_at": credential_state.updated_at.isoformat()
        if credential_state and credential_state.updated_at
        else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@router.get("/{connection_id}/check", response_model=ConnectionHealthOut | None)
async def get_connection_check(connection_id: str, request: Request) -> ConnectionHealthOut | None:
    runtime = require_runtime(request)
    snapshot = await runtime.repository.get_or_create()
    connection = _find_connection(snapshot, connection_id)
    credential_state = (
        await runtime.repository.credential_state(connection.credential_ref) if connection.credential_ref else None
    )
    saved = await runtime.repository.connection_check(connection_id, _check_fingerprint(connection, credential_state))
    return ConnectionHealthOut.model_validate(saved) if saved is not None else None


@router.post("/{connection_id}/test", response_model=ConnectionHealthOut)
async def test_connection(connection_id: str, request: Request) -> ConnectionHealthOut:
    runtime = require_runtime(request)
    snapshot = await runtime.repository.get_or_create()
    connection = _find_connection(snapshot, connection_id)
    credential_state = (
        await runtime.repository.credential_state(connection.credential_ref) if connection.credential_ref else None
    )
    fingerprint = _check_fingerprint(connection, credential_state)
    session = None
    capabilities = None
    error_code = None
    try:
        credentials = None
        if connection.credential_ref is not None:
            if not credential_state or not credential_state.configured:
                raise CredentialNotConfigured(connection.credential_ref)
            credentials = await runtime.repository.reveal_credentials(connection.credential_ref)
        adapter = runtime.venue_registry.require(connection.adapter_id)
        session = await adapter.connect(connection, credentials)
        capabilities = session.capabilities
        await session.check_connection()
    except CredentialNotConfigured:
        error_code = "credentials_missing"
    except VenueOperationError as error:
        error_code = "authentication_failed" if error.code == "authentication_failed" else "account_unavailable"
    except Exception:
        error_code = "account_unavailable"
    finally:
        if session is not None:
            try:
                await _close_session(session)
            except Exception:
                error_code = "account_unavailable"
    result = ConnectionHealthOut(
        connection_id=connection.id,
        healthy=error_code is None,
        environment=connection.environment,
        capabilities=_capabilities_out(capabilities) if capabilities is not None else None,
        credential_configured=bool(credential_state and credential_state.configured),
        checked_at=datetime.now(UTC),
        error_code=error_code,
    )
    await runtime.repository.save_connection_check(connection_id, fingerprint, result.model_dump(mode="json"))
    return result
