"""Fast asynchronous admission; GET reads only durable manual-operation facts."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from cryptotrader.accounts.models import AccountOperationOut
from cryptotrader.accounts.operations import AccountOperationService
from cryptotrader.accounts.store import OperationConflictError
from cryptotrader.runtime_config.repository import RevisionConflict

router = APIRouter(tags=["account-operations"])


class PrepareOperationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["cancel_orders", "flatten"]
    pair: str
    expected_revision: int = Field(ge=1)
    confirm_stop: bool


class ExecuteOperationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_version: int = Field(ge=1)


class OperationAcceptedOut(BaseModel):
    operation_id: str
    status: Literal["preparing", "executing"]


def operation_service(runtime):
    service = getattr(runtime, "account_operations", None)
    if service is None:
        service = runtime.account_operations = AccountOperationService(runtime)
    return service


def service_for(request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(503, "账户服务不可用")
    return operation_service(runtime)


def operation_error(error):
    if isinstance(error, LookupError):
        return HTTPException(404, "人工操作或账户不存在")
    if isinstance(error, PermissionError):
        return HTTPException(403, str(error))
    if isinstance(error, (RevisionConflict, OperationConflictError)):
        return HTTPException(409, "配置或操作状态已变化，请重新读取")  # noqa: RUF001
    return HTTPException(422, "请检查操作类型、品种及停用确认")


@router.post("/api/accounts/{connection_id}/operations/prepare", status_code=202, response_model=OperationAcceptedOut)
async def prepare_operation(connection_id: str, body: PrepareOperationIn, request: Request):
    try:
        operation_id = await service_for(request).prepare(
            connection_id, body.pair, body.kind, body.expected_revision, body.confirm_stop
        )
    except (LookupError, PermissionError, ValueError, RevisionConflict) as error:
        raise operation_error(error) from None
    return OperationAcceptedOut(operation_id=operation_id, status="preparing")


@router.get("/api/account-operations/{operation_id}", response_model=AccountOperationOut)
async def get_operation(operation_id: str, request: Request):
    try:
        return await service_for(request).get(operation_id)
    except LookupError as error:
        raise operation_error(error) from None


@router.post("/api/account-operations/{operation_id}/execute", status_code=202, response_model=OperationAcceptedOut)
async def execute_operation(operation_id: str, body: ExecuteOperationIn, request: Request):
    try:
        await service_for(request).execute(operation_id, body.plan_version)
    except (LookupError, PermissionError, ValueError, RevisionConflict) as error:
        raise operation_error(error) from None
    return OperationAcceptedOut(operation_id=operation_id, status="executing")
