"""Read-only readiness and the persisted automatic-run switch."""
# ruff: noqa: RUF001 - Chinese public messages use Chinese punctuation.

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from api.routes.config import RuntimeConfigOut, config_out
from cryptotrader.decision.readiness import ReadinessOut
from cryptotrader.runtime_config.repository import RevisionConflict

router = APIRouter(prefix="/api/runtime", tags=["runtime"])


class AutomationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    expected_revision: int = Field(ge=0)


@router.get("/status", response_model=ReadinessOut)
async def get_status(request: Request):
    return await request.app.state.runtime.run_service.readiness()


@router.put("/automation", response_model=RuntimeConfigOut)
async def set_automation(body: AutomationIn, request: Request):
    runtime = request.app.state.runtime
    try:
        snapshot = await runtime.run_service.set_automation(body.enabled, body.expected_revision)
    except (ValueError, RevisionConflict):
        raise HTTPException(409, "配置已变化，请刷新后重试。") from None
    except RuntimeError:
        raise HTTPException(503, "配置暂时无法应用，请查看就绪状态后重试。") from None
    return await config_out(runtime.repository, snapshot)
