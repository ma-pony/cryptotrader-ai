"""Explicit account-free analysis admission."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from cryptotrader.decision.service import CapabilityUnavailableError
from cryptotrader.pair import Pair
from cryptotrader.runtime import RuntimeLeaseUnavailableError
from cryptotrader.tasks import TaskManagerClosedError, TooManyTasksError

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pair: str
    expected_revision: int = Field(ge=0)


class AnalysisQueued(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: str


@router.post("", response_model=AnalysisQueued, status_code=202)
async def start_analysis(body: AnalysisRequest, request: Request):
    try:
        pair = Pair.parse(body.pair)
    except ValueError:
        raise HTTPException(422, "Invalid pair") from None
    try:
        decision_id = await request.app.state.runtime.run_service.start_analysis(pair, body.expected_revision)
    except ValueError:
        raise HTTPException(409, "Configuration revision changed; refresh before starting analysis") from None
    except TooManyTasksError:
        raise HTTPException(429, "Too many analyses are running") from None
    except CapabilityUnavailableError:
        raise HTTPException(422, "Analysis dependencies are not ready") from None
    except (RuntimeLeaseUnavailableError, TaskManagerClosedError):
        raise HTTPException(503, "Runtime is unavailable") from None
    return AnalysisQueued(decision_id=decision_id)
