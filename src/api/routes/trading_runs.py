"""Explicit confirmation of the global execution scope."""
# ruff: noqa: RUF001 - Chinese public messages use Chinese punctuation.

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from api.routes.analyses import AnalysisQueued
from cryptotrader.decision.readiness import TradingScopeOut
from cryptotrader.pair import Pair
from cryptotrader.runtime import RuntimeLeaseUnavailableError
from cryptotrader.tasks import TaskManagerClosedError, TooManyTasksError

router = APIRouter(prefix="/api/trading-runs", tags=["trading"])


class TradingRunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pair: str
    expected_revision: int = Field(ge=0)
    confirmed_book_ids: tuple[str, ...]


@router.get("/scope", response_model=TradingScopeOut)
async def trading_scope(pair: str, request: Request):
    try:
        parsed = Pair.parse(pair)
    except ValueError:
        raise HTTPException(422, "品种格式无效。") from None
    return await request.app.state.runtime.run_service.trading_scope(parsed)


@router.post("", response_model=AnalysisQueued, status_code=202)
async def start_trading(body: TradingRunIn, request: Request):
    try:
        pair = Pair.parse(body.pair)
    except ValueError:
        raise HTTPException(422, "品种格式无效。") from None
    try:
        decision_id = await request.app.state.runtime.run_service.start_trading(
            pair, body.expected_revision, body.confirmed_book_ids
        )
    except ValueError:
        raise HTTPException(409, "配置或执行范围已变化，请刷新就绪状态并重新确认所有可执行资金池。") from None
    except TooManyTasksError:
        raise HTTPException(429, "当前运行任务过多。") from None
    except (RuntimeLeaseUnavailableError, TaskManagerClosedError):
        raise HTTPException(503, "运行服务暂不可用。") from None
    return AnalysisQueued(decision_id=decision_id)
