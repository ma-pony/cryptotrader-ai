"""Business attention, read state and notification outbox."""

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import SQLAlchemyError

from cryptotrader.alerts.models import AlertOut, AlertType, DeliveryOut, Resolution

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


class AlertList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[AlertOut]
    total: int


class AlertOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alerts: list[AlertOut]
    deliveries: list[DeliveryOut]


@router.get("", response_model=AlertList)
async def list_alerts(
    request: Request,
    connection_id: str | None = None,
    book_id: str | None = None,
    unread: bool = False,
    resolution: Resolution | None = None,
    type: AlertType | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    try:
        rows = await request.app.state.runtime.alert_store.list_alerts(
            connection_id=connection_id, book_id=book_id, unread=unread, resolution=resolution, type=type
        )
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="告警存储尚未就绪") from None
    return AlertList(items=rows[offset : offset + limit], total=len(rows))


@router.get("/overview", response_model=AlertOverview)
async def overview(request: Request):
    store = request.app.state.runtime.alert_store
    try:
        return AlertOverview(alerts=await store.list_alerts(), deliveries=await store.list_deliveries())
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="告警存储尚未就绪") from None


@router.post("/{alert_id}/read", response_model=AlertOut)
async def mark_read(alert_id: str, request: Request):
    try:
        return await request.app.state.runtime.alerts.mark_read(alert_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="告警不存在") from None
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="告警存储尚未就绪") from None


@router.post("/deliveries/{delivery_id}/retry", response_model=DeliveryOut)
async def retry(delivery_id: str, request: Request):
    try:
        delivery = await request.app.state.runtime.deliveries.retry(delivery_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="投递记录不存在") from None
    except ValueError:
        raise HTTPException(status_code=409, detail="仅失败投递可以重试") from None
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="告警存储尚未就绪") from None
    owner = request.app.state.runtime.alert_owner
    if owner is not None:
        owner.refresh()
    return delivery
