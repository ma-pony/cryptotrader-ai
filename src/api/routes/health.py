"""Health and metrics endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/metrics")
async def metrics():
    return {"decisions_total": 0, "uptime_seconds": 0}
