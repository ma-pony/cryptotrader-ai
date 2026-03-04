"""Portfolio and risk status API routes."""

from __future__ import annotations

import os

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["portfolio"])


class PositionOut(BaseModel):
    pair: str
    amount: float
    avg_price: float
    value: float


class PortfolioOut(BaseModel):
    total_value: float
    daily_pnl: float
    drawdown: float
    positions: list[PositionOut]


class RiskStatusOut(BaseModel):
    hourly_trades: int
    daily_trades: int
    circuit_breaker_active: bool


@router.get("/portfolio", response_model=PortfolioOut)
async def get_portfolio():
    from cryptotrader.portfolio.manager import PortfolioManager

    pm = PortfolioManager(os.environ.get("DATABASE_URL"))
    portfolio = await pm.get_portfolio()
    daily_pnl = await pm.get_daily_pnl()
    drawdown = await pm.get_drawdown()
    positions = [
        PositionOut(
            pair=pair,
            amount=pos["amount"],
            avg_price=pos["avg_price"],
            value=pos["amount"] * pos["avg_price"],
        )
        for pair, pos in portfolio.get("positions", {}).items()
    ]
    return PortfolioOut(
        total_value=portfolio.get("total_value", 0),
        daily_pnl=daily_pnl,
        drawdown=drawdown,
        positions=positions,
    )


@router.get("/risk/status", response_model=RiskStatusOut)
async def get_risk_status():
    from cryptotrader.risk.state import RedisStateManager

    rsm = RedisStateManager(os.environ.get("REDIS_URL"))
    if rsm.available:
        hourly, daily = await rsm.get_trade_counts()
        cb = await rsm.is_circuit_breaker_active()
    else:
        hourly, daily, cb = 0, 0, False
    return RiskStatusOut(
        hourly_trades=hourly,
        daily_trades=daily,
        circuit_breaker_active=cb,
    )
