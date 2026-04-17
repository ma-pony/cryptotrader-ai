"""Portfolio and risk status API routes."""

from __future__ import annotations

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


@router.get("/portfolio", response_model=PortfolioOut)
async def get_portfolio():
    from cryptotrader.config import load_config
    from cryptotrader.portfolio.manager import PortfolioManager

    config = load_config()

    pm = PortfolioManager(config.infrastructure.database_url)
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
