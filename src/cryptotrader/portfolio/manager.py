"""Portfolio manager with SQLAlchemy persistence and in-memory fallback."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

_pm_engine = None
_pm_sessionmaker = None
_pm_table_ready = False


def _pm_models():
    from sqlalchemy import Column, DateTime, Float, String
    from sqlalchemy.orm import DeclarativeBase

    class Base(DeclarativeBase):
        pass

    class Portfolio(Base):
        __tablename__ = "portfolios"
        id = Column(String, primary_key=True)
        pair = Column(String(20), nullable=False)
        amount = Column(Float, default=0.0)
        avg_price = Column(Float, default=0.0)
        updated_at = Column(DateTime, default=lambda: datetime.now(UTC))

    class PortfolioSnapshot(Base):
        __tablename__ = "portfolio_snapshots"
        id = Column(String, primary_key=True)
        account_id = Column(String, nullable=False)
        total_value = Column(Float, default=0.0)
        timestamp = Column(DateTime, default=lambda: datetime.now(UTC), index=True)

    return Base, Portfolio, PortfolioSnapshot


async def _pm_session(database_url: str):
    global _pm_engine, _pm_sessionmaker, _pm_table_ready
    if _pm_engine is None:
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        _pm_engine = create_async_engine(database_url, pool_size=5, max_overflow=10)
        _pm_sessionmaker = async_sessionmaker(_pm_engine, expire_on_commit=False)
    if not _pm_table_ready:
        Base, _, _ = _pm_models()
        async with _pm_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _pm_table_ready = True
    return _pm_sessionmaker()


class PortfolioManager:
    def __init__(self, database_url: str | None = None):
        self._db_url = database_url
        self._memory: dict[str, dict[str, Any]] = {}
        self._snapshots: list[dict[str, Any]] = []

    async def get_portfolio(self, account_id: str = "default") -> dict[str, Any]:
        positions = {}
        if self._db_url:
            try:
                _, Portfolio, _ = _pm_models()
                from sqlalchemy import select
                async with await _pm_session(self._db_url) as session:
                    rows = await session.execute(
                        select(Portfolio).where(Portfolio.id.startswith(f"{account_id}:")))
                    for r in rows.scalars():
                        positions[r.pair] = {"amount": r.amount, "avg_price": r.avg_price}
                    total = sum(p["amount"] * p["avg_price"] for p in positions.values())
                    return {"account_id": account_id, "positions": positions, "total_value": max(total, 10000)}
            except Exception as e:
                logger.warning("DB portfolio read failed: %s", e)
        mem = self._memory.get(account_id, {})
        return {"account_id": account_id, "positions": mem, "total_value": max(
            sum(p["amount"] * p["avg_price"] for p in mem.values()), 10000)}

    async def update_position(self, account_id: str, pair: str, amount: float, price: float) -> None:
        if self._db_url:
            try:
                _, Portfolio, _ = _pm_models()
                from sqlalchemy import select
                async with await _pm_session(self._db_url) as session:
                    key = f"{account_id}:{pair}"
                    row = (await session.execute(select(Portfolio).where(Portfolio.id == key))).scalar_one_or_none()
                    if row:
                        row.amount = amount
                        row.avg_price = price
                        row.updated_at = datetime.now(UTC)
                    else:
                        session.add(Portfolio(id=key, pair=pair, amount=amount, avg_price=price))
                    await session.commit()
                return
            except Exception as e:
                logger.warning("DB position update failed: %s", e)
        if account_id not in self._memory:
            self._memory[account_id] = {}
        self._memory[account_id][pair] = {"amount": amount, "avg_price": price}

    async def get_daily_pnl(self, account_id: str = "default") -> float:
        snaps = [s for s in self._snapshots if s.get("account_id") == account_id]
        if len(snaps) < 2:
            return 0.0
        return snaps[-1]["total_value"] - snaps[-2]["total_value"]

    async def get_drawdown(self, account_id: str = "default") -> float:
        snaps = [s["total_value"] for s in self._snapshots if s.get("account_id") == account_id]
        if not snaps:
            return 0.0
        peak = max(snaps)
        return (snaps[-1] - peak) / peak if peak > 0 else 0.0

    async def get_returns(self, account_id: str = "default", days: int = 60) -> list[float]:
        snaps = [s["total_value"] for s in self._snapshots if s.get("account_id") == account_id]
        snaps = snaps[-days:]
        if len(snaps) < 2:
            return []
        return [(snaps[i] - snaps[i - 1]) / snaps[i - 1] for i in range(1, len(snaps))]

    async def snapshot(self, account_id: str = "default", total_value: float = 0.0) -> None:
        self._snapshots.append({
            "account_id": account_id, "total_value": total_value,
            "timestamp": datetime.now(UTC),
        })
