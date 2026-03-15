"""Portfolio manager with SQLAlchemy persistence and in-memory fallback."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from cryptotrader.db import get_async_session, get_engine

if TYPE_CHECKING:
    from cryptotrader.state import ArenaState

logger = logging.getLogger(__name__)

# Track which URLs have had their schema initialised
_pm_table_ready: set[str] = set()

# Singleton cache for SQLAlchemy model classes
_pm_cache: tuple | None = None


def _pm_models():
    global _pm_cache
    if _pm_cache is not None:
        return _pm_cache

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
        updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    class PortfolioSnapshot(Base):
        __tablename__ = "portfolio_snapshots"
        id = Column(String, primary_key=True)
        account_id = Column(String, nullable=False)
        total_value = Column(Float, default=0.0)
        cash = Column(Float, default=0.0)
        timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)

    class AccountBalance(Base):
        __tablename__ = "portfolio_accounts"
        id = Column(String, primary_key=True)  # account_id
        cash = Column(Float, default=0.0)
        updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    _pm_cache = (Base, Portfolio, PortfolioSnapshot, AccountBalance)
    return _pm_cache


async def _pm_ensure_tables(database_url: str) -> None:
    """Create portfolio schema on first call per database URL."""
    if database_url not in _pm_table_ready:
        Base, _, _, _ = _pm_models()
        engine = await get_engine(database_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _pm_table_ready.add(database_url)


async def _pm_session(database_url: str):
    # get_async_session initialises the engine on first call, then we ensure tables exist.
    session = await get_async_session(database_url)
    await _pm_ensure_tables(database_url)
    return session


class PortfolioManager:
    def __init__(self, database_url: str | None = None):
        self._db_url = database_url
        self._memory: dict[str, dict[str, Any]] = {}
        self._memory_cash: dict[str, float] = {}
        self._snapshots: list[dict[str, Any]] = []

    async def get_portfolio(self, account_id: str = "default") -> dict[str, Any]:
        """Return portfolio with cash, positions, and total_value (cash + position cost basis)."""
        positions = {}
        cash = 0.0
        if self._db_url:
            try:
                _, Portfolio, _, AccountBalance = _pm_models()
                from sqlalchemy import select

                async with await _pm_session(self._db_url) as session:
                    rows = await session.execute(select(Portfolio).where(Portfolio.id.startswith(f"{account_id}:")))
                    for r in rows.scalars():
                        positions[r.pair] = {"amount": r.amount, "avg_price": r.avg_price}
                    # Load cash
                    acct = (
                        await session.execute(select(AccountBalance).where(AccountBalance.id == account_id))
                    ).scalar_one_or_none()
                    if acct:
                        cash = acct.cash
                    pos_value = sum(p["amount"] * p["avg_price"] for p in positions.values())
                    return {
                        "account_id": account_id,
                        "positions": positions,
                        "cash": cash,
                        "total_value": cash + pos_value,
                    }
            except Exception as e:
                logger.warning("DB portfolio read failed: %s", e)
        mem = self._memory.get(account_id, {})
        cash = self._memory_cash.get(account_id, 0.0)
        pos_value = sum(p["amount"] * p["avg_price"] for p in mem.values())
        return {
            "account_id": account_id,
            "positions": mem,
            "cash": cash,
            "total_value": cash + pos_value,
        }

    async def update_position(self, account_id: str, pair: str, amount: float, price: float) -> None:
        if self._db_url:
            try:
                _, Portfolio, _, _ = _pm_models()
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

    async def update_cash(self, account_id: str = "default", cash: float = 0.0) -> None:
        """Persist current cash balance."""
        if self._db_url:
            try:
                _, _, _, AccountBalance = _pm_models()
                from sqlalchemy import select

                async with await _pm_session(self._db_url) as session:
                    row = (
                        await session.execute(select(AccountBalance).where(AccountBalance.id == account_id))
                    ).scalar_one_or_none()
                    if row:
                        row.cash = cash
                        row.updated_at = datetime.now(UTC)
                    else:
                        session.add(AccountBalance(id=account_id, cash=cash))
                    await session.commit()
                return
            except Exception as e:
                logger.warning("DB cash update failed: %s", e)
        self._memory_cash[account_id] = cash

    async def get_daily_pnl(self, account_id: str = "default") -> float:
        snaps = await self._load_snapshots(account_id)
        if len(snaps) < 2:
            return 0.0
        return snaps[-1]["total_value"] - snaps[-2]["total_value"]

    async def get_drawdown(self, account_id: str = "default") -> float:
        snaps = [s["total_value"] for s in await self._load_snapshots(account_id)]
        if not snaps:
            return 0.0
        peak = max(snaps)
        return (snaps[-1] - peak) / peak if peak > 0 else 0.0

    async def get_returns(self, account_id: str = "default", days: int = 60) -> list[float]:
        snaps = [s["total_value"] for s in await self._load_snapshots(account_id)]
        snaps = snaps[-days:]
        if len(snaps) < 2:
            return []
        return [(snaps[i] - snaps[i - 1]) / snaps[i - 1] for i in range(1, len(snaps)) if snaps[i - 1] > 0]

    async def snapshot(
        self,
        account_id: str = "default",
        total_value: float = 0.0,
        cash: float = 0.0,
    ) -> None:
        snap = {
            "account_id": account_id,
            "total_value": total_value,
            "cash": cash,
            "timestamp": datetime.now(UTC),
        }
        self._snapshots.append(snap)
        if self._db_url:
            try:
                _, _, PortfolioSnapshot, _ = _pm_models()

                async with await _pm_session(self._db_url) as session:
                    session.add(
                        PortfolioSnapshot(
                            id=str(uuid.uuid4()),
                            account_id=account_id,
                            total_value=total_value,
                            cash=cash,
                        )
                    )
                    await session.commit()
            except Exception as e:
                logger.warning("DB snapshot write failed: %s", e)

    async def _load_snapshots(self, account_id: str) -> list[dict]:
        """Load snapshots from DB if available, else use in-memory."""
        if self._db_url:
            try:
                _, _, PortfolioSnapshot, _ = _pm_models()
                from sqlalchemy import select

                async with await _pm_session(self._db_url) as session:
                    rows = await session.execute(
                        select(PortfolioSnapshot)
                        .where(PortfolioSnapshot.account_id == account_id)
                        .order_by(PortfolioSnapshot.timestamp)
                    )
                    return [
                        {
                            "account_id": r.account_id,
                            "total_value": r.total_value,
                            "cash": r.cash if r.cash is not None else 0.0,
                            "timestamp": r.timestamp,
                        }
                        for r in rows.scalars()
                    ]
            except Exception as e:
                logger.warning("DB snapshot read failed: %s", e)
        return [s for s in self._snapshots if s.get("account_id") == account_id]

    async def reset(self, account_id: str = "default") -> None:
        """Reset portfolio to clean state — delete positions, cash, and snapshots."""
        if self._db_url:
            try:
                _, Portfolio, PortfolioSnapshot, AccountBalance = _pm_models()
                from sqlalchemy import delete

                async with await _pm_session(self._db_url) as session:
                    await session.execute(delete(Portfolio).where(Portfolio.id.startswith(f"{account_id}:")))
                    await session.execute(delete(AccountBalance).where(AccountBalance.id == account_id))
                    await session.execute(delete(PortfolioSnapshot).where(PortfolioSnapshot.account_id == account_id))
                    await session.commit()
                    logger.info("Portfolio reset for account %s", account_id)
                return
            except Exception as e:
                logger.warning("DB portfolio reset failed: %s", e)
        self._memory.pop(account_id, None)
        self._memory_cash.pop(account_id, None)
        self._snapshots = [s for s in self._snapshots if s.get("account_id") != account_id]


async def read_portfolio_from_exchange(state: ArenaState) -> dict[str, Any] | None:
    """Read current portfolio directly from exchange.

    Lifted from nodes/execution.py to break the same-layer dependency
    between nodes/verdict.py and nodes/execution.py.  The function is
    defined here (portfolio layer) and re-exported from nodes/execution.py
    for backward compatibility.

    Returns None on failure.  Includes positions with avg_price and
    unrealized PnL.
    """
    # Lazy import to avoid a module-level circular dependency:
    # portfolio.manager -> nodes.execution -> portfolio.manager (PortfolioManager).
    # All cross-layer imports inside nodes.execution are already lazy, so this
    # late binding is safe and consistent with the existing pattern.
    from cryptotrader.nodes.execution import _get_exchange

    pair = state["metadata"].get("pair", "BTC/USDT")
    current_price = state["data"].get("snapshot_summary", {}).get("price", 0)

    try:
        exchange, _ = await _get_exchange(state, pair)
        balances = await exchange.get_balance()

        # Get positions with avg_price and unrealized PnL
        current_prices = {pair: current_price} if current_price else {}
        try:
            # PaperExchange accepts current_prices; LiveExchange does not
            positions = await exchange.get_positions(current_prices=current_prices)
        except TypeError:
            positions = await exchange.get_positions()
    except Exception:
        logger.debug("Failed to read portfolio from exchange", exc_info=True)
        return None

    cash = balances.get("USDT", 0.0)

    # Calculate total position value
    total_pos_value = 0.0
    for p_pair, pos in positions.items():
        amount = pos.get("amount", 0)
        price = current_price if p_pair == pair else 0.0
        total_pos_value += abs(amount) * price

    return {
        "cash": cash,
        "positions": positions,
        "total_value": cash + total_pos_value,
    }
