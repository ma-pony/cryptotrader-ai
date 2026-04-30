"""Portfolio manager with SQLAlchemy persistence and in-memory fallback."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from cryptotrader._compat import UTC
from cryptotrader.db import get_async_session, get_engine
from cryptotrader.pair import market_type_for as _market_type_for

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
        # Spec 013 deep-review correctness FINDING-2: bumped from VARCHAR(20)
        # to VARCHAR(50) — futures delivery symbols like "1000PEPE/USDT:USDT-241227"
        # are 25 chars, "BTC/USDT:USDT-241227" is 21 chars. The old 20-char limit
        # was inherited from spot-only days and would reject these inserts.
        pair = Column(String(50), nullable=False)
        # spec 013: market_type is derived from pair via Pair.parse(pair).market_type
        # Default 'spot' keeps legacy rows backward-compatible.
        market_type = Column(String(20), nullable=False, default="spot")
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
    """Create portfolio schema on first call per database URL.

    Mirrors the pattern in journal/store.py: ``create_all`` only adds new
    tables, not new columns. spec-013 added ``portfolios.market_type``
    after some deployments already had the table — backfill it here.
    """
    from sqlalchemy import text

    if database_url not in _pm_table_ready:
        Base, _, _, _ = _pm_models()
        engine = await get_engine(database_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            dialect = conn.dialect.name
            if dialect == "postgresql":
                # Idempotent — both ADD COLUMN IF NOT EXISTS and ALTER COLUMN TYPE
                # are no-ops on already-migrated tables. Type widening (20→50) does
                # not require a table rewrite and holds ACCESS EXCLUSIVE briefly.
                await conn.execute(
                    text(
                        "ALTER TABLE portfolios "
                        "ADD COLUMN IF NOT EXISTS market_type VARCHAR(20) NOT NULL DEFAULT 'spot'"
                    )
                )
                await conn.execute(text("ALTER TABLE portfolios ALTER COLUMN pair TYPE VARCHAR(50)"))
            elif dialect == "sqlite":
                result = await conn.execute(text("PRAGMA table_info(portfolios)"))
                existing = {row[1] for row in result.fetchall()}
                if "market_type" not in existing:
                    await conn.execute(
                        text("ALTER TABLE portfolios ADD COLUMN market_type VARCHAR(20) NOT NULL DEFAULT 'spot'")
                    )
                # SQLite ignores VARCHAR length — pair widening is a no-op there.
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
                        positions[r.pair] = {
                            "amount": r.amount,
                            "avg_price": r.avg_price,
                            "market_type": r.market_type or "spot",
                        }
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
        market_type = _market_type_for(pair)
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
                        row.market_type = market_type
                        row.updated_at = datetime.now(UTC)
                    else:
                        session.add(
                            Portfolio(id=key, pair=pair, market_type=market_type, amount=amount, avg_price=price)
                        )
                    await session.commit()
                return
            except Exception as e:
                logger.warning("DB position update failed: %s", e)
        if account_id not in self._memory:
            self._memory[account_id] = {}
        self._memory[account_id][pair] = {"amount": amount, "avg_price": price, "market_type": market_type}

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

    async def get_daily_pnl(self, account_id: str = "default") -> float | None:
        """Return PnL in absolute units since the start of the current UTC day.

        Returns ``None`` when no snapshot exists in today's window. The previous
        "two-most-recent snapshots diff" semantics produced false circuit-breaker
        trips after long snapshot gaps — see 2026-04-29 10:28 cycle that read a
        stale 04-28 snapshot diff (-$375) and reported -9.42% on a ~$3,980
        cash-only equity reading.

        - 0 snaps in today's window  → ``None``  (caller treats as "unknown")
        - 1 snap                     → ``0.0``   (baseline; no movement yet today)
        - 2+ snaps                   → ``latest - earliest_in_window``
        """
        snaps = await self.load_snapshots(account_id)
        if not snaps:
            return None

        midnight = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        today: list[float] = []
        for s in snaps:
            ts = s.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    continue
            if ts is None:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts >= midnight:
                today.append(s["total_value"])

        if not today:
            return None
        if len(today) == 1:
            return 0.0
        return today[-1] - today[0]

    async def get_drawdown(self, account_id: str = "default") -> float:
        snaps = [s["total_value"] for s in await self.load_snapshots(account_id)]
        if not snaps:
            return 0.0
        peak = max(snaps)
        return (snaps[-1] - peak) / peak if peak > 0 else 0.0

    async def get_returns(self, account_id: str = "default", days: int = 60) -> list[float]:
        snaps = [s["total_value"] for s in await self.load_snapshots(account_id)]
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

    async def load_snapshots(self, account_id: str) -> list[dict]:
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
    from cryptotrader.state import get_pair

    try:
        pair = get_pair(state).canonical()
    except (KeyError, TypeError, ValueError):
        pair = "BTC/USDT"
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
