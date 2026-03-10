"""Decision journal storage — PostgreSQL with in-memory fallback."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

from cryptotrader.db import get_async_session, get_engine
from cryptotrader.models import DecisionCommit

logger = logging.getLogger(__name__)

# Track which URLs have had their schema initialised
_table_ready: set[str] = set()

# Singleton cache for SQLAlchemy model classes
_sa_cache: tuple | None = None


def _sa_models():
    global _sa_cache
    if _sa_cache is not None:
        return _sa_cache

    from sqlalchemy import Column, DateTime, Float, Integer, String, Text
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.orm import DeclarativeBase

    class Base(DeclarativeBase):
        pass

    class DecisionCommitRow(Base):
        __tablename__ = "decision_commits"
        hash = Column(String(16), primary_key=True)
        parent_hash = Column(String(16), nullable=True)
        timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
        pair = Column(String(20), nullable=False, index=True)
        snapshot_summary = Column(JSONB, default={})
        analyses = Column(JSONB, default={})
        debate_rounds = Column(Integer, default=0)
        challenges = Column(JSONB, default=[])
        divergence = Column(Float, default=0.0)
        verdict = Column(JSONB, nullable=True)
        risk_gate = Column(JSONB, nullable=True)
        order_data = Column(JSONB, nullable=True)
        fill_price = Column(Float, nullable=True)
        slippage = Column(Float, nullable=True)
        portfolio_after = Column(JSONB, default={})
        pnl = Column(Float, nullable=True)
        retrospective = Column(Text, nullable=True)
        trace_id = Column(String(36), nullable=True, index=True)

    _sa_cache = (Base, DecisionCommitRow)
    return _sa_cache


async def _ensure_tables(database_url: str) -> None:
    """Create journal schema on first call per database URL."""
    if database_url not in _table_ready:
        Base, _ = _sa_models()
        engine = await get_engine(database_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _table_ready.add(database_url)


async def _get_session(database_url: str):
    # get_async_session initialises the engine on first call, then we ensure tables exist.
    session = await get_async_session(database_url)
    await _ensure_tables(database_url)
    return session


class JournalStore:
    """Stores decision commits. PostgreSQL when DATABASE_URL set, else in-memory."""

    _MAX_MEMORY = 10_000  # Cap in-memory store to prevent OOM
    # Shared fallback memory keyed by db_url — only used when a DB URL is configured (but unreachable).
    # Pure in-memory mode (db_url=None) keeps a per-instance list so test isolation is maintained.
    _shared_memory: dict[str, list[dict[str, Any]]] = {}

    def __init__(self, database_url: str | None = None):
        self._db_url = database_url
        if database_url:
            # Shared fallback: all instances pointing at the same DB share one memory fallback
            if database_url not in JournalStore._shared_memory:
                JournalStore._shared_memory[database_url] = []
            self._memory: list[dict[str, Any]] = JournalStore._shared_memory[database_url]
        else:
            # Pure in-memory mode: each instance is independent (preserves test isolation)
            self._memory = []

    @property
    def _use_db(self) -> bool:
        return bool(self._db_url)

    def _serialize(self, dc: DecisionCommit) -> dict[str, Any]:
        return {
            "hash": dc.hash,
            "parent_hash": dc.parent_hash,
            "timestamp": dc.timestamp.isoformat(),
            "pair": dc.pair,
            "data": json.loads(json.dumps(asdict(dc), default=str)),
        }

    def _deserialize(self, row: dict[str, Any]) -> DecisionCommit:
        d = row["data"]
        from cryptotrader.models import AgentAnalysis, GateResult, Order, TradeVerdict

        analyses = {k: AgentAnalysis(**v) for k, v in d.get("analyses", {}).items()}
        for a in analyses.values():
            if isinstance(a.timestamp, str):
                a.timestamp = datetime.fromisoformat(a.timestamp)
        verdict = TradeVerdict(**d["verdict"]) if d.get("verdict") else None
        risk_gate = GateResult(**d["risk_gate"]) if d.get("risk_gate") else None
        order = None
        if d.get("order"):
            od = dict(d["order"])
            if "status" in od:
                from cryptotrader.models import OrderStatus

                try:
                    od["status"] = OrderStatus(od["status"])
                except (ValueError, KeyError):
                    od["status"] = OrderStatus.PENDING
            order = Order(**od)
        return DecisionCommit(
            hash=d["hash"],
            parent_hash=d.get("parent_hash"),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            pair=d["pair"],
            snapshot_summary=d.get("snapshot_summary", {}),
            analyses=analyses,
            debate_rounds=d.get("debate_rounds", 0),
            challenges=d.get("challenges", []),
            divergence=d.get("divergence", 0.0),
            verdict=verdict,
            risk_gate=risk_gate,
            order=order,
            fill_price=d.get("fill_price"),
            slippage=d.get("slippage"),
            portfolio_after=d.get("portfolio_after", {}),
            pnl=d.get("pnl"),
            retrospective=d.get("retrospective"),
            trace_id=d.get("trace_id"),
        )

    def _dc_to_row_dict(self, dc: DecisionCommit) -> dict[str, Any]:
        d = json.loads(json.dumps(asdict(dc), default=str))
        return {
            "hash": dc.hash,
            "parent_hash": dc.parent_hash,
            "timestamp": dc.timestamp,
            "pair": dc.pair,
            "snapshot_summary": d.get("snapshot_summary", {}),
            "analyses": d.get("analyses", {}),
            "debate_rounds": dc.debate_rounds,
            "challenges": d.get("challenges", []),
            "divergence": dc.divergence,
            "verdict": d.get("verdict"),
            "risk_gate": d.get("risk_gate"),
            "order_data": d.get("order"),
            "fill_price": dc.fill_price,
            "slippage": dc.slippage,
            "portfolio_after": d.get("portfolio_after", {}),
            "pnl": dc.pnl,
            "retrospective": dc.retrospective,
            "trace_id": dc.trace_id,
        }

    def _row_to_dc(self, row) -> DecisionCommit:
        from cryptotrader.models import AgentAnalysis, GateResult, Order, TradeVerdict

        analyses = {}
        for k, v in (row.analyses or {}).items():
            if isinstance(v, dict):
                if isinstance(v.get("timestamp"), str):
                    v["timestamp"] = datetime.fromisoformat(v["timestamp"])
                analyses[k] = AgentAnalysis(**v)
        verdict = TradeVerdict(**row.verdict) if row.verdict else None
        risk_gate = GateResult(**row.risk_gate) if row.risk_gate else None
        order = None
        if row.order_data:
            od = dict(row.order_data)
            if "status" in od:
                from cryptotrader.models import OrderStatus

                try:
                    od["status"] = OrderStatus(od["status"])
                except (ValueError, KeyError):
                    od["status"] = OrderStatus.PENDING
            order = Order(**od)
        return DecisionCommit(
            hash=row.hash,
            parent_hash=row.parent_hash,
            timestamp=row.timestamp,
            pair=row.pair,
            snapshot_summary=row.snapshot_summary or {},
            analyses=analyses,
            debate_rounds=row.debate_rounds,
            challenges=row.challenges or [],
            divergence=row.divergence or 0.0,
            verdict=verdict,
            risk_gate=risk_gate,
            order=order,
            fill_price=row.fill_price,
            slippage=row.slippage,
            portfolio_after=row.portfolio_after or {},
            pnl=row.pnl,
            retrospective=row.retrospective,
            trace_id=getattr(row, "trace_id", None),
        )

    async def commit(self, dc: DecisionCommit) -> None:
        if self._use_db:
            try:
                _, Row = _sa_models()
                async with await _get_session(self._db_url) as session:
                    session.add(Row(**self._dc_to_row_dict(dc)))
                    await session.commit()
                # DB write succeeded — flush any pending memory commits
                await self._flush_pending()
                return
            except Exception as e:
                logger.warning("DB commit failed, falling back to memory: %s", e)
        # Fix 5: dedup — skip if hash already in memory
        if any(r["hash"] == dc.hash for r in self._memory):
            return
        self._memory.append(self._serialize(dc))
        if len(self._memory) > self._MAX_MEMORY:
            # Use in-place slice to preserve shared reference for DB-backed instances
            del self._memory[: len(self._memory) - self._MAX_MEMORY]

    async def _flush_pending(self) -> None:
        """Reconcile: write pending in-memory commits back to DB when DB is available."""
        if not self._memory:
            return
        _, Row = _sa_models()
        flushed = []
        for rec in self._memory:
            try:
                dc = self._deserialize(rec)
                async with await _get_session(self._db_url) as session:
                    session.add(Row(**self._dc_to_row_dict(dc)))
                    await session.commit()
                flushed.append(rec)
            except Exception:
                logger.debug("Flush pending commit %s failed, will retry later", rec.get("hash"), exc_info=True)
        if flushed:
            for rec in flushed:
                self._memory.remove(rec)
            logger.info("Flushed %d pending commits to DB", len(flushed))

    async def log(self, limit: int = 10, pair: str | None = None) -> list[DecisionCommit]:
        if self._use_db:
            try:
                _, Row = _sa_models()
                from sqlalchemy import select

                async with await _get_session(self._db_url) as session:
                    q = select(Row).order_by(Row.timestamp.desc())
                    if pair:
                        q = q.where(Row.pair == pair)
                    q = q.limit(limit)
                    result = await session.execute(q)
                    return [self._row_to_dc(r) for r in result.scalars().all()]
            except Exception as e:
                logger.warning("DB query failed: %s", e)
        rows = self._memory
        if pair:
            rows = [r for r in rows if r["data"].get("pair") == pair]
        return [self._deserialize(r) for r in reversed(rows[-limit:])]

    async def show(self, hash: str) -> DecisionCommit | None:
        if self._use_db:
            try:
                _, Row = _sa_models()
                from sqlalchemy import select

                async with await _get_session(self._db_url) as session:
                    result = await session.execute(select(Row).where(Row.hash == hash))
                    row = result.scalar_one_or_none()
                    return self._row_to_dc(row) if row else None
            except Exception as e:
                logger.warning("DB query failed: %s", e)
        for r in self._memory:
            if r["hash"] == hash:
                return self._deserialize(r)
        return None

    async def update_pnl(self, hash: str, pnl: float, retrospective: str) -> None:
        if self._use_db:
            try:
                _, Row = _sa_models()
                from sqlalchemy import update

                async with await _get_session(self._db_url) as session:
                    await session.execute(
                        update(Row).where(Row.hash == hash).values(pnl=pnl, retrospective=retrospective)
                    )
                    await session.commit()
                return
            except Exception as e:
                logger.warning("DB update failed: %s", e)
        for r in self._memory:
            if r["hash"] == hash:
                r["data"]["pnl"] = pnl
                r["data"]["retrospective"] = retrospective
                return
        logger.debug("update_pnl: hash %s not found in memory store", hash)
