"""Decision journal storage — PostgreSQL with in-memory fallback."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

from cryptotrader.db import get_async_session, get_engine
from cryptotrader.models import ConsensusMetrics, DecisionCommit, NodeTraceEntry
from cryptotrader.pair import market_type_for as _market_type_for

logger = logging.getLogger(__name__)


def _to_node_trace_entries(raw: list) -> list[NodeTraceEntry]:
    """Convert JSONB-roundtripped node_trace dicts back into dataclass entries.

    Tolerant of extra keys (the runtime registry may carry ``ts`` / ``output``)
    and missing ``summary`` (defaults to empty string).
    """
    out: list[NodeTraceEntry] = []
    for entry in raw or []:
        if not isinstance(entry, dict) or not entry.get("node"):
            continue
        out.append(
            NodeTraceEntry(
                node=str(entry["node"]),
                duration_ms=int(entry.get("duration_ms") or 0),
                summary=str(entry.get("summary") or ""),
            )
        )
    return out


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
        # Spec 013 deep-review correctness FINDING-2: VARCHAR(50) accommodates
        # ccxt futures delivery symbols (e.g., "BTC/USDT:USDT-241227" = 21 chars).
        pair = Column(String(50), nullable=False, index=True)
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
        # Observability columns (task 1.3)
        consensus_metrics = Column(JSONB, nullable=True, default=None)
        verdict_source = Column(String(20), nullable=False, default="ai")
        node_trace = Column(JSONB, nullable=False, default=[])
        debate_skip_reason = Column(String(500), nullable=False, default="")
        # Spec: frontend-prototype-alignment (2026-04-24)
        latency_breakdown = Column(JSONB, nullable=False, default={})
        token_usage = Column(JSONB, nullable=False, default={})
        # Spec 013: market_type derived from pair (Pair.parse(pair).market_type)
        market_type = Column(String(20), nullable=False, default="spot")
        # Execution-layer status: succeeded / stage / reason. Distinct from
        # ``risk_gate`` so analytics on gate-pass rate are not skewed by
        # post-gate execution failures (e.g. spot_short_no_inventory).
        execution_status = Column(JSONB, nullable=True, default=None)
        # Phase 2C: server-side SL/TP audit trail (LLM's numeric stop_loss /
        # take_profit + the OKX algo OCO id that was actually placed).
        stop_loss_price = Column(Float, nullable=True)
        take_profit_price = Column(Float, nullable=True)
        algo_id = Column(String(64), nullable=True, index=True)

    _sa_cache = (Base, DecisionCommitRow)
    return _sa_cache


_OBSERVABILITY_COLUMNS = [
    # (column_name, DDL_type, DEFAULT_clause)
    ("consensus_metrics", "JSONB", "DEFAULT NULL"),
    ("verdict_source", "VARCHAR(20)", "NOT NULL DEFAULT 'ai'"),
    ("node_trace", "JSONB", "NOT NULL DEFAULT '[]'"),
    ("debate_skip_reason", "VARCHAR(500)", "NOT NULL DEFAULT ''"),
    ("latency_breakdown", "JSONB", "NOT NULL DEFAULT '{}'"),
    ("token_usage", "JSONB", "NOT NULL DEFAULT '{}'"),
    # Spec 013: market_type column for distinguishing spot vs derivatives commits.
    ("market_type", "VARCHAR(20)", "NOT NULL DEFAULT 'spot'"),
    # 2026-05-06: separate execution-layer status from risk_gate so
    # post-gate failures don't pollute gate-pass-rate analytics.
    ("execution_status", "JSONB", "DEFAULT NULL"),
    # Phase 2C (2026-05-13): server-side SL/TP audit trail.
    ("stop_loss_price", "DOUBLE PRECISION", "DEFAULT NULL"),
    ("take_profit_price", "DOUBLE PRECISION", "DEFAULT NULL"),
    ("algo_id", "VARCHAR(64)", "DEFAULT NULL"),
]

# Columns to DROP from existing databases (FR-031: experience_json/experience_memory removal).
# For PostgreSQL: ALTER TABLE … DROP COLUMN IF EXISTS. For SQLite: no-op (column is harmless).
_DROP_COLUMNS = [
    "experience_memory",  # FR-031: replaced by file-based agent_memory/ two-layer architecture
]


async def _ensure_tables(database_url: str) -> None:
    """Create journal schema on first call per database URL.

    New columns must be backfilled on *existing* databases too — ``create_all`` is
    non-migrating and only adds tables, not columns. For PostgreSQL we use
    ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``. SQLite rejects IF NOT EXISTS on
    ADD COLUMN, so we first probe the column list via ``PRAGMA table_info`` and
    ALTER only when the column is absent.
    """
    from sqlalchemy import text

    if database_url not in _table_ready:
        Base, _ = _sa_models()
        engine = await get_engine(database_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            dialect = conn.dialect.name
            if dialect == "postgresql":
                for col_name, col_type, col_default in _OBSERVABILITY_COLUMNS:
                    await conn.execute(
                        text(
                            f"ALTER TABLE decision_commits ADD COLUMN IF NOT EXISTS {col_name} {col_type} {col_default}"
                        )
                    )
                # FR-031: drop legacy experience_memory column (replaced by file-based agent_memory/)
                for col_name in _DROP_COLUMNS:
                    await conn.execute(text(f"ALTER TABLE decision_commits DROP COLUMN IF EXISTS {col_name}"))
                # Spec 013 deep-review: widen pair column from VARCHAR(20) to VARCHAR(50)
                # to fit ccxt futures delivery symbols. Idempotent — re-running is a no-op.
                await conn.execute(text("ALTER TABLE decision_commits ALTER COLUMN pair TYPE VARCHAR(50)"))
            elif dialect == "sqlite":
                # SQLite `ALTER TABLE ADD COLUMN` has no IF NOT EXISTS — detect via PRAGMA.
                result = await conn.execute(text("PRAGMA table_info(decision_commits)"))
                existing_cols = {row[1] for row in result.fetchall()}  # index 1 is column name
                for col_name, col_type, col_default in _OBSERVABILITY_COLUMNS:
                    if col_name in existing_cols:
                        continue
                    # SQLite types are flexible — treat JSONB/VARCHAR as TEXT; keep DEFAULT clause.
                    sqlite_type = "TEXT" if col_type.upper() in {"JSONB", "TEXT"} else col_type
                    await conn.execute(
                        text(f"ALTER TABLE decision_commits ADD COLUMN {col_name} {sqlite_type} {col_default}")
                    )
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

    @classmethod
    def clear_backtest_memory(cls) -> None:
        """Remove all ``::backtest`` keyed fallback memory to prevent unbounded growth."""
        keys = [k for k in cls._shared_memory if k.endswith("::backtest")]
        for k in keys:
            del cls._shared_memory[k]

    def __init__(self, database_url: str | None = None, *, backtest_mode: bool = False):
        self._db_url = database_url
        self._backtest_mode = backtest_mode
        if database_url:
            # Backtest uses a separate fallback key to prevent cross-contamination
            # with live journal entries when DB is temporarily unreachable.
            mem_key = f"{database_url}::backtest" if backtest_mode else database_url
            if mem_key not in JournalStore._shared_memory:
                JournalStore._shared_memory[mem_key] = []
            self._memory: list[dict[str, Any]] = JournalStore._shared_memory[mem_key]
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
        # Observability fields — None-safe deserialization
        cm_data = d.get("consensus_metrics")
        consensus_metrics = ConsensusMetrics(**cm_data) if cm_data else None
        node_trace = _to_node_trace_entries(d.get("node_trace") or [])
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
            consensus_metrics=consensus_metrics,
            verdict_source=d.get("verdict_source", "ai"),
            node_trace=node_trace,
            debate_skip_reason=d.get("debate_skip_reason", ""),
            latency_breakdown=d.get("latency_breakdown") or {},
            token_usage=d.get("token_usage") or {},
            execution_status=d.get("execution_status"),
            stop_loss_price=d.get("stop_loss_price"),
            take_profit_price=d.get("take_profit_price"),
            algo_id=d.get("algo_id"),
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
            # Observability fields (task 1.3)
            "consensus_metrics": d.get("consensus_metrics"),
            "verdict_source": dc.verdict_source,
            "node_trace": d.get("node_trace", []),
            "debate_skip_reason": dc.debate_skip_reason,
            "latency_breakdown": d.get("latency_breakdown") or {},
            "token_usage": d.get("token_usage") or {},
            # Spec 013: derived from pair string at write time. Falls back to
            # 'spot' on parse failure so legacy / malformed pairs don't reject.
            "market_type": _market_type_for(dc.pair),
            "execution_status": dc.execution_status,
            # Phase 2C: server-side SL/TP audit trail
            "stop_loss_price": dc.stop_loss_price,
            "take_profit_price": dc.take_profit_price,
            "algo_id": dc.algo_id,
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
        # Observability fields — None-safe deserialization for old rows (NULL columns)
        cm_data = getattr(row, "consensus_metrics", None)
        consensus_metrics = ConsensusMetrics(**cm_data) if cm_data else None
        node_trace_data = getattr(row, "node_trace", None) or []
        node_trace = _to_node_trace_entries(node_trace_data)
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
            consensus_metrics=consensus_metrics,
            verdict_source=getattr(row, "verdict_source", None) or "ai",
            node_trace=node_trace,
            debate_skip_reason=getattr(row, "debate_skip_reason", None) or "",
            latency_breakdown=getattr(row, "latency_breakdown", None) or {},
            token_usage=getattr(row, "token_usage", None) or {},
            execution_status=getattr(row, "execution_status", None),
            stop_loss_price=getattr(row, "stop_loss_price", None),
            take_profit_price=getattr(row, "take_profit_price", None),
            algo_id=getattr(row, "algo_id", None),
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
                logger.warning("Flush pending commit %s failed, will retry later", rec.get("hash"), exc_info=True)
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
