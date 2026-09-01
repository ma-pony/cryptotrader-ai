# ruff: noqa: RUF001 -- Chinese user-facing messages use Chinese punctuation.
"""The durable source of truth for research runs; no in-memory result fallback."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, String, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cryptotrader.accounts.store import fill_from_payload, funding_from_payload, payload, utc
from cryptotrader.backtest.models import TERMINAL, BacktestParams, RunStatus
from cryptotrader.backtest.result import BacktestResult, ClosedTrade, EquityPoint
from cryptotrader.cycle_serialization import json_value
from cryptotrader.db import get_async_session
from cryptotrader.journal.store import MultiVenueCycleStore, _contains_secret_field
from cryptotrader.migrations.schema import require_tables


class BacktestBase(DeclarativeBase):
    pass


class BacktestRow(BacktestBase):
    __tablename__ = "backtest_runs"
    run_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    progress: Mapped[float] = mapped_column(Float, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    params: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=False)
    config_snapshot: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    error: Mapped[str | None] = mapped_column(String(200))
    incomplete_fields: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    model_evidence: Mapped[list] = mapped_column(JSON, nullable=False, default=list)


@dataclass
class BacktestRun:
    run_id: str
    params: BacktestParams
    config_snapshot: dict | None
    status: RunStatus
    progress: float
    started_at: datetime
    finished_at: datetime | None
    error: str | None
    result: BacktestResult | None
    incomplete_fields: list[str]
    model_evidence: list[dict]


def result_payload(result):
    values = json_value(
        {key: payload(getattr(result, key)) for key in result.__dataclass_fields__ if key != "cycle_records"}
    )
    # This is the existing numeric usage counter, never a credential.
    count = values["llm_tokens"]
    if type(count) is not int or count < 0:
        raise ValueError("invalid model usage counter")
    scanned = {key: value for key, value in values.items() if key != "llm_tokens"}
    if _contains_secret_field(scanned):
        raise ValueError("secret field in research result")
    return values


def result_from_payload(value):
    return BacktestResult(
        **{
            **value,
            "fees": Decimal(value["fees"]),
            "funding": Decimal(value["funding"]),
            "fills": [fill_from_payload(item) for item in value["fills"]],
            "funding_entries": [funding_from_payload(item) for item in value["funding_entries"]],
            "equity_curve": [
                EquityPoint(datetime.fromisoformat(item["time"]), Decimal(item["equity"]))
                for item in value["equity_curve"]
            ],
            "closed_trades": [
                ClosedTrade(
                    **{
                        **item,
                        "opened_at": datetime.fromisoformat(item["opened_at"]),
                        "closed_at": datetime.fromisoformat(item["closed_at"]),
                        "fill_ids": tuple(item["fill_ids"]),
                        **{key: Decimal(item[key]) for key in ("gross_pnl", "fees", "funding", "net_pnl")},
                    }
                )
                for item in value["closed_trades"]
            ],
        }
    )


class BacktestStore:
    def __init__(self, database_url, *, clock=None):
        if not database_url:
            raise ValueError("backtest history requires an explicit database URL")
        self.database_url = database_url
        self.clock = clock or (lambda: datetime.now(UTC))
        self.journal = MultiVenueCycleStore(database_url)
        self._ready = False

    async def ensure_table(self):
        if not self._ready:
            await require_tables(self.database_url, BacktestBase.metadata.tables)
            await self.journal.ensure_table()
            self._ready = True

    async def create(self, params, config_snapshot, *, incomplete_fields=(), run_id=None):
        params = BacktestParams.model_validate(params)
        snapshot = None if config_snapshot is None else json_value(config_snapshot)
        if snapshot == {}:
            raise ValueError("empty research snapshot is invalid; use null for missing evidence")
        if snapshot is not None and _contains_secret_field(snapshot):
            raise ValueError("secret field in research snapshot")
        await self.ensure_table()
        run_id = run_id or f"run_{uuid4().hex}"
        async with await get_async_session(self.database_url) as session, session.begin():
            session.add(
                BacktestRow(
                    run_id=run_id,
                    params=params.model_dump(mode="json"),
                    config_snapshot=snapshot,
                    status="queued",
                    progress=0,
                    started_at=self.clock(),
                    incomplete_fields=list(incomplete_fields),
                )
            )
        return run_id

    async def update(self, run_id, status, progress, result=None, error=None, *, model_evidence=None):
        if status not in {*TERMINAL, "queued", "running"}:
            raise ValueError("unknown backtest status")
        await self.ensure_table()
        async with await get_async_session(self.database_url) as session, session.begin():
            values = {"status": status, "progress": max(0, min(float(progress), 1))}
            if status in TERMINAL:
                values["finished_at"] = self.clock()
            if error is not None:
                values["error"] = error
            if result is not None:
                values["result"] = result_payload(result)
            if model_evidence is not None:
                values["model_evidence"] = json_value(model_evidence)
                if _contains_secret_field(values["model_evidence"]):
                    raise ValueError("secret field in research evidence")
            # Terminal rows are immutable, including late progress/cancellation callbacks.
            changed = await session.execute(
                update(BacktestRow)
                .where(
                    BacktestRow.run_id == run_id,
                    BacktestRow.status.in_(("queued", "running")),
                )
                .values(**values)
            )
            if changed.rowcount and result is not None:
                if result.decision_ids != [record.cycle_id for record in result.cycle_records]:
                    raise ValueError("backtest decisions must reference the original journal records")
                self.journal.stage_records(session, result.cycle_records)

    async def _decode(self, row, *, include_cycles=True):
        if row.config_snapshot == {}:
            raise ValueError(
                "legacy config_snapshot={} is not readable; run the explicit migrate_backtest_snapshots cutover"
            )
        result = result_from_payload(row.result) if row.result is not None else None
        if result is not None and include_cycles:
            result.cycle_records = [await self.journal.get(identity) for identity in result.decision_ids]
            if any(record is None for record in result.cycle_records):
                raise ValueError("backtest journal record is missing")
        return BacktestRun(
            row.run_id,
            BacktestParams.model_validate(row.params),
            row.config_snapshot,
            row.status,
            row.progress,
            utc(row.started_at),
            utc(row.finished_at) if row.finished_at else None,
            row.error,
            result,
            row.incomplete_fields,
            row.model_evidence,
        )

    async def get(self, run_id):
        await self.ensure_table()
        async with await get_async_session(self.database_url) as session:
            row = await session.get(BacktestRow, run_id)
        return await self._decode(row) if row else None

    async def list(self, limit=20, offset=0):
        await self.ensure_table()
        async with await get_async_session(self.database_url) as session:
            rows = (
                await session.scalars(
                    select(BacktestRow)
                    .order_by(
                        BacktestRow.started_at.desc(),
                        BacktestRow.run_id.desc(),
                    )
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        return [await self._decode(row, include_cycles=False) for row in rows]

    async def recover_interrupted(self):
        await self.ensure_table()
        async with await get_async_session(self.database_url) as session, session.begin():
            await session.execute(
                update(BacktestRow)
                .where(BacktestRow.status.in_(("queued", "running")))
                .values(
                    status="interrupted",
                    finished_at=self.clock(),
                    error="服务已重启，运行中断；不会自动续跑。",
                )
            )
