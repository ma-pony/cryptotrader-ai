"""Separate durable business facts and outbox, with database uniqueness/CAS."""
# ruff: noqa: RUF001 -- Chinese operator messages retain Chinese punctuation.

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Integer, String, UniqueConstraint, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cryptotrader.alerts.models import AlertOut, DeliveryOut
from cryptotrader.db import get_async_session


class Base(DeclarativeBase):
    pass


class AlertRow(Base):
    __tablename__ = "business_alerts"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_key: Mapped[str] = mapped_column(String(512), unique=True)
    payload: Mapped[dict] = mapped_column(JSON)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution: Mapped[str] = mapped_column(String(32))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DeliveryRow(Base):
    __tablename__ = "alert_deliveries"
    __table_args__ = (UniqueConstraint("alert_id", "channel"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    alert_id: Mapped[str] = mapped_column(String(36))
    channel: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(String(128))
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def utc(value):
    return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


def alert_out(row):
    return AlertOut(
        **row.payload, id=row.id, read_at=utc(row.read_at), resolution=row.resolution, resolved_at=utc(row.resolved_at)
    )


def delivery_out(row):
    return DeliveryOut(
        **{
            column.name: utc(getattr(row, column.name))
            if isinstance(getattr(row, column.name), datetime)
            else getattr(row, column.name)
            for column in DeliveryRow.__table__.columns
            if column.name != "created_at"
        }
    )


class AlertStore:
    def __init__(self, database_url, *, clock=None):
        if not database_url:
            raise ValueError("alerts require an explicit database URL")
        self.database_url = database_url
        self.clock = clock or (lambda: datetime.now(UTC))

    async def record_once(self, event_key, event, *, channel=None):
        if event_key != event.event_key:
            raise ValueError("event identity mismatch")
        try:
            async with await get_async_session(self.database_url) as session, session.begin():
                row = await session.scalar(select(AlertRow).where(AlertRow.event_key == event_key))
                if row is None:
                    identity = str(uuid4())
                    resolution = {"risk_adjusted": "applied", "daily_summary": "informational"}.get(event.type, "open")
                    row = AlertRow(
                        id=identity,
                        event_key=event_key,
                        payload=event.model_dump(mode="json"),
                        resolution=resolution,
                        resolved_at=event.occurred_at if resolution != "open" else None,
                    )
                    session.add(row)
                if channel is not None:
                    await self._stage_delivery(session, row.id, channel)
                return row.id
        except IntegrityError:
            # A concurrent writer won one of the unique identities. Re-read it
            # and repair a selected but missing delivery in one new transaction.
            async with await get_async_session(self.database_url) as session, session.begin():
                row = await session.scalar(select(AlertRow).where(AlertRow.event_key == event_key))
                if row is None:
                    raise
                if channel is not None:
                    await self._stage_delivery(session, row.id, channel)
                return row.id

    async def _stage_delivery(self, session, alert_id, channel):
        existing = await session.scalar(
            select(DeliveryRow).where(DeliveryRow.alert_id == alert_id, DeliveryRow.channel == channel)
        )
        if existing is None:
            session.add(
                DeliveryRow(
                    id=str(uuid4()),
                    alert_id=alert_id,
                    channel=channel,
                    status="pending",
                    attempts=0,
                    created_at=self.clock(),
                )
            )

    async def get_alert(self, alert_id):
        async with await get_async_session(self.database_url) as session:
            row = await session.get(AlertRow, alert_id)
            if row is None:
                raise LookupError("alert not found")
            return alert_out(row)

    async def get_delivery(self, delivery_id):
        async with await get_async_session(self.database_url) as session:
            row = await session.get(DeliveryRow, delivery_id)
            if row is None:
                raise LookupError("delivery not found")
            return delivery_out(row)

    async def list_alerts(self, *, connection_id=None, book_id=None, unread=False, resolution=None, type=None):
        async with await get_async_session(self.database_url) as session:
            rows = (await session.scalars(select(AlertRow))).all()
            alerts = [alert_out(row) for row in rows]
        return sorted(
            (
                alert
                for alert in alerts
                if (not unread or alert.read_at is None)
                and all(
                    value is None or getattr(alert, key) == value
                    for key, value in {
                        "connection_id": connection_id,
                        "book_id": book_id,
                        "resolution": resolution,
                        "type": type,
                    }.items()
                )
            ),
            key=lambda item: item.occurred_at,
            reverse=True,
        )

    async def list_deliveries(self, *, status=None):
        async with await get_async_session(self.database_url) as session:
            query = select(DeliveryRow).order_by(
                func.coalesce(DeliveryRow.last_attempt_at, DeliveryRow.created_at).desc(), DeliveryRow.id.desc()
            )
            if status is not None:
                query = query.where(DeliveryRow.status == status)
            return [delivery_out(row) for row in (await session.scalars(query)).all()]

    async def mark_read(self, alert_id):
        await self.get_alert(alert_id)
        async with await get_async_session(self.database_url) as session, session.begin():
            await session.execute(
                update(AlertRow).where(AlertRow.id == alert_id, AlertRow.read_at.is_(None)).values(read_at=self.clock())
            )
        return await self.get_alert(alert_id)

    async def resolve(self, alert_id, resolution):
        async with await get_async_session(self.database_url) as session, session.begin():
            await session.execute(
                update(AlertRow)
                .where(AlertRow.id == alert_id, AlertRow.resolution == "open")
                .values(resolution=resolution, resolved_at=self.clock())
            )

    async def claim_delivery(self, delivery_id):
        await self.get_delivery(delivery_id)
        async with await get_async_session(self.database_url) as session, session.begin():
            result = await session.execute(
                update(DeliveryRow)
                .where(DeliveryRow.id == delivery_id, DeliveryRow.status == "pending")
                .values(
                    status="sending", attempts=DeliveryRow.attempts + 1, last_attempt_at=self.clock(), last_error=None
                )
            )
            return result.rowcount == 1

    async def finish_delivery(self, delivery_id, error=None):
        async with await get_async_session(self.database_url) as session, session.begin():
            await session.execute(
                update(DeliveryRow)
                .where(DeliveryRow.id == delivery_id, DeliveryRow.status == "sending")
                .values(
                    status="failed" if error else "delivered",
                    last_error=error,
                    delivered_at=None if error else self.clock(),
                )
            )

    async def retry_delivery(self, delivery_id):
        await self.get_delivery(delivery_id)
        async with await get_async_session(self.database_url) as session, session.begin():
            result = await session.execute(
                update(DeliveryRow)
                .where(DeliveryRow.id == delivery_id, DeliveryRow.status == "failed")
                .values(status="pending")
            )
            if result.rowcount != 1:
                raise ValueError("only failed deliveries can be retried")
        return await self.get_delivery(delivery_id)

    async def recover_interrupted(self):
        async with await get_async_session(self.database_url) as session, session.begin():
            await session.execute(
                update(DeliveryRow)
                .where(DeliveryRow.status == "sending")
                .values(status="failed", last_error="投递中断，送达未知，可手动重试")
            )
