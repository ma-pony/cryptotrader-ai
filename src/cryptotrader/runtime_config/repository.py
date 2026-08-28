"""Database-backed runtime configuration and encrypted credentials."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, BigInteger, DateTime, LargeBinary, String, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cryptotrader.db import get_async_session, get_engine
from cryptotrader.runtime_config.defaults import minimal_runtime_document
from cryptotrader.runtime_config.models import RuntimeConfigDocument, RuntimeConfigSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from cryptotrader.runtime_config.secrets import CredentialPayload, CredentialVault

_GLOBAL_ID = "global"
_ready: set[str] = set()


class RevisionConflict(RuntimeError):  # noqa: N818 - public contract uses this exact name.
    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"runtime config revision conflict: expected {expected}, actual {actual}")


class CredentialNotConfigured(LookupError):  # noqa: N818 - public contract uses this exact name.
    def __init__(self, credential_ref: str) -> None:
        super().__init__(f"credentials are not configured for {credential_ref}")


@dataclass(frozen=True)
class CredentialState:
    credential_ref: str
    configured: bool
    updated_at: datetime | None


class _Base(DeclarativeBase):
    pass


class _RuntimeConfigRow(_Base):
    __tablename__ = "runtime_config"

    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    document: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class _VenueCredentialRow(_Base):
    __tablename__ = "venue_credentials"

    credential_ref: Mapped[str] = mapped_column(String(255), primary_key=True)
    encrypted_payload: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _normalize_datetime(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _snapshot(row: _RuntimeConfigRow) -> RuntimeConfigSnapshot:
    return RuntimeConfigSnapshot(
        revision=row.revision,
        document=RuntimeConfigDocument.model_validate(row.document),
        updated_at=_normalize_datetime(row.updated_at),
    )


class RuntimeConfigRepository:
    def __init__(
        self,
        database_url: str,
        vault: CredentialVault,
        default_factory: Callable[[], RuntimeConfigDocument] = minimal_runtime_document,
    ) -> None:
        self.database_url = database_url
        self._vault = vault
        self._default_factory = default_factory

    async def ensure_tables(self) -> None:
        if self.database_url in _ready:
            return
        engine = await get_engine(self.database_url)
        async with engine.begin() as connection:
            await connection.run_sync(_Base.metadata.create_all)
        _ready.add(self.database_url)

    async def get_or_create(self) -> RuntimeConfigSnapshot:
        await self.ensure_tables()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_RuntimeConfigRow, _GLOBAL_ID)
            if row is None:
                row = _RuntimeConfigRow(
                    id=_GLOBAL_ID,
                    revision=1,
                    document=self._default_factory().model_dump(mode="json"),
                    updated_at=datetime.now(UTC),
                )
                session.add(row)
                try:
                    await session.commit()
                except IntegrityError:
                    await session.rollback()
                    row = await session.get(_RuntimeConfigRow, _GLOBAL_ID)
                    if row is None:
                        raise RuntimeError("runtime config initialization failed") from None
            return _snapshot(row)
        finally:
            await session.close()

    async def replace(
        self,
        expected_revision: int,
        document: RuntimeConfigDocument,
    ) -> RuntimeConfigSnapshot:
        await self.ensure_tables()
        updated_at = datetime.now(UTC)
        statement = (
            update(_RuntimeConfigRow)
            .where(
                _RuntimeConfigRow.id == _GLOBAL_ID,
                _RuntimeConfigRow.revision == expected_revision,
            )
            .values(
                document=document.model_dump(mode="json"),
                revision=_RuntimeConfigRow.revision + 1,
                updated_at=updated_at,
            )
        )
        session = await get_async_session(self.database_url)
        try:
            result = await session.execute(statement)
            if result.rowcount != 1:
                actual = await self._actual_revision(session)
                await session.rollback()
                raise RevisionConflict(expected_revision, actual)
            await session.commit()
            return RuntimeConfigSnapshot(expected_revision + 1, document, updated_at)
        finally:
            await session.close()

    async def put_credentials(
        self,
        expected_revision: int,
        credential_ref: str,
        payload: CredentialPayload,
    ) -> RuntimeConfigSnapshot:
        await self.ensure_tables()
        updated_at = datetime.now(UTC)
        encrypted_payload = self._vault.seal(credential_ref, payload)
        session = await get_async_session(self.database_url)
        try:
            credential = await session.get(_VenueCredentialRow, credential_ref)
            if credential is None:
                session.add(
                    _VenueCredentialRow(
                        credential_ref=credential_ref,
                        encrypted_payload=encrypted_payload,
                        updated_at=updated_at,
                    )
                )
            else:
                credential.encrypted_payload = encrypted_payload
                credential.updated_at = updated_at

            result = await session.execute(
                update(_RuntimeConfigRow)
                .where(
                    _RuntimeConfigRow.id == _GLOBAL_ID,
                    _RuntimeConfigRow.revision == expected_revision,
                )
                .values(
                    revision=_RuntimeConfigRow.revision + 1,
                    updated_at=updated_at,
                )
            )
            if result.rowcount != 1:
                actual = await self._actual_revision(session)
                await session.rollback()
                raise RevisionConflict(expected_revision, actual)

            row = await session.get(_RuntimeConfigRow, _GLOBAL_ID)
            if row is None:
                await session.rollback()
                raise RevisionConflict(expected_revision, 0)
            await session.commit()
            return _snapshot(row)
        finally:
            await session.close()

    async def credential_state(self, credential_ref: str) -> CredentialState:
        await self.ensure_tables()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_VenueCredentialRow, credential_ref)
            return CredentialState(
                credential_ref=credential_ref,
                configured=row is not None,
                updated_at=_normalize_datetime(row.updated_at) if row is not None else None,
            )
        finally:
            await session.close()

    async def reveal_credentials(self, credential_ref: str) -> CredentialPayload:
        await self.ensure_tables()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_VenueCredentialRow, credential_ref)
            if row is None:
                raise CredentialNotConfigured(credential_ref)
            return self._vault.open(credential_ref, row.encrypted_payload)
        finally:
            await session.close()

    @staticmethod
    async def _actual_revision(session) -> int:
        actual = await session.scalar(select(_RuntimeConfigRow.revision).where(_RuntimeConfigRow.id == _GLOBAL_ID))
        return int(actual) if actual is not None else 0
