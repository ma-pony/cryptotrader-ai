"""Database-backed runtime configuration and encrypted credentials."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, BigInteger, DateTime, LargeBinary, String, inspect, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cryptotrader.db import get_async_session, get_engine
from cryptotrader.runtime_config.defaults import minimal_runtime_document
from cryptotrader.runtime_config.models import RuntimeConfigDocument, RuntimeConfigSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from cryptotrader.runtime_config.secrets import CredentialPayload, CredentialVault, TokenPayload

_GLOBAL_ID = "global"
LLM_GATEWAY_CREDENTIAL_REF = "llm-gateway"
NEWS_PROVIDER_CREDENTIAL_REF = "news-provider"
API_ACCESS_CREDENTIAL_REF = "api-access"


class RevisionConflict(RuntimeError):  # noqa: N818 - public contract uses this exact name.
    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"runtime config revision conflict: expected {expected}, actual {actual}")


class InvalidApplyTransition(RuntimeError):  # noqa: N818 - public contract uses this exact name.
    """A revision can transition only once from pending into its terminal state."""


class CredentialNotConfigured(LookupError):  # noqa: N818 - public contract uses this exact name.
    def __init__(self, credential_ref: str) -> None:
        super().__init__(f"credentials are not configured for {credential_ref}")


class RuntimeConfigUnavailable(RuntimeError):  # noqa: N818 - public staging contract uses this exact name.
    """The read-only staging probe could not find a valid persisted config."""


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
    apply_status: Mapped[str] = mapped_column(String(16), nullable=False)
    applied_revision: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    apply_error: Mapped[str | None] = mapped_column(String(256), nullable=True)


class _RuntimeCredentialRow(_Base):
    __tablename__ = "runtime_credentials"

    credential_ref: Mapped[str] = mapped_column(String(255), primary_key=True)
    encrypted_payload: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _assert_current_schema(connection) -> None:
    """Reject a database created by the removed configuration runtime.

    ``create_all`` only creates missing tables; it deliberately does not alter an
    existing table.  The runtime is a hard cutover, so serving a table missing
    its apply-state columns would split persisted and in-memory configuration.
    Operators must provision the current schema instead of receiving a lazy
    compatibility migration while handling a request.
    """
    columns = {column["name"] for column in inspect(connection).get_columns(_RuntimeConfigRow.__tablename__)}
    required = {"id", "revision", "document", "updated_at", "apply_status", "applied_revision", "apply_error"}
    missing = required - columns
    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(f"runtime_config schema is not current; missing columns: {names}")


def _normalize_datetime(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _snapshot(row: _RuntimeConfigRow) -> RuntimeConfigSnapshot:
    return RuntimeConfigSnapshot(
        revision=row.revision,
        document=RuntimeConfigDocument.model_validate(row.document),
        updated_at=_normalize_datetime(row.updated_at),
        apply_status=row.apply_status,
        applied_revision=row.applied_revision,
        apply_error=row.apply_error,
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
        engine = await get_engine(self.database_url)
        async with engine.begin() as connection:
            await connection.run_sync(_Base.metadata.create_all)
            await connection.run_sync(_assert_current_schema)

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
                    apply_status="applied",
                    applied_revision=1,
                    apply_error=None,
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

    async def get_existing(self) -> RuntimeConfigSnapshot:
        """Read the existing global config without DDL, initialization, or repair."""
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_RuntimeConfigRow, _GLOBAL_ID)
            if row is None:
                raise RuntimeConfigUnavailable("runtime configuration row is unavailable")
            snapshot = _snapshot(row)
            if snapshot.revision < 1:
                raise RuntimeConfigUnavailable("runtime configuration revision is invalid")
            return snapshot
        except RuntimeConfigUnavailable:
            raise
        except Exception as error:
            raise RuntimeConfigUnavailable("runtime configuration schema is unavailable") from error
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
                apply_status="pending",
                apply_error=None,
            )
        )
        session = await get_async_session(self.database_url)
        try:
            result = await session.execute(statement)
            if result.rowcount != 1:
                actual = await self._actual_revision(session)
                await session.rollback()
                raise RevisionConflict(expected_revision, actual)
            row = await session.get(_RuntimeConfigRow, _GLOBAL_ID)
            if row is None:
                await session.rollback()
                raise RuntimeError("runtime config row disappeared")
            await session.commit()
            return _snapshot(row)
        finally:
            await session.close()

    async def _put_encrypted_payload(
        self,
        expected_revision: int,
        credential_ref: str,
        encrypted_payload: bytes,
    ) -> RuntimeConfigSnapshot:
        """Atomically replace one encrypted runtime credential and the config revision."""
        await self.ensure_tables()
        updated_at = datetime.now(UTC)
        session = await get_async_session(self.database_url)
        try:
            result = await session.execute(
                update(_RuntimeConfigRow)
                .where(
                    _RuntimeConfigRow.id == _GLOBAL_ID,
                    _RuntimeConfigRow.revision == expected_revision,
                )
                .values(
                    revision=_RuntimeConfigRow.revision + 1,
                    updated_at=updated_at,
                    apply_status="pending",
                    apply_error=None,
                )
            )
            if result.rowcount != 1:
                actual = await self._actual_revision(session)
                await session.rollback()
                raise RevisionConflict(expected_revision, actual)

            credential = await session.get(_RuntimeCredentialRow, credential_ref)
            if credential is None:
                session.add(
                    _RuntimeCredentialRow(
                        credential_ref=credential_ref,
                        encrypted_payload=encrypted_payload,
                        updated_at=updated_at,
                    )
                )
            else:
                credential.encrypted_payload = encrypted_payload
                credential.updated_at = updated_at

            row = await session.get(_RuntimeConfigRow, _GLOBAL_ID)
            if row is None:
                await session.rollback()
                raise RevisionConflict(expected_revision, 0)
            await session.commit()
            return _snapshot(row)
        finally:
            await session.close()

    async def put_credentials(
        self,
        expected_revision: int,
        credential_ref: str,
        payload: CredentialPayload,
    ) -> RuntimeConfigSnapshot:
        encrypted_payload = self._vault.seal(credential_ref, payload)
        return await self._put_encrypted_payload(expected_revision, credential_ref, encrypted_payload)

    async def credential_state(self, credential_ref: str) -> CredentialState:
        await self.ensure_tables()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_RuntimeCredentialRow, credential_ref)
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
            row = await session.get(_RuntimeCredentialRow, credential_ref)
            if row is None:
                raise CredentialNotConfigured(credential_ref)
            return self._vault.open(credential_ref, row.encrypted_payload)
        finally:
            await session.close()

    async def put_token(
        self,
        expected_revision: int,
        credential_ref: str,
        payload: TokenPayload,
    ) -> RuntimeConfigSnapshot:
        encrypted_payload = self._vault.seal_token(credential_ref, payload)
        return await self._put_encrypted_payload(expected_revision, credential_ref, encrypted_payload)

    async def token_state(self, credential_ref: str) -> CredentialState:
        return await self.credential_state(credential_ref)

    async def reveal_token(self, credential_ref: str) -> TokenPayload:
        await self.ensure_tables()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_RuntimeCredentialRow, credential_ref)
            if row is None:
                raise CredentialNotConfigured(credential_ref)
            return self._vault.open_token(credential_ref, row.encrypted_payload)
        finally:
            await session.close()

    async def mark_applied(self, revision: int) -> RuntimeConfigSnapshot:
        return await self._transition_pending(
            revision,
            apply_status="applied",
            applied_revision=revision,
            apply_error=None,
        )

    async def mark_failed(self, revision: int, error: str) -> RuntimeConfigSnapshot:
        return await self._transition_pending(
            revision,
            apply_status="failed",
            applied_revision=None,
            apply_error=error[:256],
        )

    async def _transition_pending(
        self, revision: int, apply_status: str, *, applied_revision: int | None, apply_error: str | None
    ) -> RuntimeConfigSnapshot:
        await self.ensure_tables()
        session = await get_async_session(self.database_url)
        try:
            values: dict[str, object] = {"apply_status": apply_status, "apply_error": apply_error}
            if applied_revision is not None:
                values["applied_revision"] = applied_revision
            result = await session.execute(
                update(_RuntimeConfigRow)
                .where(
                    _RuntimeConfigRow.id == _GLOBAL_ID,
                    _RuntimeConfigRow.revision == revision,
                    _RuntimeConfigRow.apply_status == "pending",
                )
                .values(**values)
            )
            if result.rowcount != 1:
                row = await session.get(_RuntimeConfigRow, _GLOBAL_ID)
                actual = 0 if row is None else row.revision
                await session.rollback()
                if row is None or actual != revision:
                    raise RevisionConflict(revision, actual)
                raise InvalidApplyTransition("runtime config revision is not pending")
            row = await session.get(_RuntimeConfigRow, _GLOBAL_ID)
            await session.commit()
            if row is None:
                raise RuntimeError("runtime config row disappeared")
            return _snapshot(row)
        finally:
            await session.close()

    @staticmethod
    async def _actual_revision(session) -> int:
        actual = await session.scalar(select(_RuntimeConfigRow.revision).where(_RuntimeConfigRow.id == _GLOBAL_ID))
        return int(actual) if actual is not None else 0
