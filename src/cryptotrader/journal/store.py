"""`trading_cycles` 的 PostgreSQL/SQLite 持久化与内存实现。"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, String, func, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cryptotrader.db import get_async_session, get_engine
from cryptotrader.decision.models import TargetPosition
from cryptotrader.execution.codec import (
    book_execution_proposal_from_payload,
    book_execution_proposal_payload,
    book_execution_result_from_payload,
    book_execution_result_payload,
)
from cryptotrader.journal.models import (
    BookCycleResult,
    BookHitlSnapshot,
    MultiVenueCycleRecord,
    TradingCycleRecord,
)
from cryptotrader.pair import Pair
from cryptotrader.portfolio.models import BookPortfolioSnapshot, ConnectionPortfolioSnapshot
from cryptotrader.signals.fusion import ComponentContribution, FusedSignal
from cryptotrader.signals.models import ComponentSignal
from cryptotrader.venues.models import ConnectionPosition

if TYPE_CHECKING:
    from cryptotrader.decision.models import CycleStatus

_ready: set[str] = set()
_multi_venue_ready: set[str] = set()


class _Base(DeclarativeBase):
    pass


class _TradingCycleRow(_Base):
    __tablename__ = "trading_cycles"

    cycle_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    pair: Mapped[str] = mapped_column(String(50), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    profile_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
    )


class _MultiVenueBase(DeclarativeBase):
    pass


class _MultiVenueCycleRow(_MultiVenueBase):
    __tablename__ = "multi_venue_cycles"

    cycle_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    config_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    market_data_source_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    component_signals: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
    )
    fused_signal: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
    )
    target_position: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
    )
    book_results: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
    )
    cycle_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    execution_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    requires_attention: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)


def _payload(record: TradingCycleRecord) -> dict[str, Any]:
    return {
        "profile_snapshot": dict(record.profile_snapshot),
        "context_summary": dict(record.context_summary),
        "component_signals": [dict(item) for item in record.component_signals],
        "component_error": dict(record.component_error) if record.component_error is not None else None,
        "fused_signal": dict(record.fused_signal) if record.fused_signal is not None else None,
        "target_position": dict(record.target_position) if record.target_position is not None else None,
        "trade_plan": dict(record.trade_plan) if record.trade_plan is not None else None,
        "hitl_result": dict(record.hitl_result) if record.hitl_result is not None else None,
        "risk_result": dict(record.risk_result) if record.risk_result is not None else None,
        "execution_result": dict(record.execution_result) if record.execution_result is not None else None,
        "error": record.error,
    }


def _record(row: _TradingCycleRow) -> TradingCycleRecord:
    payload = row.payload
    created_at = row.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return TradingCycleRecord(
        cycle_id=row.cycle_id,
        created_at=created_at,
        pair=row.pair,
        status=cast("CycleStatus", row.status),
        profile_revision=row.profile_revision,
        profile_snapshot=payload["profile_snapshot"],
        context_summary=payload["context_summary"],
        component_signals=tuple(payload["component_signals"]),
        component_error=payload["component_error"],
        fused_signal=payload["fused_signal"],
        target_position=payload["target_position"],
        trade_plan=payload["trade_plan"],
        hitl_result=payload["hitl_result"],
        risk_result=payload["risk_result"],
        execution_result=payload["execution_result"],
        error=payload["error"],
    )


class CycleJournalStore:
    """追加并查询完整交易周期。无数据库时使用实例级内存。"""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url
        self.records: list[TradingCycleRecord] = []

    async def ensure_table(self) -> None:
        if self.database_url is None or self.database_url in _ready:
            return
        engine = await get_engine(self.database_url)
        async with engine.begin() as connection:
            await connection.run_sync(_Base.metadata.create_all)
        _ready.add(self.database_url)

    async def append(self, record: TradingCycleRecord) -> None:
        if self.database_url is None:
            if any(item.cycle_id == record.cycle_id for item in self.records):
                raise ValueError(f"cycle {record.cycle_id!r} already exists")
            self.records.append(record)
            return

        await self.ensure_table()
        session = await get_async_session(self.database_url)
        try:
            session.add(
                _TradingCycleRow(
                    cycle_id=record.cycle_id,
                    created_at=record.created_at,
                    pair=record.pair,
                    status=record.status,
                    profile_revision=record.profile_revision,
                    payload=_payload(record),
                )
            )
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise ValueError(f"cycle {record.cycle_id!r} already exists") from exc
        finally:
            await session.close()

    async def get(self, cycle_id: str) -> TradingCycleRecord | None:
        if self.database_url is None:
            return next((item for item in self.records if item.cycle_id == cycle_id), None)

        await self.ensure_table()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_TradingCycleRow, cycle_id)
            return _record(row) if row is not None else None
        finally:
            await session.close()

    async def replace(self, record: TradingCycleRecord) -> None:
        """替换同一周期的暂停状态。供 HITL 终态迁移使用。"""
        if self.database_url is None:
            for index, current in enumerate(self.records):
                if current.cycle_id == record.cycle_id:
                    self.records[index] = record
                    return
            raise LookupError(f"cycle {record.cycle_id!r} does not exist")

        await self.ensure_table()
        statement = (
            update(_TradingCycleRow)
            .where(_TradingCycleRow.cycle_id == record.cycle_id)
            .values(
                created_at=record.created_at,
                pair=record.pair,
                status=record.status,
                profile_revision=record.profile_revision,
                payload=_payload(record),
            )
        )
        session = await get_async_session(self.database_url)
        try:
            result = await session.execute(statement)
            if result.rowcount != 1:
                await session.rollback()
                raise LookupError(f"cycle {record.cycle_id!r} does not exist")
            await session.commit()
        finally:
            await session.close()

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        pair: str | None = None,
        status: CycleStatus | None = None,
    ) -> list[TradingCycleRecord]:
        if limit < 1 or offset < 0:
            return []
        if self.database_url is None:
            records = self.records
            if pair is not None:
                records = [item for item in records if item.pair == pair]
            if status is not None:
                records = [item for item in records if item.status == status]
            ordered = sorted(records, key=lambda item: item.created_at, reverse=True)
            return ordered[offset : offset + limit]

        await self.ensure_table()
        query = select(_TradingCycleRow)
        if pair is not None:
            query = query.where(_TradingCycleRow.pair == pair)
        if status is not None:
            query = query.where(_TradingCycleRow.status == status)
        query = query.order_by(_TradingCycleRow.created_at.desc()).offset(offset).limit(limit)
        session = await get_async_session(self.database_url)
        try:
            rows = (await session.execute(query)).scalars().all()
            return [_record(row) for row in rows]
        finally:
            await session.close()

    async def count(
        self,
        *,
        pair: str | None = None,
        status: CycleStatus | None = None,
    ) -> int:
        if self.database_url is None:
            records = self.records
            if pair is not None:
                records = [item for item in records if item.pair == pair]
            if status is not None:
                records = [item for item in records if item.status == status]
            return len(records)

        await self.ensure_table()
        query = select(func.count()).select_from(_TradingCycleRow)
        if pair is not None:
            query = query.where(_TradingCycleRow.pair == pair)
        if status is not None:
            query = query.where(_TradingCycleRow.status == status)
        session = await get_async_session(self.database_url)
        try:
            return int((await session.execute(query)).scalar_one())
        finally:
            await session.close()


_JOURNAL_CODEC_VERSION = 1
_ALWAYS_SECRET_TOKENS = frozenset(
    {"secret", "password", "passphrase", "token", "credential", "credentials", "authorization"}
)
_COMPACT_SECRET_KEYS = frozenset(
    {
        "apikey",
        "apisecret",
        "secret",
        "passphrase",
        "password",
        "token",
        "credential",
        "credentials",
        "privatekey",
        "authorization",
    }
)


def _secret_key_tokens(key: str) -> frozenset[str]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return frozenset(token for token in re.split(r"[^A-Za-z0-9]+", separated.lower()) if token)


def _is_secret_field(key: str) -> bool:
    tokens = _secret_key_tokens(key)
    compact = "".join(character for character in key.lower() if character.isalnum())
    return (
        bool(tokens & _ALWAYS_SECRET_TOKENS)
        or {"api", "key"} <= tokens
        or {"private", "key"} <= tokens
        or any(marker in compact for marker in _COMPACT_SECRET_KEYS)
    )


def _contains_secret_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            (type(key) is str and _is_secret_field(key)) or _contains_secret_field(item) for key, item in value.items()
        )
    if type(value) in {list, tuple}:
        return any(_contains_secret_field(item) for item in value)
    return False


def _codec_object(value: Any, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("invalid journal codec object")
    return value


def _codec_array(value: Any) -> list[Any]:
    if type(value) is not list:
        raise ValueError("invalid journal codec array")
    return value


def _require_journal_version(value: Any) -> None:
    if type(value) is not int or value != _JOURNAL_CODEC_VERSION:
        raise ValueError("unsupported journal codec version")


def _detail_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "items": [[key, _detail_payload(item)] for key, item in value.items()],
        }
    if type(value) in {list, tuple}:
        return {"kind": "sequence", "items": [_detail_payload(item) for item in value]}
    if isinstance(value, Decimal):
        return {"kind": "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {"kind": "datetime", "value": value.isoformat()}
    if isinstance(value, Pair):
        return {"kind": "pair", "value": value.canonical()}
    if value is None or type(value) in {str, bool, int}:
        return {"kind": "primitive", "value": value}
    if type(value) is float and math.isfinite(value):
        return {"kind": "primitive", "value": value}
    raise ValueError("unsupported journal detail value")


def _detail_from_payload(value: Any) -> Any:
    if type(value) is not dict or type(value.get("kind")) is not str:
        raise ValueError("invalid journal detail")
    kind = value["kind"]
    if kind == "mapping":
        return _detail_mapping_from_payload(value)
    if kind == "sequence":
        payload = _codec_object(value, {"kind", "items"})
        return tuple(_detail_from_payload(item) for item in _codec_array(payload["items"]))
    return _detail_scalar_from_payload(value, kind)


def _detail_mapping_from_payload(value: Any) -> dict[str, Any]:
    payload = _codec_object(value, {"kind", "items"})
    result: dict[str, Any] = {}
    for pair in _codec_array(payload["items"]):
        if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str or pair[0] in result:
            raise ValueError("invalid journal detail mapping")
        result[pair[0]] = _detail_from_payload(pair[1])
    return result


def _detail_scalar_from_payload(value: Any, kind: str) -> Any:
    payload = _codec_object(value, {"kind", "value"})
    decoders = {
        "decimal": _detail_decimal_from_payload,
        "datetime": _detail_datetime_from_payload,
        "pair": _detail_pair_from_payload,
    }
    if kind in decoders:
        return decoders[kind](payload["value"])
    if kind == "primitive" and (payload["value"] is None or type(payload["value"]) in {str, bool, int, float}):
        if type(payload["value"]) is float and not math.isfinite(payload["value"]):
            raise ValueError("invalid journal float")
        return payload["value"]
    raise ValueError("unsupported journal detail kind")


def _detail_decimal_from_payload(value: Any) -> Decimal:
    if type(value) is not str:
        raise ValueError("invalid journal decimal")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("invalid journal decimal")
    return result


def _detail_datetime_from_payload(value: Any) -> datetime:
    if type(value) is not str:
        raise ValueError("invalid journal datetime")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("invalid journal datetime")
    return result


def _detail_pair_from_payload(value: Any) -> Pair:
    if type(value) is not str:
        raise ValueError("invalid journal pair")
    return Pair.parse(value)


def _component_signal_payload(value: ComponentSignal) -> dict[str, Any]:
    return {
        "component_id": value.component_id,
        "direction": value.direction,
        "confidence": value.confidence,
        "reasoning": value.reasoning,
        "details": _detail_payload(value.details),
    }


def _component_signal_from_payload(value: Any) -> ComponentSignal:
    payload = _codec_object(value, {"component_id", "direction", "confidence", "reasoning", "details"})
    details = _detail_from_payload(payload["details"])
    if not isinstance(details, Mapping):
        raise ValueError("invalid component signal details")
    return ComponentSignal(
        payload["component_id"],
        payload["direction"],
        payload["confidence"],
        payload["reasoning"],
        details,
    )


def _component_signals_payload(values: tuple[ComponentSignal, ...]) -> dict[str, Any]:
    return {
        "version": _JOURNAL_CODEC_VERSION,
        "items": [_component_signal_payload(item) for item in values],
    }


def _component_signals_from_payload(value: Any) -> tuple[ComponentSignal, ...]:
    payload = _codec_object(value, {"version", "items"})
    _require_journal_version(payload["version"])
    return tuple(_component_signal_from_payload(item) for item in _codec_array(payload["items"]))


def _fused_signal_payload(value: FusedSignal | None) -> dict[str, Any]:
    fused = None
    if value is not None:
        fused = {
            "score": value.score,
            "contributions": [
                {
                    "component_id": item.component_id,
                    "weight": item.weight,
                    "signed_score": item.signed_score,
                    "weighted_score": item.weighted_score,
                }
                for item in value.contributions
            ],
            "reasoning": value.reasoning,
        }
    return {"version": _JOURNAL_CODEC_VERSION, "value": fused}


def _fused_signal_from_payload(value: Any) -> FusedSignal | None:
    wrapper = _codec_object(value, {"version", "value"})
    _require_journal_version(wrapper["version"])
    if wrapper["value"] is None:
        return None
    payload = _codec_object(wrapper["value"], {"score", "contributions", "reasoning"})
    contributions = tuple(
        ComponentContribution(
            item["component_id"],
            item["weight"],
            item["signed_score"],
            item["weighted_score"],
        )
        for raw_item in _codec_array(payload["contributions"])
        for item in [_codec_object(raw_item, {"component_id", "weight", "signed_score", "weighted_score"})]
    )
    return FusedSignal(payload["score"], contributions, payload["reasoning"])


def _target_position_payload(value: TargetPosition | None) -> dict[str, Any]:
    target = None if value is None else {"side": value.side, "size_ratio": value.size_ratio}
    return {"version": _JOURNAL_CODEC_VERSION, "value": target}


def _target_position_from_payload(value: Any) -> TargetPosition | None:
    wrapper = _codec_object(value, {"version", "value"})
    _require_journal_version(wrapper["version"])
    if wrapper["value"] is None:
        return None
    payload = _codec_object(wrapper["value"], {"side", "size_ratio"})
    return TargetPosition(payload["side"], payload["size_ratio"])


def _connection_portfolio_payload(value: ConnectionPortfolioSnapshot) -> dict[str, Any]:
    return {
        "connection_id": value.connection_id,
        "equity": str(value.equity),
        "balances": [[asset, str(amount)] for asset, amount in value.balances.items()],
        "position": {
            "pair": value.position.pair.canonical(),
            "signed_amount": str(value.position.signed_amount),
            "signed_notional": str(value.position.signed_notional),
            "entry_price": None if value.position.entry_price is None else str(value.position.entry_price),
        },
    }


def _decimal_from_payload(value: Any) -> Decimal:
    if type(value) is not str:
        raise ValueError("invalid journal decimal")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("invalid journal decimal")
    return result


def _connection_portfolio_from_payload(value: Any) -> ConnectionPortfolioSnapshot:
    payload = _codec_object(value, {"connection_id", "equity", "balances", "position"})
    balances: dict[str, Decimal] = {}
    for raw_balance in _codec_array(payload["balances"]):
        if type(raw_balance) is not list or len(raw_balance) != 2 or type(raw_balance[0]) is not str:
            raise ValueError("invalid journal balance")
        asset = raw_balance[0]
        if asset in balances:
            raise ValueError("duplicate journal balance asset")
        balances[asset] = _decimal_from_payload(raw_balance[1])
    position_payload = _codec_object(
        payload["position"],
        {"pair", "signed_amount", "signed_notional", "entry_price"},
    )
    if type(position_payload["pair"]) is not str:
        raise ValueError("invalid journal position pair")
    entry_price = position_payload["entry_price"]
    return ConnectionPortfolioSnapshot(
        payload["connection_id"],
        _decimal_from_payload(payload["equity"]),
        balances,
        ConnectionPosition(
            Pair.parse(position_payload["pair"]),
            _decimal_from_payload(position_payload["signed_amount"]),
            _decimal_from_payload(position_payload["signed_notional"]),
            None if entry_price is None else _decimal_from_payload(entry_price),
        ),
    )


def _book_portfolio_payload(value: BookPortfolioSnapshot) -> dict[str, Any]:
    return {
        "book_id": value.book_id,
        "capital_scope": value.capital_scope,
        "total_equity": str(value.total_equity),
        "total_signed_notional": str(value.total_signed_notional),
        "connections": [_connection_portfolio_payload(item) for item in value.connections],
    }


def _book_portfolio_from_payload(value: Any) -> BookPortfolioSnapshot:
    payload = _codec_object(
        value,
        {"book_id", "capital_scope", "total_equity", "total_signed_notional", "connections"},
    )
    return BookPortfolioSnapshot(
        payload["book_id"],
        payload["capital_scope"],
        _decimal_from_payload(payload["total_equity"]),
        _decimal_from_payload(payload["total_signed_notional"]),
        tuple(_connection_portfolio_from_payload(item) for item in _codec_array(payload["connections"])),
    )


def _hitl_payload(value: BookHitlSnapshot) -> dict[str, Any]:
    return {
        "approval_id": value.approval_id,
        "status": value.status,
        "config_revision": value.config_revision,
    }


def _hitl_from_payload(value: Any) -> BookHitlSnapshot:
    payload = _codec_object(value, {"approval_id", "status", "config_revision"})
    return BookHitlSnapshot(payload["approval_id"], payload["status"], payload["config_revision"])


def _book_cycle_result_payload(value: BookCycleResult) -> dict[str, Any]:
    return {
        "book_id": value.book_id,
        "capital_scope": value.capital_scope,
        "proposal": book_execution_proposal_payload(value.proposal),
        "portfolio_before": _book_portfolio_payload(value.portfolio_before),
        "hitl": _hitl_payload(value.hitl),
        "execution": None if value.execution is None else book_execution_result_payload(value.execution),
        "portfolio_after": (None if value.portfolio_after is None else _book_portfolio_payload(value.portfolio_after)),
        "portfolio_after_available": value.portfolio_after_available,
        "status": value.status,
    }


def _book_cycle_result_from_payload(value: Any) -> BookCycleResult:
    payload = _codec_object(
        value,
        {
            "book_id",
            "capital_scope",
            "proposal",
            "portfolio_before",
            "hitl",
            "execution",
            "portfolio_after",
            "portfolio_after_available",
            "status",
        },
    )
    return BookCycleResult(
        payload["book_id"],
        payload["capital_scope"],
        book_execution_proposal_from_payload(payload["proposal"]),
        _book_portfolio_from_payload(payload["portfolio_before"]),
        _hitl_from_payload(payload["hitl"]),
        None if payload["execution"] is None else book_execution_result_from_payload(payload["execution"]),
        (None if payload["portfolio_after"] is None else _book_portfolio_from_payload(payload["portfolio_after"])),
        payload["portfolio_after_available"],
        payload["status"],
    )


def _book_results_payload(values: tuple[BookCycleResult, ...]) -> dict[str, Any]:
    return {
        "version": _JOURNAL_CODEC_VERSION,
        "items": [_book_cycle_result_payload(item) for item in values],
    }


def _book_results_from_payload(value: Any) -> tuple[BookCycleResult, ...]:
    payload = _codec_object(value, {"version", "items"})
    _require_journal_version(payload["version"])
    return tuple(_book_cycle_result_from_payload(item) for item in _codec_array(payload["items"]))


def _record_payloads(record: MultiVenueCycleRecord) -> tuple[dict[str, Any], ...]:
    return (
        _component_signals_payload(record.component_signals),
        _fused_signal_payload(record.fused_signal),
        _target_position_payload(record.target_position),
        _book_results_payload(record.book_results),
    )


def _multi_venue_record(row: _MultiVenueCycleRow) -> MultiVenueCycleRecord:
    raw_payloads = (row.component_signals, row.fused_signal, row.target_position, row.book_results)
    if _contains_secret_field(raw_payloads):
        raise ValueError("secret field")
    invalid = False
    secret = False
    try:
        created_at = (
            row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at.astimezone(UTC)
        )
        record = MultiVenueCycleRecord(
            row.cycle_id,
            row.config_revision,
            row.market_data_source_id,
            _component_signals_from_payload(row.component_signals),
            _fused_signal_from_payload(row.fused_signal),
            _target_position_from_payload(row.target_position),
            _book_results_from_payload(row.book_results),
            row.cycle_status,
            row.execution_status,
            row.requires_attention,
            created_at,
        )
        if _contains_secret_field(_record_payloads(record)):
            raise ValueError("secret field")
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        secret = str(error) == "secret field"
        invalid = not secret
    if secret:
        raise ValueError("secret field")
    if invalid:
        raise ValueError("stored cycle payload is invalid")
    return record


class MultiVenueCycleStore:
    """只访问 ``multi_venue_cycles`` 的不可变追加式 Journal。"""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url
        self.records: list[MultiVenueCycleRecord] = []
        self._lock = asyncio.Lock()

    async def ensure_table(self) -> None:
        if self.database_url is None or self.database_url in _multi_venue_ready:
            return
        engine = await get_engine(self.database_url)
        async with engine.begin() as connection:
            await connection.run_sync(_MultiVenueBase.metadata.create_all)
        _multi_venue_ready.add(self.database_url)

    async def save(self, record: MultiVenueCycleRecord) -> None:
        if not isinstance(record, MultiVenueCycleRecord):
            raise ValueError("record must be a MultiVenueCycleRecord")
        if any(_contains_secret_field(signal.details) for signal in record.component_signals):
            raise ValueError("secret field")
        component_signals, fused_signal, target_position, book_results = _record_payloads(record)
        if _contains_secret_field((component_signals, fused_signal, target_position, book_results)):
            raise ValueError("secret field")

        if self.database_url is None:
            if any(item.cycle_id == record.cycle_id for item in self.records):
                raise ValueError("cycle already exists")
            self.records.append(record)
            return

        await self.ensure_table()
        session = await get_async_session(self.database_url)
        duplicate = False
        try:
            session.add(
                _MultiVenueCycleRow(
                    cycle_id=record.cycle_id,
                    config_revision=record.config_revision,
                    market_data_source_id=record.market_data_source_id,
                    component_signals=component_signals,
                    fused_signal=fused_signal,
                    target_position=target_position,
                    book_results=book_results,
                    cycle_status=record.cycle_status,
                    execution_status=record.execution_status,
                    requires_attention=record.requires_attention,
                    created_at=record.created_at,
                )
            )
            await session.commit()
        except IntegrityError:
            await session.rollback()
            duplicate = True
        finally:
            await session.close()
        if duplicate:
            raise ValueError("cycle already exists")

    @staticmethod
    def _validate_replacement(current: MultiVenueCycleRecord, replacement: MultiVenueCycleRecord) -> None:
        frozen_current = (
            current.cycle_id,
            current.config_revision,
            current.market_data_source_id,
            current.component_signals,
            current.fused_signal,
            current.target_position,
            current.created_at,
        )
        frozen_replacement = (
            replacement.cycle_id,
            replacement.config_revision,
            replacement.market_data_source_id,
            replacement.component_signals,
            replacement.fused_signal,
            replacement.target_position,
            replacement.created_at,
        )
        if frozen_current != frozen_replacement:
            raise ValueError("frozen cycle identity cannot change")
        current_books = tuple(
            (item.book_id, item.capital_scope, item.proposal, item.portfolio_before) for item in current.book_results
        )
        replacement_books = tuple(
            (item.book_id, item.capital_scope, item.proposal, item.portfolio_before)
            for item in replacement.book_results
        )
        if current_books != replacement_books:
            raise ValueError("frozen book identity cannot change")
        allowed = {
            "ready": {"ready", "completed", "partial", "failed"},
            "awaiting_approval": {
                "awaiting_approval",
                "approval_rejected",
                "completed",
                "partial",
                "failed",
            },
            "approval_rejected": {"approval_rejected"},
            "completed": {"completed"},
            "partial": {"partial"},
            "failed": {"failed"},
        }
        if any(
            new.status not in allowed[old.status]
            for old, new in zip(current.book_results, replacement.book_results, strict=True)
        ):
            raise ValueError("book cycle state cannot regress")

    async def replace(self, record: MultiVenueCycleRecord) -> None:
        """Advance mutable per-book outcomes while keeping cycle evidence frozen."""
        if not isinstance(record, MultiVenueCycleRecord):
            raise ValueError("record must be a MultiVenueCycleRecord")
        component_signals, fused_signal, target_position, book_results = _record_payloads(record)
        if _contains_secret_field((component_signals, fused_signal, target_position, book_results)):
            raise ValueError("secret field")

        if self.database_url is None:
            async with self._lock:
                index = next(
                    (index for index, current in enumerate(self.records) if current.cycle_id == record.cycle_id),
                    None,
                )
                if index is None:
                    raise LookupError("cycle was not found")
                self._validate_replacement(self.records[index], record)
                self.records[index] = record
            return

        await self.ensure_table()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_MultiVenueCycleRow, record.cycle_id)
            if row is None:
                raise LookupError("cycle was not found")
            current = _multi_venue_record(row)
            self._validate_replacement(current, record)
            statement = (
                update(_MultiVenueCycleRow)
                .where(
                    _MultiVenueCycleRow.cycle_id == current.cycle_id,
                    _MultiVenueCycleRow.config_revision == current.config_revision,
                    _MultiVenueCycleRow.market_data_source_id == current.market_data_source_id,
                    _MultiVenueCycleRow.component_signals == _component_signals_payload(current.component_signals),
                    _MultiVenueCycleRow.fused_signal == _fused_signal_payload(current.fused_signal),
                    _MultiVenueCycleRow.target_position == _target_position_payload(current.target_position),
                    _MultiVenueCycleRow.created_at == current.created_at,
                    _MultiVenueCycleRow.book_results == _book_results_payload(current.book_results),
                    _MultiVenueCycleRow.cycle_status == current.cycle_status,
                    _MultiVenueCycleRow.execution_status == current.execution_status,
                    _MultiVenueCycleRow.requires_attention == current.requires_attention,
                )
                .values(
                    book_results=book_results,
                    cycle_status=record.cycle_status,
                    execution_status=record.execution_status,
                    requires_attention=record.requires_attention,
                )
                .returning(_MultiVenueCycleRow)
                .execution_options(populate_existing=True)
            )
            returned = (await session.execute(statement)).scalar_one_or_none()
            if returned is None:
                await session.rollback()
                raise ValueError("cycle changed concurrently")
            persisted = _multi_venue_record(returned)
            if persisted != record:
                await session.rollback()
                raise ValueError("stored cycle payload is invalid")
            await session.commit()
        except (LookupError, ValueError):
            await session.rollback()
            raise
        finally:
            await session.close()

    async def get(self, cycle_id: str) -> MultiVenueCycleRecord | None:
        if self.database_url is None:
            return next((item for item in self.records if item.cycle_id == cycle_id), None)
        await self.ensure_table()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_MultiVenueCycleRow, cycle_id)
        finally:
            await session.close()
        return _multi_venue_record(row) if row is not None else None

    async def list(self, *, limit: int = 100, offset: int = 0) -> list[MultiVenueCycleRecord]:
        if limit < 1 or offset < 0:
            return []
        if self.database_url is None:
            ordered = sorted(self.records, key=lambda item: item.created_at, reverse=True)
            return ordered[offset : offset + limit]
        await self.ensure_table()
        query = select(_MultiVenueCycleRow).order_by(_MultiVenueCycleRow.created_at.desc()).offset(offset).limit(limit)
        session = await get_async_session(self.database_url)
        try:
            rows = (await session.execute(query)).scalars().all()
        finally:
            await session.close()
        return [_multi_venue_record(row) for row in rows]

    async def count(self) -> int:
        if self.database_url is None:
            return len(self.records)
        await self.ensure_table()
        query = select(func.count()).select_from(_MultiVenueCycleRow)
        session = await get_async_session(self.database_url)
        try:
            return int((await session.execute(query)).scalar_one())
        finally:
            await session.close()
