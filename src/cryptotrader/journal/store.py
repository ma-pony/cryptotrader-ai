"""Canonical multi-venue decision persistence for PostgreSQL and SQLite."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, String, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cryptotrader.db import get_async_session
from cryptotrader.decision.analysis import AnalysisFailure
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
    BookPreparationFailure,
    DecisionRun,
    MultiVenueCycleRecord,
)
from cryptotrader.migrations.schema import require_tables
from cryptotrader.pair import Pair
from cryptotrader.portfolio.models import BookPortfolioSnapshot, ConnectionPortfolioSnapshot
from cryptotrader.signals.fusion import ComponentContribution, FusedSignal
from cryptotrader.signals.models import ComponentSignal
from cryptotrader.venues.models import ConnectionPosition

_multi_venue_ready: set[str] = set()


class JournalPersistenceError(RuntimeError):
    """A redacted failure at the new Journal database write boundary."""


async def _safe_rollback(session: Any) -> bool:
    try:
        await session.rollback()
    except SQLAlchemyError:
        return False
    return True


async def _safe_close(session: Any) -> bool:
    try:
        await session.close()
    except Exception:
        return False
    return True


async def _write_session(database_url: str) -> Any:
    session = None
    with suppress(SQLAlchemyError):
        session = await get_async_session(database_url)
    if session is None:
        raise JournalPersistenceError("journal persistence failed")
    return session


class _MultiVenueBase(DeclarativeBase):
    pass


class _MultiVenueCycleRow(_MultiVenueBase):
    __tablename__ = "multi_venue_cycles"

    cycle_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_metadata: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=False)
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


def _contains_journal_secret(payloads: tuple) -> bool:
    """Exclude only validated usage counters at the canonical signal path."""
    signals, *others = payloads
    if not isinstance(signals, Mapping) or not isinstance(signals.get("items"), list):
        return _contains_secret_field(payloads)
    scanned = []
    for signal in signals["items"]:
        if isinstance(signal, Mapping):
            usage = signal.get("usage")
            if (
                isinstance(usage, Mapping)
                and set(usage) == {"input_tokens", "output_tokens"}
                and all(type(count) is int and count >= 0 for count in usage.values())
            ):
                signal = {**signal, "usage": None}
        scanned.append(signal)
    return _contains_secret_field(({**signals, "items": scanned}, *others))


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
        "blocks": [block.model_dump(mode="json") for block in value.blocks],
        "evaluation_reference": value.evaluation_reference.model_dump(mode="json")
        if value.evaluation_reference
        else None,
        "status": value.status,
        "duration_ms": value.duration_ms,
        "usage": value.usage.model_dump(mode="json") if value.usage else None,
        "cost": str(value.cost) if value.cost is not None else None,
    }


def _component_signal_from_payload(value: Any) -> ComponentSignal:
    payload = _codec_object(
        value,
        {
            "component_id",
            "direction",
            "confidence",
            "reasoning",
            "details",
            "blocks",
            "evaluation_reference",
            "status",
            "duration_ms",
            "usage",
            "cost",
        },
    )
    details = _detail_from_payload(payload["details"])
    if not isinstance(details, Mapping):
        raise ValueError("invalid component signal details")
    return ComponentSignal(
        payload["component_id"],
        payload["direction"],
        payload["confidence"],
        payload["reasoning"],
        details,
        blocks=payload["blocks"],
        evaluation_reference=payload["evaluation_reference"],
        status=payload["status"],
        duration_ms=payload["duration_ms"],
        usage=payload["usage"],
        cost=_detail_decimal_from_payload(payload["cost"]) if payload["cost"] is not None else None,
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
    result = {
        "connection_id": value.connection_id,
        "equity": None if value.equity is None else str(value.equity),
        "balances": [[asset, str(amount)] for asset, amount in value.balances.items()],
        "position": {
            "pair": value.position.pair.canonical(),
            "signed_amount": str(value.position.signed_amount),
            "signed_notional": str(value.position.signed_notional),
            "entry_price": None if value.position.entry_price is None else str(value.position.entry_price),
        },
    }
    if value.account_snapshot is not None:
        from cryptotrader.accounts.store import payload

        result["account_snapshot"] = payload(value.account_snapshot)
    return result


def _decimal_from_payload(value: Any) -> Decimal:
    if type(value) is not str:
        raise ValueError("invalid journal decimal")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("invalid journal decimal")
    return result


def _connection_portfolio_from_payload(value: Any) -> ConnectionPortfolioSnapshot:
    keys = {"connection_id", "equity", "balances", "position"}
    if isinstance(value, dict) and "account_snapshot" in value:
        keys.add("account_snapshot")
    payload = _codec_object(value, keys)
    from cryptotrader.accounts.store import snapshot_from_payload

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
        None if payload["equity"] is None else _decimal_from_payload(payload["equity"]),
        balances,
        ConnectionPosition(
            Pair.parse(position_payload["pair"]),
            _decimal_from_payload(position_payload["signed_amount"]),
            _decimal_from_payload(position_payload["signed_notional"]),
            None if entry_price is None else _decimal_from_payload(entry_price),
        ),
        snapshot_from_payload(payload["account_snapshot"]) if payload.get("account_snapshot") is not None else None,
    )


def _book_portfolio_payload(value: BookPortfolioSnapshot) -> dict[str, Any]:
    return {
        "book_id": value.book_id,
        "capital_scope": value.capital_scope,
        "total_equity": None if value.total_equity is None else str(value.total_equity),
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
        None if payload["total_equity"] is None else _decimal_from_payload(payload["total_equity"]),
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
        "config_revision": value.config_revision,
        "pair": value.pair.canonical(),
        "proposal": None if value.proposal is None else book_execution_proposal_payload(value.proposal),
        "portfolio_before": (
            None if value.portfolio_before is None else _book_portfolio_payload(value.portfolio_before)
        ),
        "hitl": _hitl_payload(value.hitl),
        "execution": None if value.execution is None else book_execution_result_payload(value.execution),
        "failure": None if value.failure is None else {"stage": value.failure.stage},
        "portfolio_after": (None if value.portfolio_after is None else _book_portfolio_payload(value.portfolio_after)),
        "portfolio_after_available": value.portfolio_after_available,
        "reconciliation_required": value.reconciliation_required,
        "status": value.status,
    }


def _book_cycle_result_from_payload(value: Any) -> BookCycleResult:
    payload = _codec_object(
        value,
        {
            "book_id",
            "capital_scope",
            "config_revision",
            "pair",
            "proposal",
            "portfolio_before",
            "hitl",
            "execution",
            "failure",
            "portfolio_after",
            "portfolio_after_available",
            "reconciliation_required",
            "status",
        },
    )
    if type(payload["pair"]) is not str:
        raise ValueError("invalid book cycle pair")
    failure_payload = payload["failure"]
    failure = None
    if failure_payload is not None:
        failure = BookPreparationFailure(_codec_object(failure_payload, {"stage"})["stage"])
    return BookCycleResult(
        book_id=payload["book_id"],
        capital_scope=payload["capital_scope"],
        config_revision=payload["config_revision"],
        pair=Pair.parse(payload["pair"]),
        proposal=(None if payload["proposal"] is None else book_execution_proposal_from_payload(payload["proposal"])),
        portfolio_before=(
            None if payload["portfolio_before"] is None else _book_portfolio_from_payload(payload["portfolio_before"])
        ),
        hitl=_hitl_from_payload(payload["hitl"]),
        execution=(None if payload["execution"] is None else book_execution_result_from_payload(payload["execution"])),
        failure=failure,
        portfolio_after=(
            None if payload["portfolio_after"] is None else _book_portfolio_from_payload(payload["portfolio_after"])
        ),
        portfolio_after_available=payload["portfolio_after_available"],
        reconciliation_required=payload["reconciliation_required"],
        status=payload["status"],
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


def _run_payload(run: DecisionRun) -> dict[str, Any]:
    return {
        "version": 1,
        "pair": run.pair,
        "mode": run.mode,
        "origin": run.origin,
        "config_snapshot": _detail_payload(run.config_snapshot),
        "finished_at": None if run.finished_at is None else run.finished_at.isoformat(),
        "failure": None if run.failure is None else run.failure.model_dump(),
        "incomplete_fields": list(run.incomplete_fields),
    }


def _run_from_payload(value: Any) -> DecisionRun:
    payload = _codec_object(
        value, {"version", "pair", "mode", "origin", "config_snapshot", "finished_at", "failure", "incomplete_fields"}
    )
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("unsupported decision metadata version")
    return DecisionRun(
        payload["pair"],
        payload["mode"],
        payload["origin"],
        _detail_mapping_from_payload(payload["config_snapshot"]),
        None if payload["finished_at"] is None else _detail_datetime_from_payload(payload["finished_at"]),
        None if payload["failure"] is None else AnalysisFailure.model_validate(payload["failure"]),
        tuple(_codec_array(payload["incomplete_fields"])),
    )


def _contains_secret_balance_key(record: MultiVenueCycleRecord) -> bool:
    snapshots = (
        snapshot
        for book in record.book_results
        for snapshot in (book.portfolio_before, book.portfolio_after)
        if snapshot is not None
    )
    return any(
        _is_secret_field(asset)
        for snapshot in snapshots
        for connection in snapshot.connections
        for asset in connection.balances
    )


def _validated_record_payloads(record: MultiVenueCycleRecord) -> tuple[dict[str, Any], ...]:
    if not isinstance(record, MultiVenueCycleRecord):
        raise ValueError("record must be a MultiVenueCycleRecord")
    if any(
        _contains_secret_field(signal.details) for signal in record.component_signals
    ) or _contains_secret_balance_key(record):
        raise ValueError("secret field")
    payloads = _record_payloads(record)
    if _contains_journal_secret(payloads) or _contains_secret_field(record.run.config_snapshot):
        raise ValueError("secret field")
    return payloads


def _multi_venue_record(row: _MultiVenueCycleRow) -> MultiVenueCycleRecord:
    raw_payloads = (row.component_signals, row.fused_signal, row.target_position, row.book_results)
    if _contains_journal_secret(raw_payloads):
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
            _run_from_payload(row.run_metadata),
        )
        if (
            _contains_journal_secret(_record_payloads(record))
            or _contains_secret_balance_key(record)
            or _contains_secret_field(record.run.config_snapshot)
        ):
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
        await require_tables(self.database_url, _MultiVenueBase.metadata.tables)
        _multi_venue_ready.add(self.database_url)

    async def _ensure_write_table(self) -> None:
        failed = False
        try:
            await self.ensure_table()
        except SQLAlchemyError:
            failed = True
        if failed:
            raise JournalPersistenceError("journal persistence failed")

    async def save(self, record: MultiVenueCycleRecord) -> None:
        payloads = _validated_record_payloads(record)

        if self.database_url is None:
            if any(item.cycle_id == record.cycle_id for item in self.records):
                raise ValueError("cycle already exists")
            self.records.append(record)
            return
        await self._save_database(record, payloads)

    @staticmethod
    def stage_records(session, records) -> None:
        """Publish immutable replay facts in their result owner's DB transaction."""
        for record in records:
            signals, fused, target, books = _validated_record_payloads(record)
            if record.run.mode != "backtest":
                raise ValueError("research publication only accepts backtest decisions")
            session.add(
                _MultiVenueCycleRow(
                    cycle_id=record.cycle_id,
                    run_metadata=_run_payload(record.run),
                    config_revision=record.config_revision,
                    market_data_source_id=record.market_data_source_id,
                    component_signals=signals,
                    fused_signal=fused,
                    target_position=target,
                    book_results=books,
                    cycle_status=record.cycle_status,
                    execution_status=record.execution_status,
                    requires_attention=record.requires_attention,
                    created_at=record.created_at,
                )
            )

    async def _save_database(
        self,
        record: MultiVenueCycleRecord,
        payloads: tuple[dict[str, Any], ...],
    ) -> None:
        await self._ensure_write_table()
        session = await _write_session(self.database_url)
        operation_completed = False
        try:
            component_signals, fused_signal, target_position, book_results = payloads
            row = _MultiVenueCycleRow(
                cycle_id=record.cycle_id,
                run_metadata=_run_payload(record.run),
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
            outcome = await self._insert_outcome(session, row, record.cycle_id)
            operation_completed = True
        finally:
            if not await _safe_close(session) and operation_completed:
                raise JournalPersistenceError("journal persistence failed") from None
        if outcome == "failed":
            raise JournalPersistenceError("journal persistence failed")
        if outcome == "duplicate":
            raise ValueError("cycle already exists")

    @staticmethod
    async def _insert_outcome(session: Any, row: _MultiVenueCycleRow, cycle_id: str) -> str:
        try:
            session.add(row)
            await session.commit()
        except IntegrityError:
            if not await _safe_rollback(session):
                return "failed"
            try:
                duplicate = await session.get(_MultiVenueCycleRow, cycle_id) is not None
            except SQLAlchemyError:
                return "failed"
            return "duplicate" if duplicate else "failed"
        except SQLAlchemyError:
            await _safe_rollback(session)
            return "failed"
        return "saved"

    @staticmethod
    def _validate_replacement(current: MultiVenueCycleRecord, replacement: MultiVenueCycleRecord) -> None:
        if current.cycle_status in {"queued", "running"}:
            if (
                current.cycle_id,
                current.config_revision,
                current.created_at,
                current.market_data_source_id,
                replace(current.run, finished_at=None, failure=None),
            ) != (
                replacement.cycle_id,
                replacement.config_revision,
                replacement.created_at,
                replacement.market_data_source_id,
                replace(replacement.run, finished_at=None, failure=None),
            ):
                raise ValueError("frozen run identity cannot change")
            if replacement.cycle_status == "queued" or (
                current.cycle_status == "running" and replacement.cycle_status == "running"
            ):
                raise ValueError("invalid run transition")
            return
        if MultiVenueCycleStore._frozen_cycle_identity(current) != MultiVenueCycleStore._frozen_cycle_identity(
            replacement
        ):
            raise ValueError("frozen cycle identity cannot change")
        current_books = tuple(MultiVenueCycleStore._frozen_book_identity(item) for item in current.book_results)
        replacement_books = tuple(MultiVenueCycleStore._frozen_book_identity(item) for item in replacement.book_results)
        if current_books != replacement_books:
            raise ValueError("frozen book identity cannot change")
        for old, new in zip(current.book_results, replacement.book_results, strict=True):
            MultiVenueCycleStore._validate_book_transition(old, new)

    @staticmethod
    def _frozen_cycle_identity(record: MultiVenueCycleRecord) -> tuple[Any, ...]:
        return (
            record.cycle_id,
            record.config_revision,
            record.market_data_source_id,
            record.component_signals,
            record.fused_signal,
            record.target_position,
            record.created_at,
            record.run,
        )

    @staticmethod
    def _frozen_book_identity(item: BookCycleResult) -> tuple[Any, ...]:
        return (
            item.book_id,
            item.capital_scope,
            item.config_revision,
            item.pair,
            item.proposal,
            item.portfolio_before,
        )

    @staticmethod
    def _validate_book_transition(old: BookCycleResult, new: BookCycleResult) -> None:
        if old == new:
            return
        if old.failure is not None or old.status in {"approval_rejected", "completed", "partial", "failed"}:
            raise ValueError("terminal book audit facts are immutable")
        if old.status == "ready":
            if new.status not in {"completed", "partial", "failed"} or new.hitl != old.hitl:
                raise ValueError("ready book may only advance to execution with unchanged HITL identity")
            return
        if old.status != "awaiting_approval" or new.status not in {
            "approval_rejected",
            "completed",
            "partial",
            "failed",
        }:
            raise ValueError("same-state replacement must be completely idempotent")
        if old.hitl.approval_id != new.hitl.approval_id or old.hitl.config_revision != new.hitl.config_revision:
            raise ValueError("approval identity must not change")

    async def replace(self, record: MultiVenueCycleRecord) -> None:
        """Advance mutable per-book outcomes while keeping cycle evidence frozen."""
        payloads = _validated_record_payloads(record)

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
        await self._replace_database(record, payloads)

    async def _replace_database(
        self,
        record: MultiVenueCycleRecord,
        payloads: tuple[dict[str, Any], ...],
    ) -> None:
        await self._ensure_write_table()
        session = await _write_session(self.database_url)
        operation_completed = False
        try:
            outcome = await self._replace_outcome(session, record, payloads[3])
            operation_completed = True
        finally:
            if not await _safe_close(session) and operation_completed:
                raise JournalPersistenceError("journal persistence failed") from None
        if outcome == "saved":
            return
        errors = {
            "missing": LookupError("cycle was not found"),
            "concurrent": ValueError("cycle changed concurrently"),
            "invalid": ValueError("stored cycle payload is invalid"),
            "failed": JournalPersistenceError("journal persistence failed"),
        }
        raise errors[outcome]

    async def _replace_outcome(
        self,
        session: Any,
        record: MultiVenueCycleRecord,
        book_results: dict[str, Any],
    ) -> str:
        try:
            row = await session.get(_MultiVenueCycleRow, record.cycle_id)
            if row is None:
                await _safe_rollback(session)
                return "missing"
            current = _multi_venue_record(row)
            self._validate_replacement(current, record)
            statement = self._replace_statement(current, record, book_results)
            returned = (await session.execute(statement)).scalar_one_or_none()
            if returned is None:
                await _safe_rollback(session)
                return "concurrent"
            persisted = _multi_venue_record(returned)
            if persisted != record:
                await _safe_rollback(session)
                return "invalid"
            await session.commit()
        except (LookupError, ValueError):
            await _safe_rollback(session)
            raise
        except SQLAlchemyError:
            await _safe_rollback(session)
            return "failed"
        return "saved"

    @staticmethod
    def _replace_statement(
        current: MultiVenueCycleRecord,
        replacement: MultiVenueCycleRecord,
        book_results: dict[str, Any],
    ) -> Any:
        return (
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
                _MultiVenueCycleRow.run_metadata == _run_payload(current.run),
            )
            .values(
                book_results=book_results,
                cycle_status=replacement.cycle_status,
                execution_status=replacement.execution_status,
                requires_attention=replacement.requires_attention,
                run_metadata=_run_payload(replacement.run),
                component_signals=_component_signals_payload(replacement.component_signals),
                fused_signal=_fused_signal_payload(replacement.fused_signal),
                target_position=_target_position_payload(replacement.target_position),
            )
            .returning(_MultiVenueCycleRow)
            .execution_options(populate_existing=True)
        )

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

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        pair=None,
        mode=None,
        origin=None,
        revision=None,
        started_at=None,
        ended_at=None,
        component_id=None,
    ) -> list[MultiVenueCycleRecord]:
        if limit < 1 or offset < 0:
            return []
        ordered = await self._filtered(
            pair=pair,
            mode=mode,
            origin=origin,
            revision=revision,
            started_at=started_at,
            ended_at=ended_at,
            component_id=component_id,
        )
        return ordered[offset : offset + limit]

    async def _filtered(self, **filters):
        if self.database_url is None:
            records = self.records
        else:
            records = await self._all_records()

        def matches(item):
            return (
                all(
                    value is None or actual == value
                    for actual, value in (
                        (item.run.pair, filters.get("pair")),
                        (item.run.mode, filters.get("mode")),
                        (item.run.origin, filters.get("origin")),
                        (item.config_revision, filters.get("revision")),
                    )
                )
                and (filters.get("started_at") is None or item.created_at >= filters["started_at"])
                and (filters.get("ended_at") is None or item.created_at <= filters["ended_at"])
                and (
                    filters.get("component_id") is None
                    or any(signal.component_id == filters["component_id"] for signal in item.component_signals)
                )
            )

        return sorted((item for item in records if matches(item)), key=lambda item: item.created_at, reverse=True)

    async def _all_records(self):
        await self.ensure_table()
        query = select(_MultiVenueCycleRow).order_by(_MultiVenueCycleRow.created_at.desc())
        session = await get_async_session(self.database_url)
        try:
            rows = (await session.execute(query)).scalars().all()
        finally:
            await session.close()
        return [_multi_venue_record(row) for row in rows]

    async def count(self, **filters) -> int:
        return len(await self._filtered(**filters))

    async def interrupt_unfinished(self, finished_at: datetime) -> None:
        for record in await self._filtered():
            if record.cycle_status in {"queued", "running"}:
                await self.replace(
                    replace(
                        record,
                        cycle_status="interrupted",
                        run=replace(
                            record.run,
                            finished_at=finished_at,
                            failure=AnalysisFailure(
                                code="interrupted", stage="runtime", message="服务已重启。未自动恢复执行。"
                            ),
                        ),
                    )
                )
