"""Multi-venue Journal persists a closed, credential-safe cycle snapshot."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from cryptotrader.decision.models import TargetPosition
from cryptotrader.execution.models import BookExecutionProposal, BookExecutionResult, ConnectionExecutionResult
from cryptotrader.pair import Pair
from cryptotrader.portfolio.models import BookPortfolioSnapshot, ConnectionPortfolioSnapshot
from cryptotrader.risk.models import BookRiskDecision, ConnectionRiskDecision
from cryptotrader.signals.fusion import ComponentContribution, FusedSignal
from cryptotrader.signals.models import ComponentSignal
from cryptotrader.venues.models import ConnectionPosition
from tests.test_execution_coordinator import _proposal, _result


def _proposal_for(
    book_id: str = "simulation",
    capital_scope: str = "simulated",
    connection_ids: tuple[str, str] = ("sim-first", "sim-second"),
    pair: Pair | None = None,
) -> BookExecutionProposal:
    base = _proposal()
    pair = pair or base.pair
    plans = tuple(
        replace(
            plan,
            book_id=book_id,
            connection_id=connection_id,
            pair=pair,
            quote=replace(plan.quote, pair=pair),
        )
        for plan, connection_id in zip(base.connection_plans, connection_ids, strict=True)
    )
    targets = tuple(
        replace(target, book_id=book_id, connection_id=connection_id)
        for target, connection_id in zip(base.risk.connection_targets, connection_ids, strict=True)
    )
    risk = BookRiskDecision(
        True,
        base.risk.requested_target_exposure,
        base.risk.capped_target_exposure,
        base.risk.connection_weights,
        targets,
    )
    return BookExecutionProposal(
        book_id,
        capital_scope,
        base.config_revision,
        pair,
        base.requested_target_exposure,
        base.target_exposure,
        risk,
        tuple(ConnectionRiskDecision(connection_id, True, True) for connection_id in connection_ids),
        plans,
        (),
        (),
        True,
    )


def _execution(proposal: BookExecutionProposal, status: str = "completed") -> BookExecutionResult:
    if status == "completed":
        results = tuple(_result(proposal, index, "completed") for index in range(2))
    elif status == "failed":
        results = tuple(
            ConnectionExecutionResult.failed(
                plan,
                "place_order",
                execution_quote=plan.quote,
                requires_attention=True,
            )
            for plan in proposal.connection_plans
        )
    else:
        results = (
            _result(proposal, 0, "completed"),
            ConnectionExecutionResult.failed(
                proposal.connection_plans[1],
                "place_order",
                execution_quote=proposal.connection_plans[1].quote,
                requires_attention=True,
            ),
        )
    return BookExecutionResult(
        proposal,
        results,
        status,
        any(result.requires_attention for result in results),
    )


def _portfolio(proposal: BookExecutionProposal, *, after: bool = False) -> BookPortfolioSnapshot:
    connections = tuple(
        ConnectionPortfolioSnapshot(
            plan.connection_id,
            Decimal("50"),
            {"USDT": Decimal("50")},
            ConnectionPosition(
                proposal.pair,
                plan.target_signed_amount if after else Decimal("0"),
                plan.target_signed_notional if after else Decimal("0"),
                plan.quote.last if after else None,
            ),
        )
        for plan in proposal.connection_plans
    )
    return BookPortfolioSnapshot(
        proposal.book_id,
        proposal.capital_scope,
        Decimal("100"),
        sum((item.position.signed_notional for item in connections), Decimal("0")),
        connections,
    )


def _hitl(proposal: BookExecutionProposal, status: str, approval_id: str | None = None):
    from cryptotrader.journal.models import BookHitlSnapshot

    return BookHitlSnapshot(approval_id, status, proposal.config_revision)


def _book_cycle(
    *,
    book_id: str = "simulation",
    capital_scope: str = "simulated",
    connection_ids: tuple[str, str] = ("sim-first", "sim-second"),
    status: str = "partial",
    pair: Pair | None = None,
):
    from cryptotrader.journal.models import BookCycleResult

    proposal = _proposal_for(book_id, capital_scope, connection_ids, pair)
    before = _portfolio(proposal)
    if status == "ready":
        return BookCycleResult(
            book_id,
            capital_scope,
            proposal,
            before,
            _hitl(proposal, "not_required"),
            None,
            None,
            None,
            status,
        )
    if status == "awaiting_approval":
        return BookCycleResult(
            book_id,
            capital_scope,
            proposal,
            before,
            _hitl(proposal, "pending", f"approval-{book_id}"),
            None,
            None,
            None,
            status,
        )
    if status == "approval_rejected":
        return BookCycleResult(
            book_id,
            capital_scope,
            proposal,
            before,
            _hitl(proposal, "rejected", f"approval-{book_id}"),
            None,
            None,
            None,
            status,
        )
    execution = _execution(proposal, status)
    hitl = (
        _hitl(proposal, "executed", f"approval-{book_id}")
        if capital_scope == "real"
        else _hitl(proposal, "not_required")
    )
    return BookCycleResult(
        book_id,
        capital_scope,
        proposal,
        before,
        hitl,
        execution,
        _portfolio(proposal, after=True),
        True,
        status,
    )


def _signals(details=None):
    return (
        ComponentSignal(
            "kronos",
            "long",
            0.8,
            "model evidence",
            details
            if details is not None
            else {
                "window": 512,
                "threshold": Decimal("0.10"),
                "as_of": datetime(2026, 8, 29, 1, tzinfo=UTC),
                "pair": Pair.parse("BTC/USDT:USDT"),
                "direction": "long",
            },
        ),
        ComponentSignal("llm_committee", "short", 0.2, "committee evidence"),
    )


def _fused():
    return FusedSignal(
        0.30000000000000004,
        (
            ComponentContribution("kronos", 0.5, 0.8, 0.4),
            ComponentContribution("llm_committee", 0.5, -0.2, -0.1),
        ),
        "weighted evidence",
    )


def _record(
    *,
    cycle_id: str = "cycle-1",
    details=None,
    book_results=None,
    cycle_status: str = "partial",
    execution_status: str = "partial",
    requires_attention: bool = True,
):
    from cryptotrader.journal.models import MultiVenueCycleRecord

    books = (_book_cycle(),) if book_results is None else book_results
    return MultiVenueCycleRecord(
        cycle_id=cycle_id,
        config_revision=9,
        market_data_source_id="market-primary",
        component_signals=_signals(details),
        fused_signal=_fused(),
        target_position=TargetPosition("long", 0.3),
        book_results=books,
        cycle_status=cycle_status,
        execution_status=execution_status,
        requires_attention=requires_attention,
        created_at=datetime(2026, 8, 29, 2, tzinfo=UTC),
    )


def test_book_cycle_result_closes_proposal_hitl_execution_and_portfolios():
    from cryptotrader.journal.models import BookCycleResult

    awaiting = _book_cycle(status="awaiting_approval")
    with pytest.raises(ValueError, match="pending"):
        replace(awaiting, hitl=_hitl(awaiting.proposal, "not_required"))
    with pytest.raises(ValueError, match="execution"):
        replace(awaiting, execution=_execution(awaiting.proposal))

    completed = _book_cycle(status="completed")
    with pytest.raises(ValueError, match="proposal"):
        replace(completed, execution=_execution(_proposal_for("other", "simulated", ("a", "b"))))
    with pytest.raises(ValueError, match="portfolio_after"):
        replace(completed, portfolio_after=None)
    with pytest.raises(ValueError, match="portfolio_after"):
        BookCycleResult(
            completed.book_id,
            completed.capital_scope,
            completed.proposal,
            completed.portfolio_before,
            completed.hitl,
            completed.execution,
            completed.portfolio_after,
            False,
            "completed",
        )


def test_multi_venue_cycle_record_is_immutable_and_target_ratio_is_exact_finite_float():
    record = _record()
    with pytest.raises(FrozenInstanceError):
        record.cycle_status = "failed"
    with pytest.raises(ValueError, match="cycle_id"):
        replace(record, cycle_id="")
    with pytest.raises(ValueError, match="UTC"):
        replace(record, created_at=record.created_at.replace(tzinfo=None))
    for invalid in (True, 1, Decimal("0.3"), float("nan")):
        with pytest.raises(ValueError, match="size_ratio"):
            replace(record, target_position=TargetPosition("long", invalid))


def test_multi_venue_cycle_record_closes_books_pairs_and_connection_ids():
    record = _record()
    first = _book_cycle(book_id="simulation", connection_ids=("one", "two"), status="completed")
    mixed_pair = _book_cycle(
        book_id="live",
        capital_scope="real",
        connection_ids=("three", "four"),
        status="completed",
        pair=Pair.parse("ETH/USDT:USDT"),
    )
    with pytest.raises(ValueError, match="pair"):
        replace(record, book_results=(first, mixed_pair), cycle_status="completed", execution_status="completed")
    with pytest.raises(ValueError, match="book IDs"):
        replace(record, book_results=(first, first), cycle_status="completed", execution_status="completed")
    reused_connection = _book_cycle(
        book_id="live",
        capital_scope="real",
        connection_ids=("one", "four"),
        status="completed",
    )
    with pytest.raises(ValueError, match="connection IDs"):
        replace(
            record,
            book_results=(first, reused_connection),
            cycle_status="completed",
            execution_status="completed",
        )


def test_multi_venue_cycle_record_derives_mixed_pre_and_post_execution_aggregates():
    sim = _book_cycle(status="completed")
    live = _book_cycle(
        book_id="live",
        capital_scope="real",
        connection_ids=("live-first", "live-second"),
        status="awaiting_approval",
    )
    record = _record(
        book_results=(sim, live),
        cycle_status="awaiting_approval",
        execution_status="partial",
        requires_attention=False,
    )
    assert record.cycle_status == "awaiting_approval"
    with pytest.raises(ValueError, match="cycle_status"):
        replace(record, cycle_status="partial")
    with pytest.raises(ValueError, match="execution_status"):
        replace(record, execution_status="completed")


def test_multi_venue_cycle_record_closes_signal_contributions_and_math():
    record = _record()
    duplicate_components = (_signals()[0], replace(_signals()[1], component_id="kronos"))
    with pytest.raises(ValueError, match="component IDs"):
        replace(record, component_signals=duplicate_components)
    with pytest.raises(ValueError, match="contribution IDs"):
        replace(record, fused_signal=replace(_fused(), contributions=tuple(reversed(_fused().contributions))))
    bad_weighted = replace(
        _fused(),
        contributions=(replace(_fused().contributions[0], weighted_score=0.41), _fused().contributions[1]),
        score=0.31,
    )
    with pytest.raises(ValueError, match="weighted_score"):
        replace(record, fused_signal=bad_weighted)
    with pytest.raises(ValueError, match="fused score"):
        replace(record, fused_signal=replace(_fused(), score=0.4))


def test_multi_venue_cycle_record_supports_only_simple_empty_book_terminal_matrix():
    record = _record(
        book_results=(),
        cycle_status="no_change",
        execution_status="not_started",
        requires_attention=False,
    )
    assert record.book_results == ()
    for status in ("component_failed", "cycle_failed", "risk_rejected", "cancelled"):
        assert replace(record, cycle_status=status).cycle_status == status
    with pytest.raises(ValueError, match="cycle_status"):
        replace(record, cycle_status="completed")


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_multi_venue_cycle_round_trip_preserves_awaiting_hitl_identity(tmp_path, database):
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'cycles.db'}" if database else None)
    live = _book_cycle(
        book_id="live",
        capital_scope="real",
        connection_ids=("live-first", "live-second"),
        status="awaiting_approval",
    )
    record = _record(
        cycle_id=f"cycle-{database}",
        book_results=(live,),
        cycle_status="awaiting_approval",
        execution_status="not_started",
        requires_attention=False,
    )

    await store.save(record)
    loaded = await store.get(record.cycle_id)

    assert loaded == record
    assert loaded is not None
    assert loaded.book_results[0].hitl.approval_id == "approval-live"
    assert loaded.book_results[0].proposal == live.proposal


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_multi_venue_cycle_round_trip_preserves_execution_and_portfolios(tmp_path, database):
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'terminal.db'}" if database else None)
    record = _record(cycle_id=f"cycle-terminal-{database}")

    await store.save(record)
    loaded = await store.get(record.cycle_id)

    assert loaded == record
    assert loaded is not None
    assert loaded.book_results[0].execution is not None
    assert loaded.book_results[0].execution.target_weights == (Decimal("0.4"), Decimal("0.6"))
    assert loaded.book_results[0].portfolio_before.total_signed_notional == Decimal("0")
    assert loaded.book_results[0].portfolio_after is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret_key",
    [
        "api_key",
        "ApiSecret",
        "SECRET",
        "pass-phrase",
        "password",
        "accessToken",
        "credential_ref",
        "privateKey",
    ],
)
async def test_journal_serialization_recursively_rejects_secret_shaped_keys(secret_key):
    from cryptotrader.journal.store import MultiVenueCycleStore

    record = _record(details={"outer": [{"safe": {secret_key: "RAW_SECRET_MARKER"}}]})

    with pytest.raises(ValueError, match=r"^secret field$") as captured:
        await MultiVenueCycleStore().save(record)
    assert "RAW_SECRET_MARKER" not in repr(captured.value)
    assert secret_key not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["secret", "bool_version"])
async def test_journal_read_path_rejects_secrets_and_bool_codec_version(tmp_path, corruption):
    from cryptotrader.journal.store import MultiVenueCycleStore

    path = tmp_path / "read-corrupt.db"
    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{path}")
    record = _record()
    await store.save(record)
    column = "component_signals" if corruption == "secret" else "book_results"
    with sqlite3.connect(path) as connection:
        payload = json.loads(
            connection.execute(
                f"SELECT {column} FROM multi_venue_cycles WHERE cycle_id = ?",  # noqa: S608
                (record.cycle_id,),
            ).fetchone()[0]
        )
        if corruption == "secret":
            payload["api_key"] = "RAW_SECRET_MARKER"  # pragma: allowlist secret
        else:
            payload["version"] = True
        connection.execute(
            f"UPDATE multi_venue_cycles SET {column} = ? WHERE cycle_id = ?",  # noqa: S608
            (json.dumps(payload), record.cycle_id),
        )

    expected = "secret field" if corruption == "secret" else "stored cycle payload"
    with pytest.raises(ValueError, match=expected) as captured:
        await store.get(record.cycle_id)
    assert "RAW_SECRET_MARKER" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_multi_venue_store_schema_is_exact_and_ignores_legacy_trading_cycles(tmp_path):
    from cryptotrader.journal.store import MultiVenueCycleStore

    path = tmp_path / "coexist.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE trading_cycles (cycle_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("INSERT INTO trading_cycles VALUES ('legacy-cycle', 'RAW_LEGACY_PAYLOAD')")

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{path}")
    record = _record()
    await store.save(record)

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(multi_venue_cycles)")}
        legacy = connection.execute("SELECT payload FROM trading_cycles").fetchone()[0]

    assert columns == {
        "cycle_id",
        "config_revision",
        "market_data_source_id",
        "component_signals",
        "fused_signal",
        "target_position",
        "book_results",
        "cycle_status",
        "execution_status",
        "requires_attention",
        "created_at",
    }
    assert legacy == "RAW_LEGACY_PAYLOAD"
    assert await store.get("legacy-cycle") is None
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_multi_venue_store_lists_newest_first_and_counts_only_new_records():
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore()
    first = _record(cycle_id="first")
    second = replace(_record(cycle_id="second"), created_at=first.created_at.replace(microsecond=1))
    await store.save(first)
    await store.save(second)

    assert [record.cycle_id for record in await store.list(limit=1)] == ["second"]
    assert [record.cycle_id for record in await store.list(limit=1, offset=1)] == ["first"]
    assert await store.count() == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_duplicate_cycle_fails_without_database_cause_or_context(tmp_path, database):
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'duplicate.db'}" if database else None)
    record = _record()
    await store.save(record)

    with pytest.raises(ValueError, match="cycle already exists") as captured:
        await store.save(record)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_corrupt_multi_venue_payload_fails_closed_without_raw_context(tmp_path):
    from cryptotrader.journal.store import MultiVenueCycleStore

    path = tmp_path / "corrupt.db"
    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{path}")
    record = _record()
    await store.save(record)

    with sqlite3.connect(path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT book_results FROM multi_venue_cycles WHERE cycle_id = ?",
                (record.cycle_id,),
            ).fetchone()[0]
        )
        payload["items"][0]["book_id"] = "RAW_SECRET_PAYLOAD"
        connection.execute(
            "UPDATE multi_venue_cycles SET book_results = ? WHERE cycle_id = ?",
            (json.dumps(payload), record.cycle_id),
        )

    with pytest.raises(ValueError, match="stored cycle payload") as captured:
        await store.get(record.cycle_id)
    assert "RAW_SECRET_PAYLOAD" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_replace_advances_same_cycle_after_live_approval(tmp_path, database):
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'replace.db'}" if database else None)
    sim = _book_cycle(status="completed")
    live = _book_cycle(
        book_id="live",
        capital_scope="real",
        connection_ids=("live-first", "live-second"),
        status="awaiting_approval",
    )
    initial = _record(
        book_results=(sim, live),
        cycle_status="awaiting_approval",
        execution_status="partial",
        requires_attention=False,
    )
    await store.save(initial)
    completed_live = _book_cycle(
        book_id="live",
        capital_scope="real",
        connection_ids=("live-first", "live-second"),
        status="completed",
    )
    advanced = replace(
        initial,
        book_results=(sim, completed_live),
        cycle_status="completed",
        execution_status="completed",
        requires_attention=False,
    )

    await store.replace(advanced)

    assert await store.get(initial.cycle_id) == advanced
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_sqlite_concurrent_replace_allows_only_one_progression(tmp_path):
    from cryptotrader.journal.store import MultiVenueCycleStore

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'replace-race.db'}"
    creator = MultiVenueCycleStore(database_url)
    sim = _book_cycle(status="completed")
    live = _book_cycle(
        book_id="live",
        capital_scope="real",
        connection_ids=("live-first", "live-second"),
        status="awaiting_approval",
    )
    initial = _record(
        book_results=(sim, live),
        cycle_status="awaiting_approval",
        execution_status="partial",
        requires_attention=False,
    )
    await creator.save(initial)
    rejected = replace(
        initial,
        book_results=(
            sim,
            _book_cycle(
                book_id="live",
                capital_scope="real",
                connection_ids=("live-first", "live-second"),
                status="approval_rejected",
            ),
        ),
        cycle_status="approval_rejected",
    )
    completed = replace(
        initial,
        book_results=(
            sim,
            _book_cycle(
                book_id="live",
                capital_scope="real",
                connection_ids=("live-first", "live-second"),
                status="completed",
            ),
        ),
        cycle_status="completed",
        execution_status="completed",
    )

    outcomes = await asyncio.gather(
        MultiVenueCycleStore(database_url).replace(rejected),
        MultiVenueCycleStore(database_url).replace(completed),
        return_exceptions=True,
    )

    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
    assert await creator.get(initial.cycle_id) in (rejected, completed)


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_replace_rejects_frozen_identity_change_without_mutating_original(tmp_path, database):
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'guard.db'}" if database else None)
    original = _record()
    await store.save(original)
    changed = replace(original, market_data_source_id="other-source")

    with pytest.raises(ValueError, match="frozen cycle identity") as captured:
        await store.replace(changed)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert await store.get(original.cycle_id) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_replace_requires_existing_cycle(tmp_path, database):
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'missing.db'}" if database else None)
    with pytest.raises(LookupError, match="cycle was not found"):
        await store.replace(_record())
