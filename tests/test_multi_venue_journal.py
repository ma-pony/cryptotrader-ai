"""Multi-venue Journal persists a closed, credential-safe cycle snapshot."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy.exc import SQLAlchemyError

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


def _rejected_proposal_for(
    book_id: str = "live",
    capital_scope: str = "real",
    connection_ids: tuple[str, str] = ("live-first", "live-second"),
) -> BookExecutionProposal:
    ready = _proposal_for(book_id, capital_scope, connection_ids)
    rejected_risk = replace(
        ready.risk,
        passed=False,
        rejected_by="book_risk",
        reason="configured book risk limit",
    )
    return replace(
        ready,
        risk=rejected_risk,
        connection_risks=(),
        connection_plans=(),
        ready=False,
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


def _portfolio(
    proposal: BookExecutionProposal,
    *,
    after: bool = False,
    execution: BookExecutionResult | None = None,
) -> BookPortfolioSnapshot:
    plans = {plan.connection_id: plan for plan in proposal.connection_plans}
    results = {} if execution is None else {item.connection_id: item for item in execution.connection_results}
    connections = tuple(
        ConnectionPortfolioSnapshot(
            target.connection_id,
            Decimal("50"),
            {"USDT": Decimal("50")},
            (
                results[target.connection_id].final_position.position
                if after
                and target.connection_id in results
                and results[target.connection_id].final_position is not None
                else ConnectionPosition(
                    proposal.pair,
                    plans[target.connection_id].current_signed_amount
                    if target.connection_id in plans
                    else Decimal("0"),
                    plans[target.connection_id].current_signed_notional
                    if target.connection_id in plans
                    else Decimal("0"),
                    None,
                )
            ),
        )
        for target in proposal.risk.connection_targets
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
            book_id=book_id,
            capital_scope=capital_scope,
            config_revision=proposal.config_revision,
            pair=proposal.pair,
            proposal=proposal,
            portfolio_before=before,
            hitl=_hitl(proposal, "not_required"),
            execution=None,
            failure=None,
            portfolio_after=None,
            portfolio_after_available=None,
            status=status,
        )
    if status == "awaiting_approval":
        return BookCycleResult(
            book_id=book_id,
            capital_scope=capital_scope,
            config_revision=proposal.config_revision,
            pair=proposal.pair,
            proposal=proposal,
            portfolio_before=before,
            hitl=_hitl(proposal, "pending", f"approval-{book_id}"),
            execution=None,
            failure=None,
            portfolio_after=None,
            portfolio_after_available=None,
            status=status,
        )
    if status == "approval_rejected":
        return BookCycleResult(
            book_id=book_id,
            capital_scope=capital_scope,
            config_revision=proposal.config_revision,
            pair=proposal.pair,
            proposal=proposal,
            portfolio_before=before,
            hitl=_hitl(proposal, "rejected", f"approval-{book_id}"),
            execution=None,
            failure=None,
            portfolio_after=None,
            portfolio_after_available=None,
            status=status,
        )
    execution = _execution(proposal, status)
    hitl = (
        _hitl(proposal, "executed", f"approval-{book_id}")
        if capital_scope == "real"
        else _hitl(proposal, "not_required")
    )
    return BookCycleResult(
        book_id=book_id,
        capital_scope=capital_scope,
        config_revision=proposal.config_revision,
        pair=proposal.pair,
        proposal=proposal,
        portfolio_before=before,
        hitl=hitl,
        execution=execution,
        failure=None,
        portfolio_after=_portfolio(proposal, after=True, execution=execution),
        portfolio_after_available=True,
        status=status,
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
            book_id=completed.book_id,
            capital_scope=completed.capital_scope,
            config_revision=completed.config_revision,
            pair=completed.pair,
            proposal=completed.proposal,
            portfolio_before=completed.portfolio_before,
            hitl=completed.hitl,
            execution=completed.execution,
            failure=None,
            portfolio_after=completed.portfolio_after,
            portfolio_after_available=False,
            status="completed",
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
    for status in ("cycle_failed", "cancelled"):
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
async def test_replace_requires_existing_cycle(tmp_path, monkeypatch, database):
    from cryptotrader.journal import store as journal_store
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'missing.db'}" if database else None)
    sessions = []
    if database:
        await store.ensure_table()
        sessions = _install_tracking_write_session(monkeypatch, journal_store)
    with pytest.raises(LookupError, match="cycle was not found"):
        await store.replace(_record())
    if database:
        assert sessions[0].close_calls == 1


def _preparation_failure_book(
    stage: str,
    *,
    book_id: str = "live",
    capital_scope: str = "real",
    connection_ids: tuple[str, str] = ("live-first", "live-second"),
):
    from cryptotrader.journal.models import BookCycleResult, BookHitlSnapshot, BookPreparationFailure

    ready = _proposal_for(book_id, capital_scope, connection_ids)
    if stage in {"portfolio", "allocation"}:
        proposal = None
    elif stage == "risk":
        proposal = _rejected_proposal_for(book_id, capital_scope, connection_ids)
    else:
        proposal = replace(ready, connection_plans=(), ready=False)
    before = None if stage == "portfolio" else _portfolio(ready)
    return BookCycleResult(
        book_id=book_id,
        capital_scope=capital_scope,
        config_revision=ready.config_revision,
        pair=ready.pair,
        proposal=proposal,
        portfolio_before=before,
        hitl=BookHitlSnapshot(None, "not_required", ready.config_revision),
        execution=None,
        failure=BookPreparationFailure(stage),
        portfolio_after=None,
        portfolio_after_available=None,
        status="failed",
    )


def test_book_preparation_failure_is_safe_and_supports_truthful_partial_cycle():
    from cryptotrader.journal.models import BookPreparationFailure

    with pytest.raises(ValueError, match="stage"):
        BookPreparationFailure("raw_exchange_error")
    sim = _book_cycle(status="completed")
    live = _preparation_failure_book("risk")

    record = _record(
        book_results=(sim, live),
        cycle_status="partial",
        execution_status="partial",
        requires_attention=False,
    )

    assert record.book_results[1].proposal is not None
    assert record.book_results[1].proposal.ready is False
    assert record.book_results[1].execution is None


def test_risk_preparation_failure_rejects_a_contradictory_passed_risk_proposal():
    failure = _preparation_failure_book("risk")
    contradictory = replace(
        _proposal_for("live", "real", ("live-first", "live-second")),
        connection_plans=(),
        ready=False,
    )

    with pytest.raises(ValueError, match="risk"):
        replace(failure, proposal=contradictory)


@pytest.mark.parametrize("stage", ["portfolio", "allocation", "risk", "planning"])
def test_preparation_failures_never_require_attention(stage):
    failure = _preparation_failure_book(stage)
    cycle_status = "risk_rejected" if stage == "risk" else "failed"
    record = _record(
        book_results=(failure,),
        cycle_status=cycle_status,
        execution_status="not_started",
        requires_attention=False,
    )

    assert record.requires_attention is False
    with pytest.raises(ValueError, match="requires_attention"):
        replace(record, requires_attention=True)


def test_only_actual_execution_attention_contributes_to_cycle_attention():
    completed = _book_cycle(status="completed")
    preparation = _preparation_failure_book("planning")
    quiet = _record(
        book_results=(completed, preparation),
        cycle_status="partial",
        execution_status="partial",
        requires_attention=False,
    )
    assert quiet.requires_attention is False

    partial_execution = _book_cycle(status="partial")
    attention = replace(
        quiet,
        book_results=(partial_execution, preparation),
        requires_attention=True,
    )
    assert attention.requires_attention is True


@pytest.mark.parametrize(
    ("action_status", "expected_cycle"),
    [
        ("awaiting_approval", "awaiting_approval"),
        ("approval_rejected", "approval_rejected"),
    ],
)
def test_user_action_cycle_status_has_priority_over_preparation_failure(action_status, expected_cycle):
    action = _book_cycle(status=action_status)
    risk_failure = _preparation_failure_book("risk")

    record = _record(
        book_results=(action, risk_failure),
        cycle_status=expected_cycle,
        execution_status="not_started",
        requires_attention=False,
    )

    assert record.cycle_status == expected_cycle


def test_all_failed_books_derive_failed_execution_and_cycle_status():
    execution_failed = _book_cycle(status="failed")
    preparation_failed = _preparation_failure_book("planning")

    record = _record(
        book_results=(execution_failed, preparation_failed),
        cycle_status="failed",
        execution_status="failed",
        requires_attention=True,
    )

    assert record.cycle_status == "failed"
    assert record.execution_status == "failed"


def test_completed_and_preparation_failed_books_are_partial_without_attention():
    record = _record(
        book_results=(_book_cycle(status="completed"), _preparation_failure_book("planning")),
        cycle_status="partial",
        execution_status="partial",
        requires_attention=False,
    )

    assert record.cycle_status == "partial"
    assert record.execution_status == "partial"


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_preparation_failure_and_completed_sibling_round_trip_truthfully(tmp_path, database):
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'prep-roundtrip.db'}" if database else None)
    record = _record(
        cycle_id=f"prep-roundtrip-{database}",
        book_results=(_book_cycle(status="completed"), _preparation_failure_book("risk")),
        cycle_status="partial",
        execution_status="partial",
        requires_attention=False,
    )

    await store.save(record)

    assert await store.get(record.cycle_id) == record


def test_risk_rejected_requires_nonempty_per_book_risk_failure_evidence():
    risk_failure = _preparation_failure_book("risk")
    rejected = _record(
        book_results=(risk_failure,),
        cycle_status="risk_rejected",
        execution_status="not_started",
        requires_attention=False,
    )
    assert rejected.cycle_status == "risk_rejected"
    with pytest.raises(ValueError, match="risk_rejected"):
        replace(rejected, book_results=())
    with pytest.raises(ValueError, match="cycle_status"):
        _record(
            book_results=(_preparation_failure_book("planning"),),
            cycle_status="risk_rejected",
            execution_status="not_started",
            requires_attention=True,
        )


def _proposal_with_unavailable_reduction() -> BookExecutionProposal:
    proposal = _proposal_for()
    failed = ConnectionRiskDecision(
        proposal.connection_risks[1].connection_id,
        False,
        False,
        "venue unavailable",
        "fetch_quote",
    )
    return replace(
        proposal,
        connection_risks=(proposal.connection_risks[0], failed),
        connection_plans=(proposal.connection_plans[0],),
        unavailable_connections=(proposal.connection_risks[1].connection_id,),
        errors=(f"connection {proposal.connection_risks[1].connection_id}: fetch_quote failed",),
    )


def test_before_portfolio_uses_complete_risk_target_set_for_unavailable_reduction():
    from cryptotrader.journal.models import BookCycleResult, BookHitlSnapshot

    proposal = _proposal_with_unavailable_reduction()
    before = _portfolio(_proposal_for())

    result = BookCycleResult(
        book_id=proposal.book_id,
        capital_scope=proposal.capital_scope,
        config_revision=proposal.config_revision,
        pair=proposal.pair,
        proposal=proposal,
        portfolio_before=before,
        hitl=BookHitlSnapshot(None, "not_required", proposal.config_revision),
        execution=None,
        failure=None,
        portfolio_after=None,
        portfolio_after_available=None,
        status="ready",
    )

    assert tuple(item.connection_id for item in result.portfolio_before.connections) == (
        "sim-first",
        "sim-second",
    )


def test_unavailable_connection_position_remains_unchanged_after_reduction_execution():
    from cryptotrader.journal.models import BookCycleResult, BookHitlSnapshot

    proposal = _proposal_with_unavailable_reduction()
    before = _portfolio(_proposal_for())
    execution = BookExecutionResult(
        proposal,
        (_result(proposal, 0, "completed"),),
        "partial",
        False,
    )
    after = _portfolio(proposal, after=True, execution=execution)

    result = BookCycleResult(
        book_id=proposal.book_id,
        capital_scope=proposal.capital_scope,
        config_revision=proposal.config_revision,
        pair=proposal.pair,
        proposal=proposal,
        portfolio_before=before,
        hitl=BookHitlSnapshot(None, "not_required", proposal.config_revision),
        execution=execution,
        failure=None,
        portfolio_after=after,
        portfolio_after_available=True,
        status="partial",
    )

    assert result.portfolio_after is not None
    assert result.portfolio_after.connections[1].position == before.connections[1].position


def test_before_portfolio_must_match_risk_equity_and_plan_current_position():
    book = _book_cycle(status="ready")
    first = book.portfolio_before.connections[0]
    mismatched = replace(
        first,
        position=replace(first.position, signed_amount=Decimal("1"), signed_notional=Decimal("100")),
    )
    bad_before = replace(
        book.portfolio_before,
        connections=(mismatched, *book.portfolio_before.connections[1:]),
        total_signed_notional=Decimal("100"),
    )

    with pytest.raises(ValueError, match="current"):
        replace(book, portfolio_before=bad_before)
    first_with_more_equity = replace(book.portfolio_before.connections[0], equity=Decimal("51"))
    bad_equity = replace(
        book.portfolio_before,
        total_equity=Decimal("101"),
        connections=(first_with_more_equity, *book.portfolio_before.connections[1:]),
    )
    with pytest.raises(ValueError, match="book_equity"):
        replace(book, portfolio_before=bad_equity)


def test_after_portfolio_positions_match_execution_finals_or_unchanged_before():
    book = _book_cycle(status="partial")
    assert book.execution is not None
    assert book.portfolio_after is not None
    first = book.portfolio_after.connections[0]
    bad_after = replace(
        book.portfolio_after,
        connections=(
            replace(first, position=book.portfolio_before.connections[0].position),
            replace(
                book.portfolio_after.connections[1],
                position=book.portfolio_before.connections[1].position,
            ),
        ),
        total_signed_notional=book.portfolio_before.total_signed_notional,
    )
    with pytest.raises(ValueError, match="final_position"):
        replace(book, portfolio_after=bad_after)


def test_failed_execution_with_unknown_final_position_accepts_real_after_snapshot():
    book = _book_cycle(status="failed")
    assert book.execution is not None
    assert all(result.final_position is None for result in book.execution.connection_results)
    assert book.portfolio_after is not None
    changed = replace(
        book.portfolio_after.connections[0],
        position=ConnectionPosition(book.pair, Decimal("0.1"), Decimal("10"), Decimal("100")),
    )
    after = replace(
        book.portfolio_after,
        connections=(changed, *book.portfolio_after.connections[1:]),
        total_signed_notional=Decimal("10"),
    )

    result = replace(book, portfolio_after=after)

    assert result.portfolio_after.connections[0].position == changed.position


def test_connection_without_execution_result_must_remain_unchanged_in_after_snapshot():
    from cryptotrader.journal.models import BookCycleResult, BookHitlSnapshot

    proposal = _proposal_with_unavailable_reduction()
    before = _portfolio(_proposal_for())
    execution = BookExecutionResult(
        proposal,
        (_result(proposal, 0, "completed"),),
        "partial",
        False,
    )
    after = _portfolio(proposal, after=True, execution=execution)
    changed_unavailable = replace(
        after.connections[1],
        position=ConnectionPosition(proposal.pair, Decimal("0.1"), Decimal("10"), Decimal("100")),
    )
    changed_after = replace(
        after,
        connections=(after.connections[0], changed_unavailable),
        total_signed_notional=after.connections[0].position.signed_notional + Decimal("10"),
    )

    with pytest.raises(ValueError, match="portfolio_before"):
        BookCycleResult(
            book_id=proposal.book_id,
            capital_scope=proposal.capital_scope,
            config_revision=proposal.config_revision,
            pair=proposal.pair,
            proposal=proposal,
            portfolio_before=before,
            hitl=BookHitlSnapshot(None, "not_required", proposal.config_revision),
            execution=execution,
            failure=None,
            portfolio_after=changed_after,
            portfolio_after_available=True,
            status="partial",
        )


@pytest.mark.parametrize(
    "fused",
    [
        FusedSignal(
            -0.3,
            (
                ComponentContribution("kronos", -0.1, 0.8, -0.08),
                ComponentContribution("llm_committee", 1.1, -0.2, -0.22),
            ),
            "invalid negative weight",
        ),
        FusedSignal(
            0.22000000000000003,
            (
                ComponentContribution("kronos", 0.4, 0.8, 0.32000000000000006),
                ComponentContribution("llm_committee", 0.5, -0.2, -0.1),
            ),
            "invalid weight sum",
        ),
        FusedSignal(
            0.24999999999999997,
            (
                ComponentContribution("kronos", 0.5, 0.7, 0.35),
                ComponentContribution("llm_committee", 0.5, -0.2, -0.1),
            ),
            "forged signed score",
        ),
        FusedSignal(
            0.30000000000005,
            (
                ComponentContribution("kronos", 0.5, 0.8000000000001, 0.40000000000005),
                ComponentContribution("llm_committee", 0.5, -0.2, -0.1),
            ),
            "slightly forged signed score",
        ),
    ],
)
def test_fusion_rejects_invalid_weights_and_forged_component_scores(fused):
    with pytest.raises(ValueError, match=r"weight|signed_score"):
        replace(_record(), fused_signal=fused)


def test_fusion_weight_sum_uses_runtime_configuration_tolerance():
    fused = FusedSignal(
        0.30000000032,
        (
            ComponentContribution("kronos", 0.5000000004, 0.8, 0.40000000032),
            ComponentContribution("llm_committee", 0.5, -0.2, -0.1),
        ),
        "validated runtime weights",
    )

    record = replace(_record(), fused_signal=fused)

    assert record.fused_signal == fused


@pytest.mark.parametrize(
    ("target", "weights"),
    [
        (TargetPosition("long", 0.3), (0.0, 1.0)),
        (TargetPosition("short", 0.3), (1.0, 0.0)),
        (TargetPosition("long", 0.3), (0.2, 0.8)),
    ],
)
def test_target_direction_must_match_fused_score(target, weights):
    weighted = (weights[0] * 0.8, weights[1] * -0.2)
    fused = FusedSignal(
        sum(weighted),
        contributions=(
            ComponentContribution("kronos", weights[0], 0.8, weighted[0]),
            ComponentContribution("llm_committee", weights[1], -0.2, weighted[1]),
        ),
        reasoning="target sign mismatch",
    )
    with pytest.raises(ValueError, match="target_position"):
        replace(_record(), fused_signal=fused, target_position=target)


def test_component_failed_and_empty_global_statuses_have_truthful_field_matrix():
    component_failed = replace(
        _record(),
        fused_signal=None,
        target_position=None,
        book_results=(),
        cycle_status="component_failed",
        execution_status="not_started",
        requires_attention=False,
    )
    assert component_failed.fused_signal is None
    with pytest.raises(ValueError, match="component_failed"):
        replace(component_failed, fused_signal=_fused(), target_position=TargetPosition("long", 0.3))
    with pytest.raises(ValueError, match="risk_rejected"):
        replace(component_failed, cycle_status="risk_rejected")


def _record_with_secret_balance_key():
    book = _book_cycle(status="partial")
    secret_connection = replace(
        book.portfolio_before.connections[0],
        balances={"api_key": Decimal("1")},  # pragma: allowlist secret
    )
    before = replace(book.portfolio_before, connections=(secret_connection, *book.portfolio_before.connections[1:]))
    return replace(_record(), book_results=(replace(book, portfolio_before=before),))


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_journal_rejects_secret_shaped_portfolio_balance_keys_before_encoding(tmp_path, database):
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'secret-balance.db'}" if database else None)
    with pytest.raises(ValueError, match=r"^secret field$") as captured:
        await store.save(_record_with_secret_balance_key())
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_journal_rejects_injected_secret_balance_key_after_decode(tmp_path):
    from cryptotrader.journal.store import MultiVenueCycleStore

    path = tmp_path / "read-secret-balance.db"
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
        payload["items"][0]["portfolio_before"]["connections"][0]["balances"][0][0] = (
            "api_key"  # pragma: allowlist secret
        )
        connection.execute(
            "UPDATE multi_venue_cycles SET book_results = ? WHERE cycle_id = ?",
            (json.dumps(payload), record.cycle_id),
        )

    with pytest.raises(ValueError, match=r"^secret field$") as captured:
        await store.get(record.cycle_id)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_replace_cannot_rewrite_terminal_execution_or_portfolio(tmp_path, database):
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'terminal-rewrite.db'}" if database else None)
    original = _record()
    await store.save(original)
    book = original.book_results[0]
    assert book.execution is not None
    changed_result = replace(book.execution.connection_results[1], trace=("pre_read", "place_order"))
    changed_execution = replace(
        book.execution,
        connection_results=(book.execution.connection_results[0], changed_result),
    )
    changed = replace(original, book_results=(replace(book, execution=changed_execution),))

    with pytest.raises(ValueError, match=r"terminal|immutable"):
        await store.replace(changed)
    assert await store.get(original.cycle_id) == original

    assert book.portfolio_after is not None
    changed_after_connection = replace(
        book.portfolio_after.connections[0],
        balances={"USDT": Decimal("49")},
    )
    changed_after = replace(
        book.portfolio_after,
        connections=(changed_after_connection, *book.portfolio_after.connections[1:]),
    )
    changed_portfolio = replace(original, book_results=(replace(book, portfolio_after=changed_after),))
    with pytest.raises(ValueError, match=r"terminal|immutable"):
        await store.replace(changed_portfolio)
    assert await store.get(original.cycle_id) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_replace_cannot_rewrite_awaiting_approval_identity(tmp_path, database):
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'approval-rewrite.db'}" if database else None)
    live = _book_cycle(
        book_id="live",
        capital_scope="real",
        connection_ids=("live-first", "live-second"),
        status="awaiting_approval",
    )
    original = _record(
        book_results=(live,),
        cycle_status="awaiting_approval",
        execution_status="not_started",
        requires_attention=False,
    )
    await store.save(original)
    changed_hitl = replace(live.hitl, approval_id="different-approval")
    changed = replace(original, book_results=(replace(live, hitl=changed_hitl),))

    with pytest.raises(ValueError, match=r"approval|idempotent"):
        await store.replace(changed)
    assert await store.get(original.cycle_id) == original

    completed = _book_cycle(
        book_id="live",
        capital_scope="real",
        connection_ids=("live-first", "live-second"),
        status="completed",
    )
    completed_with_other_approval = replace(
        completed,
        hitl=replace(completed.hitl, approval_id="different-approval"),
    )
    progressed = replace(
        original,
        book_results=(completed_with_other_approval,),
        cycle_status="completed",
        execution_status="completed",
    )
    with pytest.raises(ValueError, match="approval"):
        await store.replace(progressed)
    assert await store.get(original.cycle_id) == original


class _TrackingSession:
    def __init__(self, inner, *, fail_add=False, fail_close=False):
        self.inner = inner
        self.fail_add = fail_add
        self.fail_close = fail_close
        self.close_calls = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def add(self, row):
        if self.fail_add:
            raise RuntimeError("program add failure")
        return self.inner.add(row)

    async def close(self):
        self.close_calls += 1
        await self.inner.close()
        if self.fail_close:
            raise SQLAlchemyError("RAW_CLOSE_MARKER")


def _install_tracking_write_session(monkeypatch, journal_store, *, fail_add=False, fail_close=False):
    original = journal_store._write_session
    sessions = []

    async def tracked(database_url):
        session = _TrackingSession(
            await original(database_url),
            fail_add=fail_add,
            fail_close=fail_close,
        )
        sessions.append(session)
        return session

    monkeypatch.setattr(journal_store, "_write_session", tracked)
    return sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["save", "replace"])
async def test_database_write_session_closes_once_on_success(tmp_path, monkeypatch, operation):
    from cryptotrader.journal import store as journal_store
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / f'close-success-{operation}.db'}")
    await store.ensure_table()
    record = _record()
    if operation == "replace":
        await store.save(record)
    sessions = _install_tracking_write_session(monkeypatch, journal_store)

    if operation == "save":
        await store.save(record)
    else:
        await store.replace(record)

    assert len(sessions) == 1
    assert sessions[0].close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["save", "replace"])
async def test_normal_database_write_close_failure_is_redacted(tmp_path, monkeypatch, operation):
    from cryptotrader.journal import store as journal_store
    from cryptotrader.journal.store import JournalPersistenceError, MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / f'close-failure-{operation}.db'}")
    await store.ensure_table()
    record = _record()
    if operation == "replace":
        await store.save(record)
    sessions = _install_tracking_write_session(monkeypatch, journal_store, fail_close=True)

    async def write():
        if operation == "save":
            return await store.save(record)
        return await store.replace(record)

    with pytest.raises(JournalPersistenceError, match=r"^journal persistence failed$") as captured:
        await write()

    assert sessions[0].close_calls == 1
    assert "RAW_CLOSE_MARKER" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_illegal_terminal_replace_closes_once_and_preserves_domain_error(tmp_path, monkeypatch):
    from cryptotrader.journal import store as journal_store
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'close-domain.db'}")
    original = _record()
    await store.save(original)
    book = original.book_results[0]
    changed_result = replace(book.execution.connection_results[1], trace=("pre_read", "place_order"))
    changed_execution = replace(
        book.execution,
        connection_results=(book.execution.connection_results[0], changed_result),
    )
    changed = replace(original, book_results=(replace(book, execution=changed_execution),))
    sessions = _install_tracking_write_session(monkeypatch, journal_store, fail_close=True)

    with pytest.raises(ValueError, match=r"terminal|immutable"):
        await store.replace(changed)

    assert sessions[0].close_calls == 1


@pytest.mark.asyncio
async def test_corrupt_replace_row_closes_write_session_once(tmp_path, monkeypatch):
    from cryptotrader.journal import store as journal_store
    from cryptotrader.journal.store import MultiVenueCycleStore

    path = tmp_path / "close-corrupt.db"
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
        payload["items"][0]["book_id"] = "corrupt-book"
        connection.execute(
            "UPDATE multi_venue_cycles SET book_results = ? WHERE cycle_id = ?",
            (json.dumps(payload), record.cycle_id),
        )
    sessions = _install_tracking_write_session(monkeypatch, journal_store)

    with pytest.raises(ValueError, match="stored cycle payload"):
        await store.replace(record)

    assert sessions[0].close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["add", "validate"])
async def test_program_write_failure_closes_once_without_close_overwrite(tmp_path, monkeypatch, failure):
    from cryptotrader.journal import store as journal_store
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / f'close-program-{failure}.db'}")
    await store.ensure_table()
    record = _record()
    if failure == "validate":
        await store.save(record)
    sessions = _install_tracking_write_session(
        monkeypatch,
        journal_store,
        fail_add=failure == "add",
        fail_close=True,
    )
    if failure == "validate":

        def fail_validation(*_args):
            raise RuntimeError("program validation failure")

        monkeypatch.setattr(MultiVenueCycleStore, "_validate_replacement", staticmethod(fail_validation))

    async def write():
        if failure == "add":
            return await store.save(record)
        return await store.replace(record)

    message = "program add failure" if failure == "add" else "program validation failure"
    with pytest.raises(RuntimeError, match=message):
        await write()

    assert sessions[0].close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["save", "replace"])
async def test_sqlalchemy_write_failures_are_redacted_without_params_or_context(tmp_path, monkeypatch, operation):
    from cryptotrader.journal import store as journal_store
    from cryptotrader.journal.store import JournalPersistenceError, MultiVenueCycleStore

    path = tmp_path / f"abort-{operation}.db"
    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{path}")
    await store.ensure_table()
    record = _record()
    replacement = record
    if operation == "replace":
        live = _book_cycle(
            book_id="live",
            capital_scope="real",
            connection_ids=("live-first", "live-second"),
            status="awaiting_approval",
        )
        record = _record(
            book_results=(live,),
            cycle_status="awaiting_approval",
            execution_status="not_started",
            requires_attention=False,
        )
        await store.save(record)
        completed = _book_cycle(
            book_id="live",
            capital_scope="real",
            connection_ids=("live-first", "live-second"),
            status="completed",
        )
        replacement = replace(
            record,
            book_results=(completed,),
            cycle_status="completed",
            execution_status="completed",
        )
    action = "INSERT" if operation == "save" else "UPDATE"
    with sqlite3.connect(path) as connection:
        connection.execute(
            f"CREATE TRIGGER abort_cycle_{operation} BEFORE {action} ON multi_venue_cycles "
            "BEGIN SELECT RAISE(ABORT, 'RAW_SECRET_MARKER'); END"
        )
    sessions = _install_tracking_write_session(monkeypatch, journal_store)

    async def write():
        if operation == "save":
            return await store.save(record)
        return await store.replace(replacement)

    with pytest.raises(JournalPersistenceError, match=r"^journal persistence failed$") as captured:
        await write()
    assert not hasattr(captured.value, "params")
    assert "RAW_SECRET_MARKER" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sessions[0].close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["save", "replace"])
async def test_sqlalchemy_session_creation_failures_are_redacted(tmp_path, monkeypatch, operation):
    from sqlalchemy.exc import SQLAlchemyError

    from cryptotrader.journal import store as journal_store
    from cryptotrader.journal.store import JournalPersistenceError, MultiVenueCycleStore

    path = tmp_path / f"session-{operation}.db"
    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{path}")
    await store.ensure_table()
    record = _record()
    if operation == "replace":
        await store.save(record)

    async def fail_session(_database_url):
        raise SQLAlchemyError("RAW_SECRET_MARKER")

    async def write():
        if operation == "save":
            return await store.save(record)
        return await store.replace(record)

    monkeypatch.setattr(journal_store, "get_async_session", fail_session)
    with pytest.raises(JournalPersistenceError, match=r"^journal persistence failed$") as captured:
        await write()
    assert "RAW_SECRET_MARKER" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
