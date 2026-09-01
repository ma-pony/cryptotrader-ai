"""Paper persistence and account ingestion use only temporary databases."""

from decimal import Decimal

from cryptotrader.pair import Pair
from cryptotrader.venues.models import OrderIntent
from cryptotrader.venues.paper import PaperVenueAdapter
from tests.factories.runtime_config import connection


async def _migrated_url(path):
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    database_url = f"sqlite+aiosqlite:///{path}"
    await migrate_workbench_schema(database_url)
    return database_url


async def test_existing_session_sync_serializes_history_with_background_batch(tmp_path):
    import asyncio
    from dataclasses import replace
    from datetime import timedelta

    from cryptotrader.accounts.models import FillPage, FundingPage
    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.accounts.sync import AccountSyncService
    from tests.fakes.account_session import END, START, AccountSession

    store = AccountStore(await _migrated_url(tmp_path / "serialized.db"))
    old, current = AccountSession(), AccountSession()
    new_end = END + timedelta(hours=1)
    current.snapshot = replace(current.snapshot, observed_at=new_end)
    paused, release = asyncio.Event(), asyncio.Event()
    original_fetch = old.fetch_fills

    async def paused_fills(cursor):
        page = await original_fetch(cursor)
        paused.set()
        await release.wait()
        return page

    async def new_fills(cursor):
        current.reads.append(("fills", cursor))
        return FillPage(current.fills, "fills-new", True, END if cursor else START, new_end)

    async def new_funding(cursor):
        current.reads.append(("funding", cursor))
        return FundingPage(current.funding, "fund-new", True, END if cursor else START, new_end)

    old.fetch_fills = paused_fills
    current.fetch_fills, current.fetch_funding = new_fills, new_funding
    service = AccountSyncService(store, old.provide)
    background = asyncio.create_task(service.sync("sim-a"))
    await asyncio.wait_for(paused.wait(), 1)
    manual = asyncio.create_task(service.sync_session(current, "sim-a"))
    try:
        await asyncio.sleep(0.03)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(background, manual), 2)
    coverage = (await store.status("sim-a"))["coverage"]
    assert coverage["fills"]["coverage_end"] == new_end.isoformat()
    assert coverage["funding"]["coverage_end"] == new_end.isoformat()
    assert coverage["fills"]["cursor"] == "fills-new"
    assert coverage["funding"]["cursor"] == "fund-new"
    assert current.reads == [("fills", "fills-end"), ("funding", "fund-end")]
    assert (await store.latest("sim-a")).observed_at == new_end
    assert await store.fill_count("sim-a") == 3


async def test_paper_restart_preserves_fills_balance_start_and_quotes(tmp_path):
    database_url = await _migrated_url(tmp_path / "paper.db")
    # Bind the persistence context before connect; a fresh adapter represents restart.
    adapter = PaperVenueAdapter()
    adapter.database_url = database_url
    config = connection("sim-a", "paper", parameters={"initial_equity": "1000"})
    session = await adapter.connect(config, None)
    pair = Pair.parse("BTC/USDT")
    await session.set_quote(pair, Decimal("100"))
    order = await session.place_order(OrderIntent(pair, "buy", Decimal("2"), "market", None, False))
    before = await session.fetch_account()
    page = await session.fetch_fills(None)
    await session.close()
    restarted = PaperVenueAdapter()
    restarted.database_url = database_url
    restored = await restarted.connect(config, None)
    after = await restored.fetch_account()
    assert after.balances == before.balances
    assert after.positions == before.positions
    assert after.completeness == before.completeness
    assert "不是本次同步的新行情" in after.valuation_notes[0]
    assert (await restored.fetch_fills(None)).coverage_start == page.coverage_start
    assert (await restored.fetch_fills(None)).items == page.items
    next_order = await restored.place_order(OrderIntent(pair, "sell", Decimal("1"), "market", None, False))
    assert next_order.id != order.id


async def test_paper_protection_and_sequence_restore_and_other_database_isolated(tmp_path):
    from cryptotrader.venues.models import ProtectionSpec

    url = await _migrated_url(tmp_path / "protection.db")
    config = connection("same-name", "paper", parameters={"initial_equity": "1000"})
    pair = Pair.parse("BTC/USDT:USDT")
    adapter = PaperVenueAdapter()
    adapter.database_url = url
    first = await adapter.connect(config, None)
    await first.set_quote(pair, Decimal("100"))
    await first.place_order(OrderIntent(pair, "buy", Decimal("1"), "market", None, False))
    spec = ProtectionSpec(pair, "long", Decimal("1"), Decimal("90"), Decimal("120"))
    protection = await first.replace_protection(spec)
    restarted = PaperVenueAdapter()
    restarted.database_url = url
    restored = await restarted.connect(config, None)
    assert (await restored.list_open_state(pair)).protections == (protection,)
    replacement = await restored.replace_protection(spec)
    assert replacement.protection_ids != protection.protection_ids
    independent = PaperVenueAdapter()
    independent.database_url = await _migrated_url(tmp_path / "separate.db")
    separate = await independent.connect(config, None)
    assert (await separate.fetch_account()).positions == ()
    assert (await separate.fetch_account()).equity.amount == Decimal("1000")


async def test_sync_deduplicates_and_reconstructs_service_without_losing_identity(tmp_path):
    import pytest

    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.accounts.sync import AccountSyncError, AccountSyncService
    from tests.fakes.account_session import AccountSession

    url = await _migrated_url(tmp_path / "ledger.db")
    store = AccountStore(url)
    first, second = AccountSession("sim-a"), AccountSession("sim-b")
    sync = AccountSyncService(store, first.provide)
    await sync.sync("sim-a")
    await sync.sync("sim-a")
    await AccountSyncService(store, second.provide).sync("sim-b")
    assert await store.fill_count("sim-a") == 3
    assert await store.fill_count("sim-b") == 3
    persisted = await store.latest("sim-a")
    first.error = True
    with pytest.raises(AccountSyncError):
        await sync.sync("sim-a")
    reopened = AccountStore(url)
    assert await reopened.latest("sim-a") == persisted
    status = await reopened.status("sim-a")
    assert status["last_success_at"]
    assert status["last_failure_at"]
    assert "SECRET" not in status["failure_reason"]
    assert status["coverage"]["fills"]["cursor"] == "fills-next"


async def test_failed_history_batch_does_not_advance_checkpoint_or_snapshot(tmp_path):
    from dataclasses import replace
    from datetime import timedelta

    import pytest

    from cryptotrader.accounts.models import FillPage
    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.accounts.sync import AccountSyncError, AccountSyncService
    from tests.fakes.account_session import END, AccountSession

    store = AccountStore(await _migrated_url(tmp_path / "atomic.db"))
    session = AccountSession()
    sync = AccountSyncService(store, session.provide)
    await sync.sync("sim-a")
    status = await store.status("sim-a")
    session.snapshot = replace(session.snapshot, observed_at=END + timedelta(days=1))

    async def broken(cursor):
        if cursor == "fills-end":
            return FillPage(
                (replace(session.fills[0], venue_fill_id="new"),), "half", False, END, END + timedelta(days=1)
            )
        raise RuntimeError("offline pagination failure")

    session.fetch_fills = broken
    with pytest.raises(AccountSyncError):
        await sync.sync("sim-a")
    assert await store.fill_count("sim-a") == 3
    assert (await store.latest("sim-a")).observed_at == END
    assert (await store.status("sim-a"))["coverage"] == status["coverage"]


async def test_membership_cas_and_late_exact_order_binding(tmp_path):
    from dataclasses import replace
    from datetime import timedelta

    import pytest

    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.runtime_config.repository import RevisionConflict, RuntimeConfigRepository
    from cryptotrader.runtime_config.secrets import CredentialVault
    from tests.test_runtime_config_api import MASTER_KEY, active_document

    repository = RuntimeConfigRepository(
        await _migrated_url(tmp_path / "membership.db"), CredentialVault(MASTER_KEY), active_document
    )
    initial = await repository.get_or_create()
    store = repository.account_store
    before = initial.updated_at - timedelta(seconds=1)
    assert (await store.attribution("okx-demo", "external", None, before))["book_id"] is None
    assert (await store.attribution("okx-demo", "external", None, initial.updated_at))["book_id"] == "simulation"
    document = initial.document.model_copy(
        update={
            "execution": initial.document.execution.model_copy(
                update={"books": tuple(replace(book, enabled=False) for book in initial.document.execution.books)}
            )
        }
    )
    saved = await repository.replace(initial.revision, document)
    assert (await store.attribution("okx-demo", "external", None, saved.updated_at))["book_id"] == "simulation"
    unassigned = document.model_copy(
        update={
            "execution": document.execution.model_copy(
                update={"books": tuple(replace(book, allocations=()) for book in document.execution.books)}
            )
        }
    )
    saved = await repository.replace(saved.revision, unassigned)
    with pytest.raises(RevisionConflict):
        await repository.replace(initial.revision, initial.document)
    assert (await store.attribution("okx-demo", "external", None, saved.updated_at))["book_id"] is None
    # Attribution resolves when evidence arrives later; it is not frozen as external on first observation.
    await store.bind_order(
        "okx-demo", "realclient", book_id="simulation", decision_id="decision-1", venue_order_id="actual-order"
    )
    reopened = AccountStore(repository.database_url)
    assert await reopened.attribution("okx-demo", "actual-order", None, saved.updated_at) == {
        "source": "strategy",
        "book_id": "simulation",
        "decision_id": "decision-1",
        "operation_id": None,
    }


async def test_real_execution_intents_have_durable_exact_client_order_identity(tmp_path):
    from dataclasses import replace

    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.execution.service import VenueExecutionService
    from tests.fakes.account_session import END
    from tests.test_execution_service import _venue_plan, _VenueSession

    store = AccountStore(await _migrated_url(tmp_path / "orders.db"))
    session = _VenueSession("0")
    service = VenueExecutionService(session, connection=session.connection)
    service.account_store = store
    result = await service.execute(replace(_venue_plan("0", "1"), decision_id="decision-true"))
    assert result.status == "completed"
    assert session.orders[0].client_order_id
    owner = await store.attribution("paper-a", result.orders[0].id, session.orders[0].client_order_id, END)
    assert owner["source"] == "strategy"
    assert owner["book_id"] == "simulation"
    protection_id = session.protections[0].protection_ids[0]
    assert (await store.attribution("paper-a", protection_id, None, END))["decision_id"] == "decision-true"


async def test_rejected_submission_preserves_identity_without_fabricating_fill(tmp_path):
    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.execution.service import VenueExecutionService
    from cryptotrader.venues.protocol import VenueOperationError
    from tests.fakes.account_session import END
    from tests.test_execution_service import _venue_plan, _VenueSession

    store = AccountStore(await _migrated_url(tmp_path / "rejected.db"))
    session = _VenueSession("0")
    attempted = []

    async def reject(intent):
        attempted.append(intent)
        raise VenueOperationError("fake rejected submission")

    session.place_order = reject
    result = await VenueExecutionService(session, connection=session.connection, account_store=store).execute(
        _venue_plan("0", "1")
    )
    assert result.status == "failed"
    assert (await store.attribution("paper-a", "not-accepted", attempted[0].client_order_id, END))[
        "source"
    ] == "strategy"
    assert await store.fill_count("paper-a") == 0
    assert await store.orders("paper-a") == []


async def test_synthetic_bybit_protection_reference_does_not_become_order_identity(tmp_path):
    from dataclasses import replace

    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.execution.service import VenueExecutionService
    from tests.fakes.account_session import END
    from tests.test_execution_service import _venue_plan, _VenueSession

    store = AccountStore(await _migrated_url(tmp_path / "synthetic.db"))
    session = _VenueSession("0")
    original = session.replace_protection

    async def position_reference(spec):
        state = await original(spec)
        state = replace(state, protection_ids=("bybit-position:BTCUSDT:0",))
        session.protections = (state,)
        return state

    session.replace_protection = position_reference
    result = await VenueExecutionService(session, connection=session.connection, account_store=store).execute(
        _venue_plan("0", "1")
    )
    assert result.status == "completed"
    assert (await store.attribution("paper-a", "bybit-position:BTCUSDT:0", None, END))["source"] == "external"


async def test_execution_projection_uses_full_account_and_rejects_usd_usdt_conversion():
    from dataclasses import replace

    from cryptotrader.accounts.models import AccountPosition, Money
    from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
    from cryptotrader.portfolio.aggregator import PortfolioAggregator
    from tests.fakes.account_session import INSTRUMENT, PAIR, AccountSession

    session = AccountSession()
    session.snapshot = replace(
        session.snapshot,
        positions=(
            AccountPosition(
                INSTRUMENT,
                Decimal("2"),
                Decimal("2"),
                Money(Decimal("200"), "USDT"),
                Decimal("100"),
                Money(Decimal("0"), "USDT"),
            ),
        ),
    )
    book = ExecutionBook("sim", "模拟", "simulated", True, False, (ConnectionAllocation("sim-a", True, 1.0),))
    projected = await PortfolioAggregator().read(book, {"sim-a": session}, PAIR)
    assert projected.total_equity == Decimal("1000")
    assert projected.connections[0].account_snapshot.positions == session.snapshot.positions
    original = projected.connections[0]
    assert original != replace(original, account_snapshot=replace(session.snapshot, positions=()))
    from cryptotrader.journal.store import _connection_portfolio_from_payload, _connection_portfolio_payload

    assert (
        _connection_portfolio_from_payload(_connection_portfolio_payload(original)).account_snapshot == session.snapshot
    )
    session.snapshot = replace(session.snapshot, equity=Money(Decimal("1000"), "USD"))
    unknown = await PortfolioAggregator().read(book, {"sim-a": session}, PAIR)
    assert unknown.total_equity is None
    assert unknown.connections[0].account_snapshot.equity.currency == "USD"


async def test_explicit_read_owner_syncs_disabled_unassigned_account_and_survives_pause(tmp_path):
    import asyncio
    from dataclasses import replace

    from fastapi import FastAPI

    from api.main import _clear_runtime_owners, _init_account_sync
    from cryptotrader.runtime import build_runtime
    from cryptotrader.runtime_config.defaults import minimal_runtime_document
    from cryptotrader.runtime_config.models import AccountsConfig, ExecutionConfig
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository
    from cryptotrader.runtime_config.secrets import CredentialVault
    from cryptotrader.venues.registry import VenueAdapterRegistry
    from tests.fakes.account_session import AccountSession
    from tests.test_runtime_config_api import MASTER_KEY

    document = minimal_runtime_document().model_copy(
        update={
            "execution": ExecutionConfig(connections=(replace(connection("sim-a", "paper"), enabled=False),), books=()),
            "accounts": AccountsConfig(sync_interval_seconds=1),
        }
    )
    repository = RuntimeConfigRepository(
        await _migrated_url(tmp_path / "owner.db"), CredentialVault(MASTER_KEY), lambda: document
    )
    observed = asyncio.Event()

    class Adapter:
        adapter_id = "paper"

        def __init__(self):
            self.opens = 0

        def capabilities(self, environment):
            return PaperVenueAdapter().capabilities(environment)

        async def connect(self, config, credentials):
            self.opens += 1
            observed.set()
            return AccountSession(config.id)

    adapter = Adapter()
    runtime = await build_runtime(repository=repository, venue_registry=VenueAdapterRegistry((adapter,)))
    assert adapter.opens == 0
    app = FastAPI()
    app.state.runtime = runtime
    await _init_account_sync(app)
    await asyncio.wait_for(observed.wait(), 2)
    owner = app.state.account_sync_owner
    await _clear_runtime_owners(app)
    assert app.state.account_sync_owner is owner
    await owner.stop()
    assert await repository.account_store.fill_count("sim-a") == 3
    assert (await repository.get_existing()).document.execution.connections[0].enabled is False
    assert (await repository.get_existing()).document.scheduler.automation_enabled is False
    await runtime.close()


async def test_checkpoint_database_failure_rolls_back_facts(tmp_path):
    import pytest
    from sqlalchemy import event

    from cryptotrader.accounts.store import AccountStore, CursorRow
    from cryptotrader.accounts.sync import AccountSyncError, AccountSyncService
    from tests.fakes.account_session import AccountSession

    store = AccountStore(await _migrated_url(tmp_path / "rollback.db"))
    session = AccountSession()

    def fail_checkpoint(mapper, connection, target):
        raise RuntimeError("fake disk failure at checkpoint")

    event.listen(CursorRow, "before_insert", fail_checkpoint)
    try:
        with pytest.raises(AccountSyncError):
            await AccountSyncService(store, session.provide).sync("sim-a")
    finally:
        event.remove(CursorRow, "before_insert", fail_checkpoint)
    assert await store.latest("sim-a") is None
    assert await store.fill_count("sim-a") == 0
    assert await store.history("sim-a", "funding") == []
    assert (await store.status("sim-a"))["coverage"] == {}


async def test_sync_catches_up_contiguous_windows_before_committing(tmp_path):
    from dataclasses import replace
    from datetime import timedelta

    from cryptotrader.accounts.models import FillPage, FundingPage
    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.accounts.sync import AccountSyncService
    from tests.fakes.account_session import START, AccountSession

    class WindowSession(AccountSession):
        async def fetch_fills(self, cursor):
            index = int(cursor or 0)
            start = START + timedelta(days=7 * index)
            end = min(start + timedelta(days=7), self.snapshot.observed_at)
            self.calls.append(("fills", index))
            return FillPage(self.fills if index == 0 else (), str(index + 1), True, start, end)

        async def fetch_funding(self, cursor):
            index = int(cursor or 0)
            start = START + timedelta(days=7 * index)
            end = min(start + timedelta(days=7), self.snapshot.observed_at)
            self.calls.append(("funding", index))
            return FundingPage(self.funding if index == 0 else (), str(index + 1), True, start, end)

    store = AccountStore(await _migrated_url(tmp_path / "windows.db"))
    session = WindowSession()
    session.calls = []
    session.snapshot = replace(session.snapshot, observed_at=START + timedelta(days=20))
    await AccountSyncService(store, session.provide).sync("sim-a")
    assert session.calls == [(kind, index) for kind in ("fills", "funding") for index in range(3)]
    assert (await store.status("sim-a"))["coverage"]["fills"][
        "coverage_end"
    ] == session.snapshot.observed_at.isoformat()
    assert await store.fill_count("sim-a") == 3


async def test_explicit_account_migration_backs_up_config_and_starts_membership_now(tmp_path):
    import json
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import delete

    from cryptotrader.accounts.store import MembershipRow
    from cryptotrader.db import get_async_session
    from cryptotrader.migrations.workbench import migrate_account_ledger
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository, _RuntimeConfigRow
    from cryptotrader.runtime_config.secrets import CredentialVault
    from tests.test_runtime_config_api import MASTER_KEY, active_document

    repository = RuntimeConfigRepository(
        await _migrated_url(tmp_path / "migration.db"), CredentialVault(MASTER_KEY), active_document
    )
    initial = await repository.get_or_create()
    async with await get_async_session(repository.database_url) as session, session.begin():
        row = await session.get(_RuntimeConfigRow, "global")
        row.document = {key: value for key, value in row.document.items() if key != "accounts"}
        await session.execute(delete(MembershipRow))
    before = datetime.now(UTC) - timedelta(seconds=1)
    backup = tmp_path / "before-accounts.json"
    assert await migrate_account_ledger(repository.database_url, backup) == 1
    assert "accounts" not in json.loads(backup.read_text())["document"]
    saved = await repository.get_existing()
    assert saved.revision == initial.revision + 1
    assert saved.document.accounts.sync_interval_seconds == 60
    assert (await repository.account_store.attribution("okx-demo", "external", None, before))["book_id"] is None
    assert (await repository.account_store.attribution("okx-demo", "external", None, datetime.now(UTC)))[
        "book_id"
    ] == "simulation"
    assert await migrate_account_ledger(repository.database_url, backup) == 0
