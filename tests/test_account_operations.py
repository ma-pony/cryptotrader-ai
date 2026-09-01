"""Manual account operations exercise the real HTTP/service/store against offline facts."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from decimal import Decimal
from types import MethodType

import pytest

from cryptotrader.accounts.models import AccountOrder, AccountPosition, Money
from cryptotrader.accounts.sync import AccountSyncService
from cryptotrader.runtime import Runtime
from cryptotrader.venues.models import EXIT_OPERATIONS, NormalizedOrder, VenueCapabilities
from tests.fakes.account_session import END, INSTRUMENT, AccountSession


class ExitSession(AccountSession):
    def __init__(self, connection_id="okx-demo", *, pair=None):
        super().__init__(connection_id)
        self.instrument = INSTRUMENT if pair is None else replace(INSTRUMENT, pair=pair, market_type=pair.market_type)
        self.amount = Decimal("2")
        self.available = Decimal("2")
        self.ordinary = True
        self.protection = True
        self.calls = []
        self.cancel_fill = False
        self.close_fail = False
        self.read_fail = False
        self.after_close_fail = False
        self.capabilities = VenueCapabilities(
            frozenset({"spot", "swap"}), True, True, True, frozenset({"market"}), exit_operations=EXIT_OPERATIONS
        )

    async def fetch_account(self):
        if self.read_fail:
            raise RuntimeError("SECRET must not escape")
        self.calls.append("confirm_flat" if self.amount == 0 else "read")
        positions = (
            AccountPosition(
                self.instrument,
                self.amount,
                self.available,
                Money(abs(self.amount) * 100, "USDT"),
                Decimal("100"),
                Money(Decimal("0"), "USDT"),
            ),
        )
        orders = tuple(
            AccountOrder(
                self.connection_id,
                order_id,
                self.instrument,
                "sell",
                "limit",
                Decimal("2"),
                Decimal("0"),
                None,
                "open",
                protected,
                protected,
                None,
                END,
                Money(Decimal("200"), "USDT"),
            )
            for order_id, protected, exists in (
                ("ordinary", False, self.ordinary),
                ("protection", True, self.protection),
            )
            if exists
        )
        return replace(self.snapshot, positions=positions, orders=orders)

    async def normalize_amount(self, pair, amount):
        return amount

    async def cancel_order(self, order_id, pair):
        assert order_id == "ordinary"
        self.calls.append("cancel_ordinary")
        self.ordinary = False
        if self.cancel_fill:
            self.amount += 1
            self.available += 1
            self.cancel_fill = False

    async def cancel_protection(self, ids):
        assert self.amount == 0
        assert ids == ("protection",)
        self.calls.append("cancel_protection")
        self.protection = False

    async def place_order(self, intent):
        self.calls.append("close_reduce_only" if intent.reduce_only else "close_spot")
        if self.close_fail:
            raise RuntimeError("SECRET close failure")
        assert intent.amount <= self.available
        self.amount -= intent.amount
        self.available -= intent.amount
        self.read_fail = self.after_close_fail
        self.fills = (
            *self.fills,
            replace(
                self.fills[0],
                venue_fill_id="manual-fill",
                venue_order_id="close-1",
                instrument=self.instrument,
                client_order_id=intent.client_order_id,
                amount=intent.amount,
                side=intent.side,
            ),
        )
        return NormalizedOrder(
            "close-1",
            intent.pair,
            intent.side,
            intent.order_type,
            intent.amount,
            intent.amount,
            Decimal("100"),
            "filled",
            intent.reduce_only,
            intent.client_order_id,
        )


@pytest.fixture
async def operation_harness(api_harness, monkeypatch):
    runtime = api_harness.runtime
    venue = ExitSession()
    runtime.account_session = venue.provide
    runtime.account_sync = AccountSyncService(runtime.repository.account_store, venue.provide)
    runtime.refresh_owners = None
    runtime.clear_owners = None
    runtime.apply_automation = MethodType(Runtime.apply_automation, runtime)
    runtime._fail_automation = MethodType(Runtime._fail_automation, runtime)
    runtime._leases_drained = asyncio.Event()
    runtime._leases_drained.set()
    runtime.wait_for_execution_idle = (
        MethodType(Runtime.wait_for_execution_idle, runtime)
        if hasattr(Runtime, "wait_for_execution_idle")
        else runtime._leases_drained.wait
    )
    runtime.ownership_calls = []

    @asynccontextmanager
    async def account_operation_lease():
        yield

    runtime.account_operation_lease = account_operation_lease
    from tests.fakes.account_session import snapshot

    await runtime.repository.account_store.ingest(snapshot("bybit-testnet"))

    @asynccontextmanager
    async def lease(_self, key):
        runtime.ownership_calls.append(key)
        await runtime._leases_drained.wait()
        yield

    monkeypatch.setattr("cryptotrader.execution_ownership.ExecutionOwnership.book", lease)
    monkeypatch.setattr("cryptotrader.execution_ownership.ExecutionOwnership.connection", lease)
    api_harness.venue = venue
    yield api_harness
    service = getattr(runtime, "account_operations", None)
    if service:
        await service.close()


async def wait_operation(client, operation_id):
    for _ in range(100):
        response = await client.get(f"/api/account-operations/{operation_id}")
        assert response.status_code == 200, response.text
        value = response.json()
        if value["status"] not in {"preparing", "executing"}:
            return value
        await asyncio.sleep(0.01)
    pytest.fail("operation did not settle")


async def prepare(harness, *, kind="flatten"):
    revision = (await harness.runtime.repository.get_existing()).revision
    response = await harness.client.post(
        "/api/accounts/okx-demo/operations/prepare",
        json={
            "pair": str(harness.venue.instrument.pair),
            "kind": kind,
            "expected_revision": revision,
            "confirm_stop": True,
        },
    )
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "preparing"
    return await wait_operation(harness.client, response.json()["operation_id"])


async def execute(harness, operation):
    response = await harness.client.post(
        f"/api/account-operations/{operation['operation_id']}/execute",
        json={"plan_version": operation["plan"]["version"]},
    )
    assert response.status_code == 202, response.text
    return await wait_operation(harness.client, operation["operation_id"])


async def test_prepare_stops_only_owned_pool_and_never_places_orders(operation_harness):
    h = operation_harness
    operation = await prepare(h)
    assert operation["status"] == "awaiting_confirmation"
    assert operation["plan"]["position_amount"] == "2"
    assert operation["plan"]["ordinary_order_ids"] == ["ordinary"]
    assert set(h.venue.calls) == {"read"}
    books = (await h.runtime.repository.get_existing()).document.execution.books
    assert books[0].enabled is False
    assert books[1].enabled is True
    assert h.runtime.ownership_calls == ["simulation"]
    h.runtime.reload_for_cycle.assert_not_awaited()


async def test_manual_prepare_and_execute_never_call_signal_models_or_strategy_approvals(
    operation_harness, monkeypatch
):
    from unittest.mock import AsyncMock

    models = AsyncMock(side_effect=AssertionError("manual exit must not run signal models"))
    approvals = AsyncMock(side_effect=AssertionError("manual exit must not request strategy approval"))
    monkeypatch.setattr("cryptotrader.signals.runner.ComponentRunner.run", models)
    monkeypatch.setattr("cryptotrader.signals.components.llm_committee.LLMCommitteeComponent.evaluate", models)
    monkeypatch.setattr("cryptotrader.signals.components.kronos.KronosComponent.evaluate", models)
    monkeypatch.setattr("cryptotrader.hitl.store.BookApprovalStore.create", approvals)
    operation = await prepare(operation_harness)
    models.assert_not_called()
    approvals.assert_not_called()
    completed = await execute(operation_harness, operation)
    assert completed["status"] == "completed"
    models.assert_not_called()
    approvals.assert_not_called()


@pytest.mark.parametrize("derivative", [False, True])
async def test_exit_cancels_in_safe_order_and_persists_real_receipt(operation_harness, derivative):
    h = operation_harness
    if derivative:
        from cryptotrader.pair import Pair

        h.venue.instrument = replace(INSTRUMENT, pair=Pair.parse("BTC/USDT:USDT"), market_type="swap")
    operation = await execute(h, await prepare(h))
    assert operation["status"] == "completed", operation
    close = "close_reduce_only" if derivative else "close_spot"
    assert h.venue.calls.index("cancel_ordinary") < h.venue.calls.index(close)
    assert h.venue.calls.index("confirm_flat") < h.venue.calls.index("cancel_protection")
    assert operation["result"]["remaining_position"] == "0"
    assert operation["result"]["orders"][0]["id"] == "close-1"
    store = h.runtime.repository.account_store
    binding = await store.attribution("okx-demo", "close-1", None, None)
    assert binding["operation_id"] == operation["operation_id"]
    assert binding["source"] == "manual"
    fills = await store.history("okx-demo")
    assert "manual-fill" in {fill.venue_fill_id for fill in fills}
    from cryptotrader.accounts.store import AccountOperationStore

    assert (await AccountOperationStore(store.database_url).get(operation["operation_id"])).status == "completed"


async def test_changed_position_invalidates_without_cancel(operation_harness):
    h = operation_harness
    operation = await prepare(h)
    h.venue.amount = Decimal("3")
    result = await execute(h, operation)
    assert result["status"] == "invalidated"
    assert "cancel_ordinary" not in h.venue.calls


async def test_fill_during_cancel_needs_new_confirmation(operation_harness):
    h = operation_harness
    operation = await prepare(h)
    h.venue.cancel_fill = True
    result = await execute(h, operation)
    assert result["status"] == "awaiting_confirmation"
    assert result["plan"]["version"] == 2
    assert result["plan"]["position_amount"] == "3"
    assert "close_spot" not in h.venue.calls
    completed = await execute(h, result)
    assert completed["status"] == "completed"
    assert completed["result"]["failure_reason"] is None
    assert completed["result"]["canceled_order_ids"] == ["ordinary"]
    assert completed["result"]["orders"][0]["amount"] == "3"


@pytest.mark.parametrize("after_close", [False, True])
async def test_failure_preserves_protection_and_actual_execution(operation_harness, after_close):
    h = operation_harness
    operation = await prepare(h)
    h.venue.close_fail = not after_close
    h.venue.after_close_fail = after_close
    result = await execute(h, operation)
    assert result["status"] == "failed"
    assert h.venue.protection
    assert "SECRET" not in str(result)
    assert bool(result["result"]["orders"]) is after_close
    assert (
        result["result"]["remaining_position"] is None if after_close else result["result"]["remaining_position"] == "2"
    )


async def test_cancel_orders_never_closes_or_removes_protection(operation_harness):
    operation = await execute(operation_harness, await prepare(operation_harness, kind="cancel_orders"))
    assert operation["status"] == "completed"
    assert set(operation_harness.venue.calls) <= {"read", "cancel_ordinary"}
    assert operation_harness.venue.protection


async def test_prepare_returns_before_waiting_for_inflight_owner(operation_harness):
    h = operation_harness
    h.runtime._leases_drained.clear()
    response = await h.client.post(
        "/api/accounts/okx-demo/operations/prepare",
        json={
            "pair": "BTC/USDT",
            "kind": "flatten",
            "expected_revision": 1,
            "confirm_stop": True,
        },
    )
    assert response.status_code == 202
    await asyncio.sleep(0.02)
    assert h.venue.calls == []
    # The admitted strategy's final fill becomes visible before its pool lease drains.
    h.venue.amount = h.venue.available = Decimal("3")
    h.runtime._leases_drained.set()
    operation = await wait_operation(h.client, response.json()["operation_id"])
    assert operation["status"] == "awaiting_confirmation"
    assert operation["plan"]["position_amount"] == "3"
    assert operation["plan"]["close_amount"] == "3"


async def test_exit_final_sync_joins_runtime_background_history_batch(operation_harness):
    from datetime import timedelta

    from cryptotrader.accounts.models import FillPage, FundingPage
    from tests.fakes.account_session import START

    h = operation_harness
    old = AccountSession("okx-demo")
    h.runtime.account_sync.session_provider = old.provide
    new_end = END + timedelta(hours=1)
    h.venue.snapshot = replace(h.venue.snapshot, observed_at=new_end)
    paused, release, final_sync = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_fetch = old.fetch_fills
    original_sync = h.runtime.account_sync.sync_session

    async def paused_fills(cursor):
        page = await original_fetch(cursor)
        paused.set()
        await release.wait()
        return page

    async def shared_sync(session, connection_id):
        if session is h.venue:
            final_sync.set()
        return await original_sync(session, connection_id)

    async def new_fills(cursor):
        final_sync.set()
        h.venue.reads.append(("fills", cursor))
        return FillPage(h.venue.fills, "manual-fills", True, END if cursor else START, new_end)

    async def new_funding(cursor):
        h.venue.reads.append(("funding", cursor))
        return FundingPage(h.venue.funding, "manual-fund", True, END if cursor else START, new_end)

    old.fetch_fills = paused_fills
    h.runtime.account_sync.sync_session = shared_sync
    h.venue.fetch_fills, h.venue.fetch_funding = new_fills, new_funding
    operation = await prepare(h)
    background = asyncio.create_task(h.runtime.account_sync.sync("okx-demo"))
    await asyncio.wait_for(paused.wait(), 1)
    execution = asyncio.create_task(execute(h, operation))
    try:
        await asyncio.wait_for(final_sync.wait(), 1)
        await asyncio.sleep(0.03)
        reads_before_release = list(h.venue.reads)
    finally:
        release.set()
        _, completed = await asyncio.wait_for(asyncio.gather(background, execution), 2)
    assert reads_before_release == []
    assert completed["status"] == "completed"
    assert h.venue.reads == [("fills", "fills-end"), ("funding", "fund-end")]
    store = h.runtime.repository.account_store
    coverage = (await store.status("okx-demo"))["coverage"]
    assert coverage["fills"]["cursor"] == "manual-fills"
    assert coverage["funding"]["cursor"] == "manual-fund"
    assert coverage["fills"]["coverage_end"] == new_end.isoformat()
    assert coverage["funding"]["coverage_end"] == new_end.isoformat()
    assert (await store.latest("okx-demo")).observed_at == new_end
    assert "manual-fill" in {fill.venue_fill_id for fill in await store.history("okx-demo")}


async def test_protection_audit_uses_actual_order_identity_not_adapter_reference(operation_harness):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from cryptotrader.execution.service import VenueExecutionService

    h = operation_harness
    h.venue.replace_protection = AsyncMock(
        return_value=SimpleNamespace(
            protection_ids=("new-platform-logical-reference",), actual_order_ids=("actual-protection-order",)
        )
    )
    document = (await h.runtime.repository.get_existing()).document
    executor = VenueExecutionService(
        h.venue, connection=document.execution.connections[0], account_store=h.runtime.repository.account_store
    )
    plan = SimpleNamespace(connection_id="okx-demo", book_id="simulation", decision_id="decision")
    await executor._replace_protection(plan, None, [])
    binding = await h.runtime.repository.account_store.attribution("okx-demo", "actual-protection-order", None, None)
    assert binding["decision_id"] == "decision"
    logical = await h.runtime.repository.account_store.attribution(
        "okx-demo", "new-platform-logical-reference", None, None
    )
    assert logical["decision_id"] is None


async def test_explicit_connection_stop_stops_owned_pool_without_order_writes(operation_harness):
    h = operation_harness
    document = (await h.runtime.repository.get_existing()).document
    connection = document.execution.connections[0]
    body = {k: v for k, v in connection.__dict__.items() if k not in {"id", "credential_ref"}}
    body["parameters"] = dict(connection.parameters)
    response = await h.client.put(
        "/api/venue-connections/okx-demo", json={**body, "enabled": False, "expected_revision": 1, "confirm_stop": True}
    )
    assert response.status_code == 200, response.text
    updated = (await h.runtime.repository.get_existing()).document
    assert not updated.execution.books[0].enabled
    assert updated.execution.books[1].enabled
    assert updated.execution.books[0].allocations == document.execution.books[0].allocations
    assert not h.venue.calls


async def test_spot_only_sells_confirmed_available_amount_and_keeps_remainder_protected(operation_harness):
    h = operation_harness
    h.venue.available = Decimal("1")
    operation = await prepare(h)
    assert operation["plan"]["close_amount"] == "1"
    final = await execute(h, operation)
    assert final["status"] == "failed"
    assert final["result"]["orders"][0]["amount"] == "1"
    assert final["result"]["remaining_position"] == "1"
    assert h.venue.protection


async def test_real_account_requires_explicit_authorization_even_when_stopped(operation_harness):
    h = operation_harness
    h.venue = ExitSession("okx-live")
    h.venue.snapshot = replace(h.venue.snapshot, capital_scope="real")
    h.runtime.account_session = h.venue.provide
    response = await h.client.post(
        "/api/accounts/okx-live/operations/prepare",
        json={"pair": "BTC/USDT", "kind": "flatten", "expected_revision": 1, "confirm_stop": True},
    )
    assert response.status_code == 202
    operation = await wait_operation(h.client, response.json()["operation_id"])
    assert operation["status"] == "awaiting_confirmation"
    response = await h.client.post(
        f"/api/account-operations/{operation['operation_id']}/execute", json={"plan_version": 1}
    )
    assert response.status_code == 403
    assert "cancel_ordinary" not in h.venue.calls


async def test_stale_version_and_duplicate_execute_never_repeat_orders(operation_harness):
    h = operation_harness
    operation = await prepare(h)
    url = f"/api/account-operations/{operation['operation_id']}/execute"
    assert (await h.client.post(url, json={"plan_version": 2})).status_code == 409
    first, second = await asyncio.gather(
        h.client.post(url, json={"plan_version": 1}), h.client.post(url, json={"plan_version": 1})
    )
    assert sorted([first.status_code, second.status_code]) == [202, 409]
    assert (await wait_operation(h.client, operation["operation_id"]))["status"] == "completed"
    assert h.venue.calls.count("close_spot") == 1


async def test_unassigned_exit_stops_connection_and_keeps_all_pools_running(operation_harness):
    h = operation_harness
    h.venue = ExitSession("paper-spare")
    h.runtime.account_session = h.venue.provide
    response = await h.client.post(
        "/api/accounts/paper-spare/operations/prepare",
        json={"pair": "BTC/USDT", "kind": "flatten", "expected_revision": 1, "confirm_stop": True},
    )
    assert response.status_code == 202
    operation = await wait_operation(h.client, response.json()["operation_id"])
    assert operation["status"] == "awaiting_confirmation"
    assert operation["plan"]["stopped_scope"] == ["connection:paper-spare"]
    current = (await h.runtime.repository.get_existing()).document
    assert all(b.enabled for b in current.execution.books)
    assert not next(c for c in current.execution.connections if c.id == "paper-spare").enabled
    assert h.runtime.ownership_calls == ["paper-spare"]


async def test_restart_marks_interrupted_work_failed_without_resubmitting(operation_harness):
    from cryptotrader.accounts.store import AccountOperationStore

    h = operation_harness
    operation = await prepare(h)
    store = AccountOperationStore(h.runtime.repository.database_url)
    saved = await store.get(operation["operation_id"])
    await store.update(saved.model_copy(update={"status": "executing"}), expected_status="awaiting_confirmation")
    await AccountOperationStore(store.database_url).recover_interrupted()
    recovered = await store.get(operation["operation_id"])
    assert recovered.status == "failed"
    assert recovered.result.reconciliation_required
    assert set(h.venue.calls) == {"read"}
