"""Offline account HTTP contracts: reading never opens a trading session."""

import pytest


async def test_accounts_include_unallocated_disabled_connections_without_sync(api_harness):
    api_harness.runtime.cycle = None
    response = await api_harness.client.get("/api/accounts")
    assert response.status_code == 200, response.text
    rows = response.json()["items"]
    assert {row["connection_id"] for row in rows} == {"okx-demo", "bybit-testnet", "okx-live", "paper-spare"}
    assert all(row["snapshot"] is None for row in rows)
    assert all(not adapter.connect_calls for adapter in api_harness.adapters.values())


async def test_sync_interval_is_database_configuration(api_harness):
    response = await api_harness.client.get("/api/config")
    assert response.json()["document"].get("accounts") == {"sync_interval_seconds": 60}
    value = response.json()
    from tests.test_runtime_config_api import active_payload

    document = active_payload()
    document["accounts"]["sync_interval_seconds"] = 120
    saved = await api_harness.client.put(
        "/api/config", json={"expected_revision": value["revision"], "document": document}
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["document"]["accounts"]["sync_interval_seconds"] == 120


async def test_account_detail_is_readable_without_automatic_runtime(api_harness):
    api_harness.runtime.cycle = None
    response = await api_harness.client.get("/api/accounts/paper-spare")
    assert response.status_code == 200, response.text
    assert response.json()["connection_id"] == "paper-spare"
    assert response.json()["last_success_at"] is None
    assert all(not adapter.connect_calls for adapter in api_harness.adapters.values())


async def test_manual_sync_refreshes_only_facts_and_history_paginates(api_harness):
    from cryptotrader.accounts.sync import AccountSyncService
    from tests.fakes.account_session import START, AccountSession

    original = await api_harness.runtime.repository.get_existing()
    session = AccountSession("paper-spare")
    api_harness.runtime.account_sync = AccountSyncService(api_harness.runtime.repository.account_store, session.provide)
    refreshed = await api_harness.client.post("/api/accounts/paper-spare/sync")
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["last_success_at"]
    assert await api_harness.runtime.repository.get_existing() == original
    response = await api_harness.client.get(
        "/api/accounts/paper-spare/fills",
        params={"start": START.isoformat(), "symbol": "BTC/USDT", "offset": 1, "limit": 1},
    )
    assert response.status_code == 200, response.text
    assert response.json()["total"] == 3
    assert [fill["venue_fill_id"] for fill in response.json()["items"]] == [session.fills[1].venue_fill_id]


async def test_book_reads_persisted_full_account_when_disabled_without_sessions(api_harness):
    from dataclasses import replace

    from tests.fakes.account_session import snapshot

    store = api_harness.runtime.repository.account_store
    await store.ingest(snapshot("okx-demo", amount="100", currency="USD"))
    await store.ingest(snapshot("bybit-testnet", amount="200", currency="USDT"))
    current = await api_harness.runtime.repository.get_existing()
    document = current.document.model_copy(
        update={
            "execution": current.document.execution.model_copy(
                update={"books": tuple(replace(book, enabled=False) for book in current.document.execution.books)}
            )
        }
    )
    await api_harness.runtime.repository.replace(current.revision, document)
    api_harness.runtime.cycle = None
    api_harness.runtime.sessions = {}
    response = await api_harness.client.get("/api/portfolio/books/simulation")
    assert response.status_code == 200, response.text
    assert response.json()["enabled"] is False
    assert {(item["currency"], item["amount"]) for item in response.json()["total_equity"]} == {
        ("USD", "100"),
        ("USDT", "200"),
    }


@pytest.mark.parametrize(
    ("case", "expected_book"), [("unselected", None), ("active_wins", "primary"), ("ambiguous", None)]
)
async def test_account_and_book_reads_share_effective_membership(api_harness, case, expected_book):
    from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
    from tests.fakes.account_session import snapshot

    store = api_harness.runtime.repository.account_store
    await store.ingest(snapshot("paper-spare", amount="100", currency="USDT"))
    primary = ExecutionBook(
        "primary",
        "主池",
        "simulated",
        case == "active_wins",
        False,
        (ConnectionAllocation("paper-spare", case != "unselected", 0.0 if case == "unselected" else 1.0),),
    )
    draft = ExecutionBook(
        "draft", "停用草稿", "simulated", False, False, (ConnectionAllocation("paper-spare", True, 1.0),)
    )
    books = (primary,) if case == "unselected" else (primary, draft)
    current = await api_harness.runtime.repository.get_existing()
    document = current.document.model_copy(
        update={"execution": current.document.execution.model_copy(update={"books": books})}
    )
    saved = await api_harness.runtime.repository.replace(current.revision, document)
    account = (await api_harness.client.get("/api/accounts/paper-spare")).json()
    assert account["book_ids"] == (["primary"] if expected_book else [])
    attributed = await store.attribution("paper-spare", "external-order", None, saved.updated_at)
    assert attributed["book_id"] == expected_book
    listed = (await api_harness.client.get("/api/portfolio/books")).json()["simulated"]
    assert len(listed["books"]) == len(books)  # disabled pools remain visible
    assert listed["equity"] == (
        [{"amount": "100", "currency": "USDT", "unavailable_reason": None}] if expected_book else []
    )
    for book in listed["books"]:
        expected = ["paper-spare"] if book["book_id"] == expected_book else []
        assert [item["connection_id"] for item in book["connections"]] == expected
        if not expected:
            assert book["total_equity"] == []
            assert book["total_signed_notional"] == []
    assert all(not adapter.connect_calls for adapter in api_harness.adapters.values())


async def test_orders_keep_last_proven_status_and_exact_decision_link(api_harness):
    from dataclasses import replace
    from decimal import Decimal

    from cryptotrader.accounts.models import AccountOrder, Money
    from tests.fakes.account_session import END, INSTRUMENT, AccountSession, snapshot

    store = api_harness.runtime.repository.account_store
    order = AccountOrder(
        "paper-spare",
        "actual-order",
        INSTRUMENT,
        "buy",
        "limit",
        Decimal("2"),
        Decimal("0"),
        None,
        "open",
        False,
        False,
        "actual-client",
        END,
        Money(Decimal("200"), "USDT"),
    )
    await store.ingest(replace(snapshot("paper-spare"), orders=(order,)))
    await store.bind_order(
        "paper-spare",
        "actual-client",
        book_id="simulation",
        decision_id="decision-exact",
        venue_order_id="actual-order",
    )
    await store.ingest(snapshot("paper-spare"))
    response = await api_harness.client.get("/api/accounts/paper-spare")
    saved = response.json()["orders"][0]
    assert saved["status"] == "open"  # disappearance is not proof of cancellation/fill
    assert saved["currently_open"] is False
    assert saved["attribution"]["decision_id"] == "decision-exact"
    fill = replace(AccountSession("paper-spare").fills[0], venue_order_id="actual-order")
    await store.ingest(snapshot("paper-spare"), (fill,))
    response = await api_harness.client.get("/api/accounts/paper-spare")
    assert response.json()["orders"][0]["status"] == "filled"
    assert response.json()["orders"][0]["filled_amount"] == "2"


async def test_income_default_end_is_persisted_snapshot_not_wall_clock(api_harness):
    from cryptotrader.accounts.sync import AccountSyncService
    from tests.fakes.account_session import END, START, AccountSession

    session = AccountSession("paper-spare")
    await AccountSyncService(api_harness.runtime.repository.account_store, session.provide).sync("paper-spare")
    response = await api_harness.client.get("/api/accounts/paper-spare/income", params={"start": START.isoformat()})
    assert response.status_code == 200
    assert response.json()["net_trading"] == [{"amount": "29", "currency": "USDT", "unavailable_reason": None}]
    assert response.json()["end"].replace("Z", "+00:00") == END.isoformat()
    explicit = await api_harness.client.get(
        "/api/accounts/paper-spare/income", params={"start": START.isoformat(), "end": "2026-08-04T00:00:00Z"}
    )
    assert explicit.json()["net_trading"][0]["amount"] is None


async def test_historical_income_api_keeps_current_valuation_time_separate(api_harness):
    from dataclasses import replace
    from datetime import timedelta
    from decimal import Decimal

    from cryptotrader.accounts.models import AccountPosition, Money
    from cryptotrader.accounts.sync import AccountSyncService
    from tests.fakes.account_session import END, INSTRUMENT, START, AccountSession

    session = AccountSession("paper-spare")
    store = api_harness.runtime.repository.account_store
    await AccountSyncService(store, session.provide).sync("paper-spare")
    current = END + timedelta(days=28)
    await store.ingest(
        replace(
            session.snapshot,
            observed_at=current,
            positions=(
                AccountPosition(
                    INSTRUMENT,
                    Decimal("2"),
                    Decimal("2"),
                    Money(Decimal("1200"), "USDT"),
                    Decimal("100"),
                    Money(Decimal("999"), "USDT"),
                ),
            ),
        )
    )
    response = await api_harness.client.get(
        "/api/accounts/paper-spare/income",
        params={
            "start": START.isoformat(),
            "end": END.isoformat(),
        },
    )
    assert response.status_code == 200
    summary = response.json()
    assert summary["net_trading"][0]["amount"] == "29"
    assert summary["unrealized"][0]["amount"] == "999"
    assert summary["end"].replace("Z", "+00:00") == END.isoformat()
    assert summary["unrealized_as_of"].replace("Z", "+00:00") == current.isoformat()
