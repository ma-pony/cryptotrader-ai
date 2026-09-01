"""Unknown history is not a fabricated zero return."""


async def _migrated_url(path):
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    database_url = f"sqlite+aiosqlite:///{path}"
    await migrate_workbench_schema(database_url)
    return database_url


async def test_unobserved_account_income_has_unknown_amount_and_reason(api_harness):
    response = await api_harness.client.get(
        "/api/accounts/paper-spare/income",
        params={"start": "2026-08-01T00:00:00Z", "end": "2026-08-31T00:00:00Z"},
    )
    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["net_trading"][0]["amount"] is None
    assert summary["net_trading"][0]["unavailable_reason"]
    assert summary["completeness"]
    assert summary["unrealized_as_of"] is None


async def test_moving_average_keeps_fee_funding_and_unrealized_separate(tmp_path):
    from decimal import Decimal

    from cryptotrader.accounts.income import IncomeService
    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.accounts.sync import AccountSyncService
    from tests.fakes.account_session import END, START, AccountSession

    store = AccountStore(await _migrated_url(tmp_path / "income.db"))
    session = AccountSession()
    await AccountSyncService(store, session.provide).sync("sim-a")
    summary = await IncomeService(AccountStore(store.database_url)).summary("sim-a", START, END)
    assert summary.realized_gross[0].amount == Decimal("30")
    assert summary.fees[0].amount == Decimal("3")
    assert summary.fees[0].currency == "USDT"
    assert summary.funding[0].amount == Decimal("2")
    assert summary.net_trading[0].amount == Decimal("29")
    assert summary.unrealized[0].unavailable_reason == "缺少当前估值"
    assert summary.completeness == ()


async def test_historical_income_identifies_current_unrealized_valuation_time(tmp_path):
    from dataclasses import replace
    from datetime import timedelta
    from decimal import Decimal

    from cryptotrader.accounts.income import IncomeService
    from cryptotrader.accounts.models import AccountPosition, Money
    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.accounts.sync import AccountSyncService
    from tests.fakes.account_session import END, INSTRUMENT, START, AccountSession

    store = AccountStore(await _migrated_url(tmp_path / "current-valuation.db"))
    session = AccountSession()
    await AccountSyncService(store, session.provide).sync("sim-a")
    recent = replace(
        session.snapshot,
        observed_at=END + timedelta(days=28),
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
    await store.ingest(recent)
    summary = await IncomeService(store).summary("sim-a", START, END)
    assert summary.net_trading[0].amount == Decimal("29")
    assert summary.unrealized[0].amount == Decimal("999")
    assert summary.unrealized_as_of == END + timedelta(days=28)
    no_valuation = await IncomeService(store).summary("missing", START, END)
    assert no_valuation.unrealized_as_of is None
    assert no_valuation.unrealized[0].amount is None


async def test_unknown_opening_cost_never_becomes_zero_or_equity_difference(tmp_path):
    from cryptotrader.accounts.income import IncomeService
    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.accounts.sync import AccountSyncService
    from tests.fakes.account_session import END, START, AccountSession

    store = AccountStore(await _migrated_url(tmp_path / "unknown.db"))
    session = AccountSession()
    session.history_from_inception = False
    await AccountSyncService(store, session.provide).sync("sim-a")
    summary = await IncomeService(store).summary("sim-a", START, END)
    assert summary.realized_gross[0].amount is None
    assert summary.net_trading[0].amount is None


async def test_external_known_settlement_is_used_once_even_with_unknown_opening_cost(tmp_path):
    from dataclasses import replace
    from decimal import Decimal

    from cryptotrader.accounts.income import IncomeService
    from cryptotrader.accounts.models import Money
    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.accounts.sync import AccountSyncService
    from tests.fakes.account_session import END, START, AccountSession

    store = AccountStore(await _migrated_url(tmp_path / "settlement.db"))
    session = AccountSession()
    session.history_from_inception = False
    session.fills = (
        replace(session.fills[2], realized_pnl=Money(Decimal("15"), "USDT"), fee=Money(Decimal("0.01"), "BTC")),
    )
    await AccountSyncService(store, session.provide).sync("sim-a")
    summary = await IncomeService(store).summary("sim-a", START, END)
    assert summary.realized_gross[0].amount == Decimal("15")
    assert {(item.currency, item.amount) for item in summary.net_trading} == {
        ("USDT", Decimal("17")),
        ("BTC", Decimal("-0.01")),
    }
