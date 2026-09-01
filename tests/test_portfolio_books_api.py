"""资金池 portfolio list/detail API 保持 simulated/real 隔离。"""

# ruff: noqa: F401, F811 - 导入 fixture 供本模块的 pytest 参数解析。

from __future__ import annotations

import pytest

from cryptotrader.pair import Pair
from tests.fakes.account_session import account_from_portfolio
from tests.test_runtime_config_api import api_harness

PAIR = Pair.parse("BTC/USDT:USDT")


@pytest.fixture(autouse=True)
async def saved_accounts(api_harness):
    for session in api_harness.runtime.sessions.values():
        await api_harness.runtime.repository.account_store.ingest(account_from_portfolio(session.snapshot))


async def test_portfolio_books_group_scopes_and_never_create_cross_scope_total(api_harness):
    response = await api_harness.client.get("/api/portfolio/books", params={"pair": PAIR.canonical()})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"simulated", "real"}
    assert [item["book_id"] for item in body["simulated"]["books"]] == ["simulation"]
    assert body["simulated"]["equity"] == [{"amount": "400", "currency": "USDT", "unavailable_reason": None}]
    assert body["simulated"]["books"][0]["total_signed_notional"] == [
        {"amount": "10", "currency": "USDT", "unavailable_reason": None}
    ]
    assert [item["book_id"] for item in body["real"]["books"]] == ["production"]
    assert body["real"]["equity"] == [{"amount": "1000", "currency": "USDT", "unavailable_reason": None}]
    assert body["real"]["books"][0]["total_signed_notional"] == [
        {"amount": "250", "currency": "USDT", "unavailable_reason": None}
    ]
    assert "total_equity" not in body
    assert "totals" not in body


async def test_portfolio_without_pair_keeps_full_account_instead_of_product_default(api_harness):
    response = await api_harness.client.get("/api/portfolio/books")

    assert response.status_code == 200
    assert "pair" not in response.json()
    assert (
        response.json()["simulated"]["books"][0]["connections"][0]["snapshot"]["positions"][0]["instrument"]["pair"]
        == PAIR.canonical()
    )


async def test_portfolio_book_detail_preserves_connection_breakdown(api_harness):
    response = await api_harness.client.get(
        "/api/portfolio/books/simulation",
        params={"pair": PAIR.canonical()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["book_id"] == "simulation"
    assert body["capital_scope"] == "simulated"
    assert [item["connection_id"] for item in body["connections"]] == ["okx-demo", "bybit-testnet"]
    assert body["connections"][0]["snapshot"]["positions"][0]["instrument"]["pair"] == PAIR.canonical()


async def test_portfolio_book_detail_returns_not_found_for_unknown_book(api_harness):
    response = await api_harness.client.get(
        "/api/portfolio/books/missing",
        params={"pair": PAIR.canonical()},
    )

    assert response.status_code == 404


async def test_portfolio_reads_history_without_an_active_runtime(api_harness):
    api_harness.runtime.cycle = None

    response = await api_harness.client.get("/api/portfolio/books", params={"pair": PAIR.canonical()})

    assert response.status_code == 200


async def test_portfolio_does_not_open_temporary_sessions(api_harness):
    before = {key: len(adapter.connect_calls) for key, adapter in api_harness.adapters.items()}

    response = await api_harness.client.get("/api/portfolio/books", params={"pair": PAIR.canonical()})

    assert response.status_code == 200
    assert {key: len(adapter.connect_calls) for key, adapter in api_harness.adapters.items()} == before


async def test_portfolio_preserves_stored_accounts_when_runtime_session_is_missing(api_harness):
    api_harness.runtime.sessions = {"okx-demo": api_harness.runtime.sessions["okx-demo"]}
    api_harness.runtime.cycle.sessions = api_harness.runtime.sessions

    response = await api_harness.client.get("/api/portfolio/books", params={"pair": PAIR.canonical()})

    assert response.status_code == 200
    assert "bybit-testnet" in response.text


async def test_risk_endpoints_read_persisted_peak_time_and_unknown_facts_without_account_io(api_harness, monkeypatch):
    from dataclasses import replace
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal

    from pydantic import ValidationError

    from api.routes.portfolio_books import AccountBookOut
    from cryptotrader.accounts.models import Money
    from cryptotrader.risk.book_state import BookRiskStateStore

    store = api_harness.runtime.repository.account_store
    risk_store = BookRiskStateStore(store)
    observed = datetime(2026, 9, 1, tzinfo=UTC)
    original = (await store.latest("okx-demo"), await store.latest("bybit-testnet"))
    peak = tuple(replace(item, equity=Money(Decimal("50"), "USDT"), observed_at=observed) for item in original)
    await risk_store.update("simulation", peak)
    fresh = tuple(
        replace(item, equity=Money(Decimal("40"), "USDT"), observed_at=observed + timedelta(minutes=1)) for item in peak
    )
    fresh = (
        replace(
            fresh[0],
            available_margin=Money(None, "USDT", "margin_not_reported"),
            completeness=("available_margin:margin_not_reported",),
        ),
        fresh[1],
    )
    for item in fresh:
        await store.ingest(item)
    await risk_store.update("simulation", fresh)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("GET risk state must not access accounts or submit orders")

    for session in api_harness.runtime.sessions.values():
        for name in (
            "fetch_account",
            "fetch_portfolio",
            "fetch_quote",
            "list_instruments",
            "place_order",
            "replace_protection",
        ):
            monkeypatch.setattr(session, name, forbidden, raising=False)

    detail = await api_harness.client.get("/api/portfolio/books/simulation")
    listing = await api_harness.client.get("/api/portfolio/books")
    assert detail.status_code == 200
    assert listing.status_code == 200
    body = detail.json()
    risk = body["risk_state"]
    assert risk == listing.json()["simulated"]["books"][0]["risk_state"]
    assert risk["peak_equity"] == "100"
    assert risk["equity"] == "80"
    assert risk["observed_at"] == "2026-09-01T00:01:00Z"
    assert risk["valuation_currency"] == "USDT"
    assert risk["available_margin"] is None
    assert "okx-demo:available_margin:margin_not_reported" in risk["completeness"]
    assert body["connections"][0]["snapshot"]["available_margin"]["unavailable_reason"] == "margin_not_reported"
    assert "snapshots" not in risk
    assert AccountBookOut.model_validate(body).risk_state.peak_equity == Decimal("100")
    with pytest.raises(ValidationError):
        AccountBookOut.model_validate(body | {"risk_state": risk | {"unexpected_fact": "not allowed"}})
