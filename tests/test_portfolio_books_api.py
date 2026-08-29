"""资金池 portfolio list/detail API 保持 simulated/real 隔离。"""

# ruff: noqa: F401, F811 - 导入 fixture 供本模块的 pytest 参数解析。

from __future__ import annotations

from cryptotrader.pair import Pair
from tests.test_runtime_config_api import api_harness

PAIR = Pair.parse("BTC/USDT:USDT")


async def test_portfolio_books_group_scopes_and_never_create_cross_scope_total(api_harness):
    response = await api_harness.client.get("/api/portfolio/books", params={"pair": PAIR.canonical()})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"pair", "simulated", "real"}
    assert [item["book_id"] for item in body["simulated"]["books"]] == ["simulation"]
    assert body["simulated"]["totals"] == {"equity": "400", "signed_notional": "10"}
    assert [item["book_id"] for item in body["real"]["books"]] == ["production"]
    assert body["real"]["totals"] == {"equity": "1000", "signed_notional": "250"}
    assert "total_equity" not in body
    assert "totals" not in body


async def test_portfolio_without_pair_uses_product_default_and_echoes_canonical_pair(api_harness):
    response = await api_harness.client.get("/api/portfolio/books")

    assert response.status_code == 200
    assert response.json()["pair"] == "BTC/USDT"


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
    assert body["connections"][0]["position"]["pair"] == PAIR.canonical()


async def test_portfolio_book_detail_returns_not_found_for_unknown_book(api_harness):
    response = await api_harness.client.get(
        "/api/portfolio/books/missing",
        params={"pair": PAIR.canonical()},
    )

    assert response.status_code == 404


async def test_portfolio_requires_an_active_runtime(api_harness):
    api_harness.runtime.cycle = None

    response = await api_harness.client.get("/api/portfolio/books", params={"pair": PAIR.canonical()})

    assert response.status_code == 503


async def test_portfolio_does_not_open_temporary_sessions(api_harness):
    before = {key: len(adapter.connect_calls) for key, adapter in api_harness.adapters.items()}

    response = await api_harness.client.get("/api/portfolio/books", params={"pair": PAIR.canonical()})

    assert response.status_code == 200
    assert {key: len(adapter.connect_calls) for key, adapter in api_harness.adapters.items()} == before


async def test_portfolio_returns_service_unavailable_when_runtime_session_is_missing(api_harness):
    api_harness.runtime.sessions = {"okx-demo": api_harness.runtime.sessions["okx-demo"]}
    api_harness.runtime.cycle.sessions = api_harness.runtime.sessions

    response = await api_harness.client.get("/api/portfolio/books", params={"pair": PAIR.canonical()})

    assert response.status_code == 503
    assert "bybit-testnet" not in response.text
