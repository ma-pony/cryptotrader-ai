"""Full-account reads must preserve facts, provenance and never execute trades."""

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from cryptotrader.pair import Pair
from cryptotrader.runtime_config.secrets import CredentialPayload
from cryptotrader.venues.models import OrderIntent, ProtectionSpec
from cryptotrader.venues.paper import PaperVenueAdapter
from tests.factories.runtime_config import connection
from tests.factories.workbench_extensions import SampleVenue
from tests.fakes.account_client import account_session


def require_reads(session):
    for name in ("fetch_account", "list_instruments", "fetch_fills", "fetch_funding"):
        assert callable(getattr(session, name, None)), f"full-account read missing: {name}"


@pytest.mark.asyncio
async def test_sample_reads_preserve_unmapped_positions_orders_money_and_no_writes():
    session = await SampleVenue().connect(
        connection("sample-a", "sandbox", adapter_id="sample_venue"),
        CredentialPayload(values={"access_token": "local-fixture"}),
    )
    require_reads(session)
    from cryptotrader.venues.protocol import VenueSession

    assert isinstance(session, VenueSession)
    snapshot = await session.fetch_account()
    assert snapshot.connection_id == "sample-a"
    assert snapshot.capital_scope == "simulated"
    assert {p.instrument.venue_symbol for p in snapshot.positions} == {"BTCUSDT", "ETHUSDT", "UNKNOWN"}
    assert snapshot.positions[-1].instrument.pair is None
    assert snapshot.positions[-1].instrument.tradable is False
    assert snapshot.used_margin.amount is None
    assert snapshot.used_margin.unavailable_reason
    assert {o.protection for o in snapshot.orders} == {True, False}
    assert all(o.connection_id == "sample-a" for o in snapshot.orders)
    assert (await session.fetch_fills(None)).items[0].fee.amount == Decimal("0.1")
    assert (await session.fetch_funding(None)).items[0].amount.amount == Decimal("-0.2")
    assert len(await session.list_instruments()) == 3
    assert session.write_calls == []
    with pytest.raises(FrozenInstanceError):
        snapshot.connection_id = "other"


@pytest.mark.asyncio
async def test_paper_account_read_is_multi_asset_and_does_not_trigger_protection():
    adapter = PaperVenueAdapter()
    config = connection("paper-a", "paper", parameters={"initial_equity": "20000"})
    session = await adapter.connect(config, None)
    btc, eth = Pair.parse("BTC/USDT:USDT"), Pair.parse("ETH/USDT")
    await session.set_quote(btc, Decimal("100"))
    await session.set_quote(eth, Decimal("20"))
    order = await session.place_order(OrderIntent(btc, "buy", Decimal("2"), "market", None, False, "owned1"))
    await session.place_order(OrderIntent(eth, "buy", Decimal("3"), "market", None, False, "owned2"))
    protection = await session.replace_protection(ProtectionSpec(btc, "long", Decimal("2"), Decimal("90"), None))
    await session.set_quote(btc, Decimal("85"))
    require_reads(session)
    snapshot = await session.fetch_account()
    assert len(snapshot.positions) == 2
    assert snapshot.equity.amount == Decimal("19970")
    assert snapshot.equity.currency == "USDT"
    assert snapshot.used_margin.amount == Decimal("170")
    assert snapshot.available_margin.amount == Decimal("19800")
    assert snapshot.orders[0].venue_order_id == protection.protection_ids[0]
    assert snapshot.orders[0].protection is True
    fills = await session.fetch_fills(None)
    assert len(fills.items) == 2
    assert fills.items[0].venue_order_id == order.id
    assert fills.items[0].venue_fill_id
    assert fills.items[0].fee.amount == 0
    assert session._account.orders[order.id].client_order_id == "owned1"
    assert (await session.fetch_funding(None)).complete is True
    assert (await session.fetch_account()).positions == snapshot.positions
    assert len(await session.list_instruments()) == 2
    await session.close()
    reopened = await adapter.connect(config, None)
    assert (await reopened.fetch_account()).equity.amount == Decimal("19970")
    assert len((await reopened.fetch_fills(None)).items) == 2
    other = await adapter.connect(connection("paper-b", "paper", parameters={"initial_equity": "20000"}), None)
    assert not (await other.fetch_account()).positions
    assert (await other.fetch_account()).equity.amount == Decimal("20000")


@pytest.mark.asyncio
async def test_paper_fills_record_average_cost_realized_pnl_only_when_actually_filled():
    session = await PaperVenueAdapter().connect(connection("paper", "paper"), None)
    pair = Pair.parse("BTC/USDT:USDT")
    await session.set_quote(pair, Decimal("100"))
    await session.place_order(OrderIntent(pair, "buy", Decimal("2"), "market", None, False))
    await session.set_quote(pair, Decimal("120"))
    await session.place_order(OrderIntent(pair, "buy", Decimal("2"), "market", None, False))
    await session.place_order(OrderIntent(pair, "sell", Decimal("1"), "market", None, True))
    await session.place_order(OrderIntent(pair, "sell", Decimal("100"), "market", None, True))
    require_reads(session)
    fills = (await session.fetch_fills(None)).items
    assert len(fills) == 3
    assert fills[-1].realized_pnl.amount == Decimal("10")
    assert fills[-1].realized_pnl.currency == "USDT"
    assert fills[-1].source == "local_calculation"


@pytest.mark.asyncio
async def test_unknown_money_and_external_order_source_are_explicit():
    session = await PaperVenueAdapter().connect(connection("paper", "paper"), None)
    require_reads(session)
    from cryptotrader.accounts.models import Money, external_order_source

    with pytest.raises(ValueError, match="reason"):
        Money(None, "USD", None)
    with pytest.raises(ValueError):
        Money(Decimal("NaN"), "USD", None)
    assert external_order_source("owned", {"owned"}) == "strategy"
    assert external_order_source(None, {"owned"}) == "external"
    assert external_order_source("manual", {"owned"}) == "external"


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_id", ["okx", "bybit"])
async def test_external_account_reads_keep_all_positions_unknown_orders_and_real_currency(adapter_id):
    session, client = await account_session(adapter_id)
    require_reads(session)
    snapshot = await session.fetch_account()
    positions = {p.instrument.venue_symbol: p for p in snapshot.positions}
    expected = (
        {"BTCUSDT", "ETHUSDT", "UNKNOWN"} if adapter_id == "bybit" else {"BTC-USDT-SWAP", "ETH-USDT-SWAP", "UNKNOWN"}
    )
    assert expected <= positions.keys()
    btc = positions["BTCUSDT" if adapter_id == "bybit" else "BTC-USDT-SWAP"]
    assert btc.signed_amount == Decimal("0.02")
    assert positions["ETHUSDT" if adapter_id == "bybit" else "ETH-USDT-SWAP"].signed_amount == Decimal("-0.03")
    assert positions["UNKNOWN"].instrument.pair is None
    assert not positions["UNKNOWN"].instrument.tradable
    assert positions["UNKNOWN"].signed_notional.amount is None
    assert snapshot.equity.currency == "USD"
    assert snapshot.used_margin.amount is None
    assert snapshot.used_margin.unavailable_reason
    assert snapshot.available_margin.amount is None
    assert snapshot.completeness
    assert {o.venue_order_id for o in snapshot.orders} == {"manual", "protect1"}
    assert {o.protection for o in snapshot.orders} == {False, True}
    assert next(o for o in snapshot.orders if o.venue_order_id == "manual").instrument.pair is None
    assert all(
        "post" not in name and "create" not in name and "cancel" not in name and "set_" not in name
        for name, _ in client.calls
    )
    if adapter_id == "bybit":
        positions_calls = [params for name, params in client.calls if name == "account_positions"]
        assert {p["category"] for p in positions_calls} == {"linear", "inverse", "option"}
        assert {p.get("settleCoin") for p in positions_calls if p["category"] == "linear"} == {"USDT", "USDC"}
        assert any(p.get("cursor") == "position-page-2" for p in positions_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_id", ["okx", "bybit"])
@pytest.mark.parametrize(("price", "expected"), [("100", Decimal("1")), ("", None)])
async def test_pending_order_value_uses_remaining_quantity_and_limit_not_average(adapter_id, price, expected):
    from unittest.mock import AsyncMock

    session, client = await account_session(adapter_id)
    if adapter_id == "okx":
        client.private_get_account_positions = AsyncMock(return_value={"code": "0", "data": []})
        client.private_get_trade_orders_pending = AsyncMock(
            side_effect=[
                {
                    "code": "0",
                    "data": [
                        {
                            "ordId": "limit-no-position",
                            "instId": "BTC-USDT-SWAP",
                            "instType": "SWAP",
                            "side": "buy",
                            "ordType": "limit",
                            "sz": "2",
                            "accFillSz": "1",
                            "avgPx": "999",
                            "px": price,
                            "state": "live",
                            "reduceOnly": "false",
                        }
                    ],
                },
                {"code": "0", "data": []},
            ]
        )
    else:
        client.private_get_v5_position_list = AsyncMock(
            return_value={"retCode": 0, "result": {"list": [], "nextPageCursor": ""}}
        )

        async def orders(params):
            rows = (
                [
                    {
                        "orderId": "limit-no-position",
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "orderType": "Limit",
                        "qty": "0.02",
                        "cumExecQty": "0.01",
                        "avgPrice": "999",
                        "price": price,
                        "orderStatus": "PartiallyFilled",
                        "reduceOnly": False,
                    }
                ]
                if params["category"] == "linear" and params.get("settleCoin") == "USDT"
                else []
            )
            return {"retCode": 0, "result": {"list": rows, "nextPageCursor": ""}}

        client.private_get_v5_order_realtime = orders
    account = await session.fetch_account()
    assert not any(p.instrument.pair == Pair.parse("BTC/USDT:USDT") for p in account.positions)
    order = next(o for o in account.orders if o.venue_order_id == "limit-no-position")
    assert order.amount == Decimal("0.02")
    assert order.filled_amount == Decimal("0.01")
    assert order.average_price == Decimal("999")
    assert order.remaining_notional.amount == expected
    assert order.remaining_notional.currency == "USDT"
    assert bool(order.remaining_notional.unavailable_reason) == (expected is None)


@pytest.mark.asyncio
async def test_remaining_order_value_cannot_be_negative():
    from dataclasses import replace

    from cryptotrader.accounts.models import Money

    session, _ = await account_session("bybit")
    order = (await session.fetch_account()).orders[0]
    with pytest.raises(ValueError, match="remaining"):
        replace(order, remaining_notional=Money(Decimal("-1"), "USDT"))


async def collect_pages(method):
    cursor, items, pages = None, [], []
    for _ in range(20):
        page = await method(cursor)
        items.extend(page.items)
        pages.append(page)
        cursor = page.next_cursor
        if page.complete:
            return items, pages
        assert cursor
    pytest.fail("history failed to finish its bounded range")


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_id", ["okx", "bybit"])
async def test_history_keeps_same_timestamp_fills_fees_and_resumable_coverage(adapter_id):
    import json

    session, client = await account_session(adapter_id)
    require_reads(session)
    fills, pages = await collect_pages(session.fetch_fills)
    main = [fill for fill in fills if fill.venue_order_id == "order1"]
    assert len(main) == 2
    assert len({f.venue_fill_id for f in main}) == 2
    assert main[0].occurred_at == main[1].occurred_at
    assert main[0].fee.amount == Decimal("0.5")
    assert main[0].fee.currency == "USDT"
    assert main[0].amount == Decimal("0.02")
    assert main[0].client_order_id == "owned1"
    assert main[0].realized_pnl.amount == (Decimal("20") if adapter_id == "okx" else None)
    assert pages[-1].next_cursor
    assert pages[-1].coverage_start
    assert pages[-1].coverage_end
    if adapter_id == "bybit":
        assert any(f.venue_fill_id == "eth-option-fill" for f in fills)
    checkpoint = json.loads(pages[-1].next_cursor)
    assert checkpoint["complete"] is True
    continuation = await session.fetch_fills(pages[-1].next_cursor)
    assert continuation.coverage_start == pages[-1].coverage_end
    times = {(p.coverage_start, p.coverage_end) for p in pages}
    assert len(times) == 1
    history = [params for kind, params in client.history_calls if kind == "fills"]
    begin_key, end_key = ("begin", "end") if adapter_id == "okx" else ("startTime", "endTime")
    assert len({(params[begin_key], params[end_key]) for params in history[:-1]}) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_id", ["okx", "bybit"])
async def test_funding_uses_actual_income_direction_currency_and_id_pagination(adapter_id):
    session, _ = await account_session(adapter_id)
    require_reads(session)
    items, pages = await collect_pages(session.fetch_funding)
    assert [(item.venue_entry_id, item.amount.amount, item.amount.currency) for item in items] == [
        ("fund1", Decimal("-2"), "USDT"),
        ("fund2", Decimal("0.1"), "ETH" if adapter_id == "okx" else "USDC"),
    ]
    assert pages[-1].complete
    assert pages[-1].next_cursor


@pytest.mark.parametrize(("adapter_id", "environment"), [("paper", "paper"), ("okx", "demo"), ("bybit", "testnet")])
def test_registered_capabilities_expose_read_and_exit_support_in_strict_dto(adapter_id, environment):
    from api.routes.config import venue_definition

    output = venue_definition(adapter_id, environment)
    capabilities = output.capabilities.model_dump()
    assert capabilities.get("account_reads") == ["balances", "fills", "funding", "instruments", "orders", "positions"]
    assert capabilities.get("exit_operations") == ["cancel_order", "cancel_protection", "close_position"]
    assert capabilities.get("history_initial_days") == (None if adapter_id == "paper" else 7)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_id", ["okx", "bybit"])
async def test_completed_history_checkpoint_catches_up_without_skipping_old_windows(adapter_id, monkeypatch):
    import json

    from cryptotrader.venues import account_reads
    from cryptotrader.venues.protocol import VenueOperationError

    now = 1788177600000
    monkeypatch.setattr(account_reads, "milliseconds", lambda: now)
    session, client = await account_session(adapter_id)
    _, pages = await collect_pages(session.fetch_fills)
    old_end = pages[-1].coverage_end
    now += 20 * account_reads.DAY_MS
    resumed = await session.fetch_fills(pages[-1].next_cursor)
    assert resumed.coverage_start == old_end
    assert (resumed.coverage_end - resumed.coverage_start).days == 7
    checkpoint = json.loads(resumed.next_cursor)
    assert checkpoint["end"] == 1788782400000
    with pytest.raises(VenueOperationError, match="cursor"):
        await session.fetch_funding(pages[-1].next_cursor)
    checkpoint["begin"] = now - 1000 * account_reads.DAY_MS
    checkpoint["end"] = checkpoint["begin"] + account_reads.DAY_MS
    checkpoint["complete"] = False
    with pytest.raises(VenueOperationError, match="retention"):
        await session.fetch_fills(json.dumps(checkpoint))
    assert not any("create" in name or "cancel" in name for name, _ in client.calls)


@pytest.mark.asyncio
async def test_account_collections_are_frozen_even_when_constructed_from_mutable_input():
    from dataclasses import replace

    session = await PaperVenueAdapter().connect(connection("paper", "paper"), None)
    positions = []
    snapshot = replace(await session.fetch_account(), positions=positions)
    positions.append("not-a-position")
    assert snapshot.positions == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["fills", "funding"])
async def test_paper_completed_checkpoint_resumes_coverage_across_sessions(kind):
    import json

    from cryptotrader.venues.protocol import VenueOperationError

    adapter = PaperVenueAdapter()
    config = connection("paper-checkpoint", "paper")
    session = await adapter.connect(config, None)
    pair = Pair.parse("BTC/USDT")
    await session.set_quote(pair, Decimal("100"))
    await session.place_order(OrderIntent(pair, "buy", Decimal("1"), "market", None, False))
    page = await getattr(session, f"fetch_{kind}")(None)
    assert page.complete is True
    assert page.next_cursor
    assert page.coverage_start == session._account.started_at
    await session.close()
    reopened = await adapter.connect(config, None)
    new_order = await reopened.place_order(OrderIntent(pair, "buy", Decimal("1"), "market", None, False))
    following = await getattr(reopened, f"fetch_{kind}")(page.next_cursor)
    assert following.coverage_start == page.coverage_end
    assert following.coverage_end >= following.coverage_start
    checkpoint = json.loads(following.next_cursor)
    assert checkpoint["connection_id"] == "paper-checkpoint"
    assert checkpoint["kind"] == kind
    assert checkpoint["complete"] is True
    if kind == "fills":
        assert [fill.venue_order_id for fill in following.items] == [new_order.id]
    else:
        assert following.items == ()
    other_kind = "funding" if kind == "fills" else "fills"
    with pytest.raises(VenueOperationError, match="cursor"):
        await getattr(reopened, f"fetch_{other_kind}")(page.next_cursor)
    other = await adapter.connect(connection("other-paper", "paper"), None)
    with pytest.raises(VenueOperationError, match="cursor"):
        await getattr(other, f"fetch_{kind}")(page.next_cursor)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["fills", "funding"])
async def test_paper_checkpoint_continues_seven_day_windows_without_skipping_a_gap(kind):
    from datetime import UTC, datetime, timedelta

    session = await PaperVenueAdapter().connect(connection("paper-gap", "paper"), None)
    session._account.started_at = datetime.now(UTC) - timedelta(days=20)
    pair = Pair.parse("BTC/USDT")
    await session.set_quote(pair, Decimal("100"))
    order = await session.place_order(OrderIntent(pair, "buy", Decimal("1"), "market", None, False))
    first = await getattr(session, f"fetch_{kind}")(None)
    second = await getattr(session, f"fetch_{kind}")(first.next_cursor)
    third = await getattr(session, f"fetch_{kind}")(second.next_cursor)
    assert first.coverage_start == session._account.started_at
    assert first.coverage_end - first.coverage_start == timedelta(days=7)
    assert second.coverage_start == first.coverage_end
    assert third.coverage_start == second.coverage_end
    assert third.coverage_end - third.coverage_start <= timedelta(days=7)
    assert first.items == second.items == ()
    if kind == "fills":
        assert [fill.venue_order_id for fill in third.items] == [order.id]
    else:
        assert third.items == ()


@pytest.mark.asyncio
async def test_paper_fill_at_previous_window_end_is_not_skipped(monkeypatch):
    from datetime import datetime

    from cryptotrader.venues import paper

    session = await PaperVenueAdapter().connect(connection("paper-boundary", "paper"), None)
    pair = Pair.parse("BTC/USDT")
    await session.set_quote(pair, Decimal("100"))
    first = await session.fetch_fills(None)

    class BoundaryClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return first.coverage_end

    monkeypatch.setattr(paper, "datetime", BoundaryClock)
    order = await session.place_order(OrderIntent(pair, "buy", Decimal("1"), "market", None, False))
    next_page = await session.fetch_fills(first.next_cursor)
    assert next_page.coverage_start == first.coverage_end
    assert [fill.venue_order_id for fill in next_page.items] == [order.id]
