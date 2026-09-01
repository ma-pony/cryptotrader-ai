"""Frozen evidence evaluation: no inference, account reads or current-price substitution."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from importlib import import_module, util
from types import SimpleNamespace

import pytest

from cryptotrader.decision.service import configuration_summary
from cryptotrader.journal.models import DecisionRun, MultiVenueCycleRecord
from cryptotrader.journal.store import MultiVenueCycleStore, _component_signal_payload
from cryptotrader.runtime_config.models import RuntimeConfigSnapshot
from cryptotrader.signals.models import ComponentSignal
from cryptotrader.signals.presentation import EvaluationReference, Series, SeriesBlock, SeriesPoint
from tests.factories.runtime_config import active_document

BASE = datetime(2026, 8, 31, 12, tzinfo=UTC)


def evaluation_api():
    assert util.find_spec("cryptotrader.signals.evaluation") is not None, "frozen evaluation service is missing"
    return import_module("cryptotrader.signals.evaluation")


def record(
    key="one",
    *,
    direction="long",
    status="completed",
    reference=True,
    interval="2h",
    mode="analysis",
    curve=False,
    curve_target="candle_close",
):
    ref = (
        EvaluationReference(
            reference_time=BASE,
            reference_price=Decimal("100"),
            due_at=BASE + timedelta(hours=int(interval[:-1])),
            interval=interval,
            market_source_id="frozen-source",
        )
        if reference
        else None
    )
    blocks = (
        (
            SeriesBlock(
                title="预测",
                forecast_start=BASE,
                evaluation_target=curve_target,
                series=(
                    Series(
                        name="预测收盘价",
                        points=(
                            SeriesPoint(time=BASE, value=Decimal("108")),
                            SeriesPoint(time=BASE + timedelta(hours=4), value=Decimal("118")),
                        ),
                    ),
                ),
            ),
        )
        if curve
        else ()
    )
    signal = ComponentSignal("custom", direction, 0.8, "saved", blocks=blocks, evaluation_reference=ref, status=status)
    frozen = {
        "market_data": {"source_id": "frozen-source", "timeframe": "4h", "parameters": {"market": "original"}},
        "signals": {"components": [{"component_id": "custom", "parameters": {"timeframe": "4h"}}]},
    }
    return MultiVenueCycleRecord(
        key,
        7,
        "frozen-source",
        (signal,),
        None,
        None,
        (),
        "completed" if mode == "analysis" else "component_failed",
        "not_started",
        False,
        BASE,
        DecisionRun("BTC/USDT", mode, "backtest" if mode == "backtest" else "manual", frozen, BASE, None),
    )


class Market:
    id = "frozen-source"

    def __init__(self):
        self.calls = []
        self.rows = {}
        self.fail = False

    async def read_candles(self, pair, timeframe, start, end, as_of):
        self.calls.append((pair.canonical(), timeframe, start, end, as_of))
        if self.fail:
            raise RuntimeError("private-provider-key-not-for-storage")
        return self.rows.get(timeframe, ())


async def setup(tmp_path, items):
    api = evaluation_api()
    store_module = import_module("cryptotrader.signals.evaluation_store")
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    url = f"sqlite+aiosqlite:///{tmp_path / 'evaluation.db'}"
    await migrate_workbench_schema(url)
    journal = MultiVenueCycleStore(url)
    for item in items:
        await journal.save(item)
    market = Market()
    seen = []

    def source_factory(config):
        seen.append(config)
        assert config.parameters["market"] == "original"
        return market

    store = store_module.EvaluationStore(url)
    return SimpleNamespace(
        api=api,
        store=store,
        journal=journal,
        market=market,
        seen=seen,
        service=api.EvaluationService(journal, store, source_factory=source_factory),
    )


def candle(opened, price):
    protocol = import_module("cryptotrader.market_sources.protocol")
    return protocol.HistoricalCandle(
        open_time=opened,
        open=Decimal(price),
        high=Decimal(price),
        low=Decimal(price),
        close=Decimal(price),
        volume=Decimal("1"),
    )


def test_freezes_original_market_parameters_with_defaults():
    snapshot = RuntimeConfigSnapshot(7, active_document(), BASE)
    summary = configuration_summary(snapshot)
    assert summary["market_data"].get("parameters", {}).get("market_adapter_id") == "binance"


@pytest.mark.parametrize(
    ("direction", "actual", "hit"),
    [("long", "110", True), ("short", "110", False), ("long", "100", False), ("short", "90", True)],
)
def test_direction_uses_strict_signed_change(direction, actual, hit):
    api = evaluation_api()
    assert api.evaluate_direction(direction, Decimal("100"), Decimal(actual)) is hit
    assert api.direction_hit_rate(0, 0) is None
    assert api.direction_hit_rate(1, 2) == 0.5
    with pytest.raises(ValueError):
        api.evaluate_direction("neutral", Decimal("100"), Decimal("110"))


@pytest.mark.asyncio
async def test_deadline_missing_retry_and_frozen_journal_survive_rebuild(tmp_path):
    original = record()
    f = await setup(tmp_path, [original])
    before = _component_signal_payload(original.component_signals[0])
    assert await f.service.evaluate_due(BASE + timedelta(hours=1)) == 0
    assert (await f.store.get("one", "custom")).status == "pending"
    assert f.market.calls == []
    due = BASE + timedelta(hours=2)
    f.market.rows["1m"] = (candle(due, "999"),)  # Next bar is NOT the due close.
    assert await f.service.evaluate_due(due) == 0
    assert (await f.store.get("one", "custom")).status == "missing_market"
    f.market.rows["1m"] = (candle(due - timedelta(minutes=1), "110"),)
    assert await f.service.evaluate_due(due) == 1
    assert await f.service.evaluate_due(due + timedelta(hours=1)) == 0
    saved = await f.store.get("one", "custom")
    assert (saved.actual_price, saved.actual_time, saved.hit, saved.return_ratio) == (
        Decimal("110"),
        due,
        True,
        Decimal("0.1"),
    )
    assert f.market.calls[-1][1:4] == ("1m", due - timedelta(minutes=1), due)
    rebuilt = type(f.store)(f.store.database_url)
    assert await rebuilt.get("one", "custom") == saved
    assert _component_signal_payload((await f.journal.get("one")).component_signals[0]) == before
    assert saved.cost is None


@pytest.mark.asyncio
async def test_statuses_denominator_and_period_mode_groups(tmp_path):
    items = [
        record("hit"),
        record("miss", direction="short"),
        record("neutral", direction="neutral"),
        record("skipped", status="skipped"),
        record("failed", status="failed"),
        record("no-reference", reference=False),
        record("later", interval="4h"),
        record("backtest", mode="backtest"),
    ]
    f = await setup(tmp_path, items)
    due = BASE + timedelta(hours=2)
    f.market.rows["1m"] = (candle(due - timedelta(minutes=1), "110"),)
    await f.service.evaluate_due(due)
    summary = await f.store.summary({"component_id": "custom"})
    groups = {(g.mode, g.interval): g for g in summary.groups}
    realtime = groups["analysis", "2h"]
    assert (realtime.total, realtime.matured_directional, realtime.hits, realtime.hit_rate) == (5, 2, 1, 0.5)
    assert (realtime.neutral, realtime.skipped, realtime.failed) == (1, 1, 1)
    assert groups["analysis", "4h"].pending == 1
    assert groups["analysis", None].missing_market == 1
    assert groups["backtest", "2h"].hit_rate == 1
    assert groups["analysis", "4h"].hit_rate is None


@pytest.mark.asyncio
async def test_missing_frozen_source_never_backfilled_and_failures_are_redacted(tmp_path):
    old = record("old")
    old = replace(
        old, run=replace(old.run, config_snapshot={"market_data": {"source_id": "frozen-source", "timeframe": "4h"}})
    )
    f = await setup(tmp_path, [old, record("error")])
    f.market.fail = True
    await f.service.evaluate_due(BASE + timedelta(hours=2))
    assert (await f.store.get("old", "custom")).reason == "frozen_market_configuration_missing"
    failed = await f.store.get("error", "custom")
    assert failed.status == "failed"
    assert "private-provider" not in failed.model_dump_json()
    assert len(f.seen) == 1
    f.market.fail = False
    due = BASE + timedelta(hours=2)
    f.market.rows["1m"] = (candle(due - timedelta(minutes=1), "110"),)
    assert await f.service.evaluate_due(due) == 1
    recovered = await f.store.get("error", "custom")
    assert (recovered.status, recovered.reason, recovered.hit) == ("evaluated", None, True)
    assert len(f.seen) == 2


@pytest.mark.asyncio
async def test_curve_matures_by_bar_close_and_continues_after_direction(tmp_path):
    original = record(curve=True)
    f = await setup(tmp_path, [original])
    before = _component_signal_payload(original.component_signals[0])
    due = BASE + timedelta(hours=2)
    f.market.rows["1m"] = (candle(due - timedelta(minutes=1), "110"),)
    await f.service.evaluate_due(due)
    saved = await f.store.get("one", "custom")
    assert saved.status == "evaluated"
    assert [p.status for p in saved.comparisons[0].points] == ["pending", "pending"]
    f.market.fail = True
    await f.service.evaluate_due(BASE + timedelta(hours=4))
    assert (await f.store.get("one", "custom")).reason == "market_read_failed"
    f.market.fail = False
    f.market.rows["4h"] = (candle(BASE, "110"),)
    await f.service.evaluate_due(BASE + timedelta(hours=4))
    saved = await f.store.get("one", "custom")
    assert [p.actual for p in saved.comparisons[0].points] == [Decimal("110"), None]
    assert saved.reason is None
    assert saved.comparisons[0].mae is None
    f.market.rows["4h"] += (candle(BASE + timedelta(hours=4), "120"),)
    await f.service.evaluate_due(BASE + timedelta(hours=8))
    saved = await f.store.get("one", "custom")
    assert [p.difference for p in saved.comparisons[0].points] == [Decimal("-2"), Decimal("-2")]
    assert saved.comparisons[0].mae == Decimal("2")
    assert saved.comparisons[0].rmse == Decimal("2")
    assert _component_signal_payload((await f.journal.get("one")).component_signals[0]) == before


@pytest.mark.asyncio
async def test_default_historical_source_uses_explicit_window_without_live_collectors(monkeypatch):
    from cryptotrader.data.market import MarketCollector

    calls = []
    opened = BASE - timedelta(minutes=1)

    class Exchange:
        timeframes = {"1m": "1m"}

        async def load_markets(self):
            pass

        async def fetch_ohlcv(self, pair, timeframe, *, since, limit):
            calls.append((pair, timeframe, since, limit))
            return [
                [int(opened.timestamp() * 1000), 100, 110, 90, 110, 1],
                [int(BASE.timestamp() * 1000), 110, 999, 90, 999, 1],
            ]

        async def close(self):
            calls.append("closed")

    monkeypatch.setattr("cryptotrader.data.market.ccxt.binance", lambda options: Exchange())
    bars = await MarketCollector().read_candles("BTC/USDT", "binance", "1m", opened, BASE + timedelta(minutes=1), BASE)
    assert len(bars) == 1
    assert (bars[0].open_time, bars[0].close) == (opened, Decimal("110"))
    assert calls == [("BTC/USDT", "1m", int(opened.timestamp() * 1000), 1), "closed"]


@pytest.mark.asyncio
async def test_evaluation_owner_runs_with_paused_trading_and_can_stop(tmp_path):
    import asyncio

    from fastapi import FastAPI

    from api.main import _clear_runtime_owners
    from cryptotrader.signals.evaluation import EvaluationOwner

    f = await setup(tmp_path, [record()])
    f.market.rows["1m"] = (candle(BASE + timedelta(hours=2) - timedelta(minutes=1), "110"),)
    evaluated = asyncio.Event()

    class ObservedService:
        async def evaluate_due(self, now):
            result = await f.service.evaluate_due(now)
            evaluated.set()
            return result

    owner = EvaluationOwner(ObservedService(), clock=lambda: BASE + timedelta(hours=2), interval_seconds=0.01)
    app = FastAPI()
    app.state.evaluation_owner = owner
    owner.start()
    try:
        await _clear_runtime_owners(app)  # Trading pause/apply must not stop evaluation.
        async with asyncio.timeout(2):
            await evaluated.wait()
        assert (await f.store.get("one", "custom")).status == "evaluated"
    finally:
        await owner.stop()
    assert owner._task is None


@pytest.mark.asyncio
async def test_registered_custom_source_evaluates_without_component_or_account_calls(tmp_path, monkeypatch):
    from cryptotrader.configuration import registry
    from cryptotrader.configuration.catalog import PluginConfiguration
    from cryptotrader.configuration.fields import LocalizedText
    from cryptotrader.configuration.parameters import PluginParameters
    from cryptotrader.configuration.registry import ExtensionRegistration
    from cryptotrader.signals.models import DataRequirements

    f = await setup(tmp_path, [record()])
    due = BASE + timedelta(hours=2)

    class Parameters(PluginParameters):
        market: str

    class RegisteredMarket(Market):
        def requirements(self):
            return DataRequirements()

        async def collect(self, *args):
            raise AssertionError("evaluation must not collect a fresh live snapshot")

    seen = []
    market = RegisteredMarket()
    market.rows["1m"] = (candle(due - timedelta(minutes=1), "110"),)

    def factory(config, **kwargs):
        seen.append(dict(config.parameters))
        return market

    extensions = registry.get_extension_registry()
    label = LocalizedText("历史夹具", "Historical fixture")
    extensions.market_sources["frozen-source"] = ExtensionRegistration(
        PluginConfiguration("frozen-source", label, label, Parameters), factory
    )
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
    assert await f.api.EvaluationService(f.journal, f.store).evaluate_due(due) == 1
    assert seen == [{"market": "original"}]
    assert (await f.store.get("one", "custom")).hit is True


@pytest.mark.asyncio
async def test_frozen_market_configuration_survives_strict_journal_and_http_projection(tmp_path):
    from cryptotrader.decision.read_service import DecisionReadService

    snapshot = RuntimeConfigSnapshot(7, active_document(), BASE)
    original = record()
    original = replace(original, run=replace(original.run, config_snapshot=configuration_summary(snapshot)))
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'roundtrip.db'}"
    await migrate_workbench_schema(database_url)
    journal = MultiVenueCycleStore(database_url)
    await journal.save(original)
    result = await DecisionReadService(MultiVenueCycleStore(journal.database_url)).get("one")
    assert result is not None
    text = result.model_dump_json()
    assert "market_adapter_id" in text
    assert "binance" in text
    assert "credential_ref" not in text


@pytest.mark.asyncio
async def test_unmarked_non_price_forecast_does_not_get_close_price_comparison(tmp_path):
    f = await setup(tmp_path, [record(curve=True, curve_target=None)])
    due = BASE + timedelta(hours=2)
    f.market.rows["1m"] = (candle(due - timedelta(minutes=1), "110"),)
    await f.service.evaluate_due(due)
    result = await f.store.get("one", "custom")
    assert result.status == "evaluated"  # Direction still applies to all components.
    assert result.comparisons == ()  # No declared price semantics: do not invent them.


def _with_frozen_timeframe(item, timeframe):
    frozen = {
        "market_data": dict(item.run.config_snapshot["market_data"]),
        "signals": {"components": [{"component_id": "custom", "parameters": {"timeframe": timeframe}}]},
    }
    return replace(item, run=replace(item.run, config_snapshot=frozen), created_at=BASE + timedelta(minutes=1))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "skipped", "completed"])
async def test_no_curve_unsupported_timeframe_does_not_block_older_healthy_sample(tmp_path, status):
    newer = _with_frozen_timeframe(record("newer", status=status), "1M")
    f = await setup(tmp_path, [newer, record("healthy")])
    due = BASE + timedelta(hours=2)
    f.market.rows["1m"] = (candle(due - timedelta(minutes=1), "110"),)

    assert await f.service.evaluate_due(due) == (2 if status == "completed" else 1)
    saved = await f.store.get("newer", "custom")
    assert saved.status == ("evaluated" if status == "completed" else status)
    assert saved.comparisons == ()
    assert (await f.store.get("healthy", "custom")).hit is True
    summary = (await f.store.summary({"component_id": "custom"})).groups[0]
    assert summary.total == 2
    assert summary.failed == (1 if status == "failed" else 0)
    assert summary.skipped == (1 if status == "skipped" else 0)
    assert summary.matured_directional == (2 if status == "completed" else 1)
    assert await f.service.evaluate_due(due) == 0


@pytest.mark.asyncio
async def test_curve_initialization_failure_is_persisted_without_blocking_peer_or_older_sample(tmp_path):
    broken = _with_frozen_timeframe(record("broken", curve=True), "1M")
    peer = replace(record().component_signals[0], component_id="peer")
    broken = replace(broken, component_signals=(*broken.component_signals, peer))
    before = _component_signal_payload(broken.component_signals[0])
    f = await setup(tmp_path, [broken, record("healthy")])
    due = BASE + timedelta(hours=2)
    f.market.rows["1m"] = (candle(due - timedelta(minutes=1), "110"),)

    assert await f.service.evaluate_due(due) == 2
    bad = await f.store.get("broken", "custom")
    assert (bad.status, bad.reason, bad.comparisons) == ("failed", "curve_initialization_failed", ())
    assert bad.hit is None
    assert "1M" not in bad.model_dump_json()
    assert (await f.store.get("broken", "peer")).hit is True
    assert (await f.store.get("healthy", "custom")).hit is True
    assert len(f.market.calls) == 2
    assert _component_signal_payload((await f.journal.get("broken")).component_signals[0]) == before
    summary = (await f.store.summary({"component_id": "custom"})).groups[0]
    assert (summary.total, summary.failed, summary.matured_directional, summary.hit_rate) == (2, 1, 1, 1)
    rebuilt = f.api.EvaluationService(f.journal, type(f.store)(f.store.database_url), source_factory=lambda _: f.market)
    assert await rebuilt.evaluate_due(due) == 0
    assert await rebuilt.store.get("broken", "custom") == bad
    assert len(f.market.calls) == 2
