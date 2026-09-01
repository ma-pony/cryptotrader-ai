"""Saved component evidence uses actual outputs, never historical inference."""

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pytest

from cryptotrader.signals.presentation import EvaluationReference, SeriesBlock, SeriesPoint, TimelineEntry
from cryptotrader.signals.runner import ComponentRunError, ComponentRunner
from tests.test_runtime_signal_components import (
    _committee_agents,
    _committee_context,
    _kronos,
    _kronos_context,
    _Predictor,
    _Sink,
)


@pytest.mark.parametrize(
    "timestamp",
    [datetime(2026, 8, 31, 9), "2026-08-31T09:00:00"],  # noqa: DTZ001 - intentionally invalid input
)
@pytest.mark.parametrize(
    ("model", "fields", "time_field"),
    [
        (SeriesPoint, {"value": "101"}, "time"),
        (SeriesBlock, {"title": "预测", "series": ()}, "forecast_start"),
        (TimelineEntry, {"actor": "研究员", "body": "保存意见"}, "time"),
        (
            EvaluationReference,
            {
                "reference_price": "101",
                "due_at": "2026-08-31T10:00:00Z",
                "interval": "1h",
                "market_source_id": "default",
            },
            "reference_time",
        ),
        (
            EvaluationReference,
            {
                "reference_price": "101",
                "reference_time": "2026-08-31T08:00:00Z",
                "interval": "1h",
                "market_source_id": "default",
            },
            "due_at",
        ),
    ],
)
def test_saved_result_times_reject_naive_datetime_and_strings(model, fields, time_field, timestamp):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        model.model_validate({**fields, time_field: timestamp})


def timestamped_context():
    context = _kronos_context()
    frame = context.snapshots["4h"].market.ohlcv
    frame["timestamp"] = pd.date_range(end="2025-12-31T20:00:00Z", periods=len(frame), freq="4h")
    return context


class CountingPredictor(_Predictor):
    call_count = 0

    def predict(self, **kwargs):
        self.call_count += 1
        return super().predict(**kwargs)


@pytest.mark.asyncio
async def test_actual_forecast_is_saved_and_journal_reads_do_not_repeat_prediction(tmp_path):
    from cryptotrader.journal.store import MultiVenueCycleStore

    predictor = CountingPredictor()
    signal = await _kronos(predictor=predictor).evaluate(timestamped_context())
    blocks = getattr(signal, "blocks", ())
    assert any(block.kind == "series" and block.forecast_start for block in blocks)
    series = next(block for block in blocks if block.kind == "series")
    assert series.evaluation_target == "candle_close"
    assert len(series.series[0].points) == 20
    assert len(series.series[1].points) == 50
    assert series.series[1].points[0].time == datetime(2026, 1, 1, tzinfo=UTC)
    assert series.series[1].points[0].value == Decimal("110.00000000000001")
    from cryptotrader.journal.models import DecisionRun, MultiVenueCycleRecord

    original = MultiVenueCycleRecord(
        cycle_id="prediction-evidence",
        config_revision=9,
        market_data_source_id="default",
        component_signals=(signal,),
        fused_signal=None,
        target_position=None,
        book_results=(),
        cycle_status="cycle_failed",
        execution_status="not_started",
        requires_attention=False,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        run=DecisionRun("BTC/USDT:USDT", "trading", "manual", {}, datetime(2026, 1, 1, tzinfo=UTC), None),
    )
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'evidence.db'}"
    await migrate_workbench_schema(database_url)
    store = MultiVenueCycleStore(database_url)
    try:
        await store.save(original)
        for _ in range(2):
            restored = await store.get(original.cycle_id)
            assert restored.component_signals[0].blocks == signal.blocks
            assert restored.component_signals[0].evaluation_reference == signal.evaluation_reference
        assert predictor.call_count == 1
    finally:
        from cryptotrader.db import get_engine

        await (await get_engine(store.database_url)).dispose()


@pytest.mark.asyncio
async def test_gate_skip_has_no_forecast_but_weak_filter_keeps_actual_prediction():
    skipped = await _kronos(probability=0.49).evaluate(timestamped_context())
    assert getattr(skipped, "status", None) == "skipped"
    assert not any(block.kind == "series" and block.forecast_start for block in skipped.blocks)
    filtered = await _kronos(predictor=_Predictor(-0.02)).evaluate(timestamped_context())
    assert filtered.status == "completed"
    assert filtered.direction == "neutral"
    assert any(block.kind == "series" and block.forecast_start for block in filtered.blocks)


@pytest.mark.asyncio
async def test_kronos_error_never_exposes_provider_payload():
    with pytest.raises(Exception) as caught:
        await _kronos(predictor=_Predictor(error=RuntimeError("api_key=secret-provider-payload"))).evaluate(
            timestamped_context()
        )
    assert "secret-provider-payload" not in str(caught.value)


@pytest.mark.asyncio
async def test_committee_keeps_initial_opinions_and_complete_debate_timeline():
    from cryptotrader.signals.components.llm_committee import DebateSettings, LLMCommitteeComponent

    async def challenge(agent_id, analysis, others, context, round_number):
        return {**analysis, "reasoning": "修订意见"}, {
            "round": round_number,
            "from": agent_id,
            "to": next(iter(others)),
            "challenge": "完整质疑",
            "response": "完整回应",
            "move": "revise",
        }

    async def summary(_):
        return {"direction": "long", "confidence": 0.7, "reasoning": "最终观点"}

    component = LLMCommitteeComponent(
        None,
        agents=_committee_agents(),
        summary=summary,
        challenger=challenge,
        default_timeframe="1h",
        ohlcv_limit=30,
        models=object(),
        debate=DebateSettings(max_rounds=1, skip_debate=False),
    )
    result = await component.evaluate(_committee_context())
    blocks = getattr(result, "blocks", ())
    assert any(block.kind == "timeline" for block in blocks)
    timeline = next(block for block in blocks if block.kind == "timeline")
    assert len(timeline.entries) == 8
    assert all("analysis" in item.body for item in timeline.entries[:4])
    assert all("完整质疑" in item.body and "完整回应" in item.body for item in timeline.entries[4:])
    assert result.component_id == "llm_committee"


@pytest.mark.asyncio
async def test_runner_retains_completed_and_safe_failed_evidence_without_degrading_to_success():
    good = _kronos()
    bad = _kronos(predictor=_Predictor(error=RuntimeError("credential-private")))
    bad.id = "broken"
    with pytest.raises(ComponentRunError) as caught:
        await ComponentRunner(_Sink()).run([good, bad], timestamped_context())
    signals = getattr(caught.value, "component_signals", ())
    assert len(signals) == 2
    assert [signal.status for signal in signals] == ["completed", "failed"]
    assert all(signal.duration_ms >= 0 for signal in signals)
    assert "credential-private" not in repr(signals)


@pytest.mark.asyncio
async def test_cycle_freezes_reference_at_closed_bar_and_requests_global_timeframe():
    from cryptotrader.decision.models import CycleRequest
    from cryptotrader.signals.models import CandleRequirement, DataRequirements
    from tests.test_multi_book_cycle import PAIR, _cycle, _snapshot
    from tests.test_runtime_signal_components import _snapshot as market_snapshot

    cycle, _, _, _, _ = _cycle(_snapshot())
    cycle.clock = lambda: datetime(2026, 1, 1, 11, 5, tzinfo=UTC)
    captured = []

    class Source:
        id = "default"

        async def collect(self, pair, as_of, requirements):
            captured.append(requirements)
            snapshot = market_snapshot(3)
            snapshot.market.ohlcv["timestamp"] = pd.date_range("2026-01-01T09:00:00Z", periods=3, freq="1h")
            snapshot.market.ohlcv["close"] = [90, 101, 999]
            return replace(timestamped_context(), pair=pair, as_of=as_of, current_price=888, snapshots={"1h": snapshot})

    class Runner:
        async def run(self, components, context):
            captured.append(context)
            return ()

    cycle.market_source, cycle.runner = Source(), Runner()
    cycle.exit_requirement = DataRequirements(candles=(CandleRequirement("4h", 20),))
    cycle.analysis.market_source, cycle.analysis.runner = cycle.market_source, cycle.runner
    cycle.snapshot = replace(
        cycle.snapshot,
        document=cycle.snapshot.document.model_copy(
            update={
                "market_data": cycle.snapshot.document.market_data.model_copy(
                    update={"parameters": {"timeframe": "4h", "limit": 20}}
                ),
            }
        ),
    )
    await cycle.run(CycleRequest(PAIR))
    assert {c.timeframe for c in captured[0].candles} == {"1h", "4h"}
    assert captured[0].candles[0].timeframe == "4h"  # Preserve the existing primary/ATR input.
    reference = captured[1].evaluation_reference
    assert reference.reference_time == datetime(2026, 1, 1, 11, tzinfo=UTC)
    assert reference.reference_price == Decimal("101")
    assert reference.due_at == datetime(2026, 1, 1, 12, tzinfo=UTC)


@pytest.mark.asyncio
async def test_cycle_preserves_failure_snapshot_and_does_not_execute():
    from cryptotrader.decision.models import CycleRequest
    from cryptotrader.signals.models import ComponentSignal
    from tests.test_multi_book_cycle import PAIR, _book, _cycle, _snapshot

    cycle, _, coordinator, journal, _ = _cycle(_snapshot(_book("sim", "simulated", ("a", "b"), hitl=False)))
    failed = ComponentSignal("fixture", "neutral", 0.0, "安全失败原因", status="failed")

    class Runner:
        async def run(self, components, context):
            raise ComponentRunError({"fixture": RuntimeError("secret")}, (failed,))

    cycle.runner = Runner()
    cycle.analysis.runner = cycle.runner
    outcome = await cycle.run(CycleRequest(PAIR))
    assert outcome.status == "component_failed"
    assert journal.records[0].component_signals == (failed,)
    assert journal.records[0].fused_signal is None
    assert coordinator.proposals == []


def test_unverifiable_timestamps_never_use_ticker_or_as_of_and_interval_can_differ():
    from cryptotrader.signals.presentation import freeze_evaluation_reference

    missing = _committee_context()
    missing.snapshots["1h"].market.ohlcv.drop(columns=["timestamp"], inplace=True)
    assert freeze_evaluation_reference(missing, "1h", None) is None
    reference = freeze_evaluation_reference(timestamped_context(), "4h", "2h")
    assert reference.reference_time == datetime(2026, 1, 1, tzinfo=UTC)
    assert reference.due_at == datetime(2026, 1, 1, 2, tzinfo=UTC)


def test_results_reject_html_extra_fields_and_object_table_cells():
    from pydantic import ValidationError

    from cryptotrader.signals.presentation import RESULT_BLOCKS

    for block in (
        {"kind": "text", "title": "正文", "body": "<script>alert(1)</script>"},
        {"kind": "text", "title": "正文", "body": "正常", "html": "escape hatch"},
        {
            "kind": "table",
            "title": "表格",
            "columns": [{"key": "a", "label": "字段"}],
            "rows": [{"cells": [{"column_key": "a", "value": {"nested": "invalid"}}]}],
        },
    ):
        with pytest.raises(ValidationError):
            RESULT_BLOCKS.validate_python([block])


def test_decisions_dto_carries_saved_blocks_without_rebuilding():
    from api.routes.response_dto import cycle_out
    from cryptotrader.signals.presentation import TextBlock
    from tests.test_multi_venue_journal import _record

    record = _record()
    record = replace(
        record,
        component_signals=(
            replace(record.component_signals[0], blocks=(TextBlock(title="证据", body="保存原文"),)),
            record.component_signals[1],
        ),
    )
    result = cycle_out(record).model_dump(mode="json")
    assert result["shared_signals"]["components"][0].get("blocks") == [
        {"kind": "text", "title": "证据", "body": "保存原文"}
    ]


@pytest.mark.asyncio
async def test_runner_preserves_an_explicit_failed_result_and_keeps_all_success_gate():
    from cryptotrader.signals.models import ComponentSignal
    from cryptotrader.signals.presentation import TextBlock
    from tests.test_component_runner import FakeComponent

    failed = ComponentSignal(
        "custom",
        "neutral",
        0.0,
        "行情不完整",
        status="failed",
        blocks=(TextBlock(title="失败说明", body="缺少已收盘K线"),),
    )
    with pytest.raises(ComponentRunError) as caught:
        await ComponentRunner(_Sink()).run([FakeComponent("custom", result=failed)], timestamped_context())
    assert caught.value.component_signals[0].blocks == failed.blocks
    assert caught.value.component_signals[0].reasoning == "行情不完整"


@pytest.mark.asyncio
async def test_component_filter_pages_saved_results_only():
    from cryptotrader.decision.read_service import DecisionReadService
    from cryptotrader.journal.models import DecisionRun, MultiVenueCycleRecord
    from cryptotrader.journal.store import MultiVenueCycleStore
    from cryptotrader.signals.models import ComponentSignal

    store = MultiVenueCycleStore()
    for index, component_id in enumerate(("custom", "other", "custom")):
        await store.save(
            MultiVenueCycleRecord(
                cycle_id=str(index),
                config_revision=1,
                market_data_source_id="default",
                component_signals=(ComponentSignal(component_id, "neutral", 0.0, "样本"),),
                fused_signal=None,
                target_position=None,
                book_results=(),
                cycle_status="cycle_failed",
                execution_status="not_started",
                requires_attention=False,
                created_at=datetime(2026, 1, index + 1, tzinfo=UTC),
                run=DecisionRun(
                    None, "trading", None, {}, None, None, ("pair", "origin", "finished_at", "config_snapshot")
                ),
            )
        )
    page = await DecisionReadService(store).list(limit=1, offset=1, component_id="custom")
    assert page.total == 2
    assert [item.decision_id for item in page.items] == ["0"]
    assert page.has_next is False


@pytest.mark.parametrize(
    ("interval", "milliseconds"), [("15m", 900_000), ("2h", 7_200_000), ("1d", 86_400_000), ("1w", 604_800_000)]
)
def test_global_intervals_use_the_same_close_boundary_in_market_collection(interval, milliseconds):
    from cryptotrader.data.market import _closed_ohlcv
    from cryptotrader.signals.presentation import interval_delta

    row = [0, 1, 2, 1, 2, 3]
    assert interval_delta(interval).total_seconds() * 1000 == milliseconds
    assert _closed_ohlcv([row], interval, milliseconds - 1) == []
    assert _closed_ohlcv([row], interval, milliseconds) == [row]


def test_weekly_candle_closes_from_its_actual_monday_open_not_unix_epoch_week():
    from cryptotrader.data.market import _closed_ohlcv

    opened = int(datetime(2026, 8, 31, tzinfo=UTC).timestamp() * 1000)
    friday = int(datetime(2026, 9, 4, tzinfo=UTC).timestamp() * 1000)
    next_monday = int(datetime(2026, 9, 7, tzinfo=UTC).timestamp() * 1000)
    row = [opened, 100, 110, 90, 105, 20]
    assert _closed_ohlcv([row], "1w", friday) == []
    assert _closed_ohlcv([row], "1w", next_monday) == [row]
