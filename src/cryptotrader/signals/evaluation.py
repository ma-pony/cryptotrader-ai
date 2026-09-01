"""Evaluate frozen component facts against their original public market source."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from cryptotrader.market_sources.registry import MarketSourceRegistry
from cryptotrader.pair import Pair
from cryptotrader.runtime_config.models import MarketDataConfig
from cryptotrader.signals.evaluation_store import EvaluationRecord
from cryptotrader.signals.presentation import (
    PredictionComparison,
    PredictionComparisonPoint,
    SeriesBlock,
    interval_delta,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from decimal import Decimal


def evaluate_direction(direction: str, reference_price: Decimal, actual_price: Decimal) -> bool:
    if direction not in {"long", "short"}:
        raise ValueError("direction evaluation accepts long/short only")
    return actual_price > reference_price if direction == "long" else actual_price < reference_price


def direction_hit_rate(hits: int, matured_directional: int) -> float | None:
    return hits / matured_directional if matured_directional else None


def _source(config):
    return MarketSourceRegistry.discover(config).require(config.source_id)


def _comparisons(record, signal):
    forecasts = [
        (block, series, future)
        for block in signal.blocks
        if isinstance(block, SeriesBlock)
        and block.forecast_start is not None
        and block.evaluation_target == "candle_close"
        for series in block.series
        if (future := tuple(p for p in series.points if p.time >= block.forecast_start))
    ]
    if not forecasts:
        return ()
    components = record.run.config_snapshot.get("signals", {}).get("components", ())
    component = next((item for item in components if item["component_id"] == signal.component_id), {})
    timeframe = component.get("parameters", {}).get("timeframe")
    if not timeframe:
        return ()
    delta = interval_delta(timeframe)
    result = []
    for block, series, future in forecasts:
        points = tuple(
            PredictionComparisonPoint(time=p.time, close_time=p.time + delta, predicted=p.value) for p in future
        )
        result.append(
            PredictionComparison(
                title=block.title,
                name=series.name,
                timeframe=timeframe,
                points=points,
                matched=0,
                total=len(points),
            )
        )
    return tuple(result)


def _initial(record, signal):
    reference = signal.evaluation_reference
    status, reason = "pending", None
    if signal.status in {"failed", "skipped"}:
        status = signal.status
    elif reference is None:
        status, reason = "missing_market", "reference_missing"
    elif signal.direction == "neutral":
        status = "not_directional"
    comparisons = ()
    if signal.status == "completed" and reference is not None:
        try:
            comparisons = _comparisons(record, signal)
        except Exception:
            # Frozen curve metadata cannot improve on a later market refresh.
            # Keep the failure in this component and never retain exception text.
            status, reason = "failed", "curve_initialization_failed"
    return EvaluationRecord(
        decision_id=record.cycle_id,
        component_id=signal.component_id,
        pair=record.run.pair,
        mode=record.run.mode,
        config_revision=record.config_revision,
        interval=reference.interval if reference else None,
        created_at=record.created_at,
        status=status,
        direction=signal.direction,
        reference=reference,
        reason=reason,
        comparisons=comparisons,
        cost=signal.cost,
    )


class EvaluationService:
    def __init__(self, journal, store, *, source_factory=_source):
        self.journal, self.store, self.source_factory = journal, store, source_factory
        self._lock = asyncio.Lock()

    async def evaluate_due(self, now: datetime) -> int:
        """Return newly completed directional evaluations, not curve refresh count."""
        if now.utcoffset() is None:
            raise ValueError("evaluation clock must be timezone-aware")
        completed = 0
        async with self._lock:
            offset = 0
            while True:
                records = await self.journal.list(limit=100, offset=offset)
                for record in records:
                    if record.cycle_status in {"queued", "running"}:
                        continue
                    for signal in record.component_signals:
                        previous = await self.store.get(record.cycle_id, signal.component_id)
                        current = previous or _initial(record, signal)
                        result = await self._evaluate(record, signal, current, now)
                        if result != previous:
                            await self.store.upsert(result)
                        completed += result.status == "evaluated" and (
                            previous is None or previous.status != "evaluated"
                        )
                if len(records) < 100:
                    break
                offset += len(records)
        return completed

    async def _evaluate(self, record, signal, current, now):
        ref = current.reference
        if signal.status != "completed" or ref is None or current.reason == "curve_initialization_failed":
            return current
        needs_direction = current.status != "evaluated" and signal.direction != "neutral" and ref.due_at <= now
        needs_curve = any(p.status != "matched" and p.close_time <= now for c in current.comparisons for p in c.points)
        if not needs_direction and not needs_curve:
            return current
        frozen = record.run.config_snapshot.get("market_data", {})
        if not current.pair or "parameters" not in frozen or frozen.get("source_id") != ref.market_source_id:
            return current.model_copy(
                update={
                    "status": "missing_market" if needs_direction else current.status,
                    "reason": "frozen_market_configuration_missing",
                }
            )
        try:
            config = MarketDataConfig.model_validate(dict(frozen))
            source = self.source_factory(config)
            if source.id != ref.market_source_id:
                raise ValueError("source identity mismatch")
            if needs_direction:
                delta = timedelta(minutes=1)
                opened = ref.due_at - delta
                bars = await source.read_candles(Pair.parse(current.pair), "1m", opened, ref.due_at, now)
                bar = next((b for b in bars if b.open_time == opened and b.open_time + delta <= now), None)
                if bar is None:
                    current = current.model_copy(update={"status": "missing_market", "reason": "due_candle_missing"})
                else:
                    current = current.model_copy(
                        update={
                            "status": "evaluated",
                            "actual_price": bar.close,
                            "actual_time": ref.due_at,
                            "hit": evaluate_direction(current.direction, ref.reference_price, bar.close),
                            "return_ratio": (bar.close - ref.reference_price) / ref.reference_price,
                            "reason": None,
                        }
                    )
            if needs_curve:
                current = current.model_copy(
                    update={
                        "comparisons": await self._update_curves(current, source, now),
                        "reason": None
                        if current.reason == "market_read_failed" and current.status in {"evaluated", "not_directional"}
                        else current.reason,
                    }
                )
        except Exception:
            # External messages may contain credentials/URLs; never persist or echo them.
            current = current.model_copy(
                update={
                    "status": "failed" if needs_direction and current.status != "evaluated" else current.status,
                    "reason": "market_read_failed",
                }
            )
        return current

    async def _update_curves(self, record, source, now):
        result = []
        for comparison in record.comparisons:
            due = [p for p in comparison.points if p.status != "matched" and p.close_time <= now]
            if not due:
                result.append(comparison)
                continue
            delta = interval_delta(comparison.timeframe)
            bars = await source.read_candles(
                Pair.parse(record.pair),
                comparison.timeframe,
                min(p.time for p in due),
                max(p.time for p in due) + delta,
                now,
            )
            prices = {bar.open_time: bar.close for bar in bars if bar.open_time + delta <= now}
            points = []
            for point in comparison.points:
                if point.status == "matched" or point.close_time > now:
                    points.append(point)
                    continue
                actual = prices.get(point.time)
                matched = actual is not None and point.predicted is not None
                points.append(
                    point.model_copy(
                        update={
                            "actual": actual,
                            "difference": point.predicted - actual if matched else None,
                            "status": "matched" if matched else "missing_market",
                        }
                    )
                )
            errors = [p.difference for p in points if p.status == "matched"]
            complete = len(errors) == len(points)
            result.append(
                comparison.model_copy(
                    update={
                        "points": tuple(points),
                        "matched": len(errors),
                        "mae": sum(abs(e) for e in errors) / len(errors) if complete else None,
                        "rmse": (sum(e * e for e in errors) / len(errors)).sqrt() if complete else None,
                    }
                )
            )
        return tuple(result)


class EvaluationOwner:
    """Independent of trading automation, scheduler pause and account lifecycle."""

    def __init__(self, service, *, clock=None, interval_seconds=60):
        self.service = service
        self.clock = clock or (lambda: datetime.now(UTC))
        self.interval_seconds = interval_seconds
        self._task = None
        self._stop = asyncio.Event()

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="component-evaluation")

    async def stop(self):
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self):
        while not self._stop.is_set():
            try:
                await self.service.evaluate_due(self.clock())
            except Exception:
                logger.warning("Component evaluation refresh failed")
            with suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
