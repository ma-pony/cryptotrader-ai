"""Prometheus metrics endpoints.

GET /metrics  — Prometheus text format (existing, unchanged)
GET /metrics/summary — JSON snapshot of key metrics for Dashboard consumption
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from pydantic import BaseModel

from cryptotrader._compat import UTC
from cryptotrader.metrics import get_metrics_collector

logger = logging.getLogger(__name__)
router = APIRouter()
api_router = APIRouter(prefix="/api/metrics", tags=["metrics"])


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class MetricsSummaryResponse(BaseModel):
    """JSON snapshot of key MetricsCollector counters and histogram percentiles."""

    llm_calls_total: int
    debate_skipped_total: int
    verdict_distribution: dict[str, int]
    risk_rejected_total: int
    risk_rejected_by_check: dict[str, int]
    trade_executed_total: int
    pipeline_duration_p50_ms: float
    pipeline_duration_p95_ms: float
    execution_latency_p50_ms: float
    execution_latency_p95_ms: float
    snapshot_time: datetime.datetime


# ---------------------------------------------------------------------------
# Internal helpers: read prometheus-client registry
# ---------------------------------------------------------------------------


def _sum_counter_samples(metric_name: str) -> int:
    """Sum all *_total samples for a Counter with the given base metric name.

    prometheus-client stores Counter samples with the name ``{base}_total``.
    ``metric_name`` should be the full name including ``_total`` suffix.
    Returns 0 when no samples exist yet.
    """
    base_name = metric_name.removesuffix("_total")
    total = 0.0
    for m in REGISTRY.collect():
        if m.name == base_name:
            for sample in m.samples:
                if sample.name == f"{base_name}_total":
                    total += sample.value
    return int(total)


def _collect_labeled_counter(metric_name: str) -> dict[str, int]:
    """Collect per-label counts for a Counter that has exactly one label dimension.

    Returns a dict of {label_value: count}. The label key is the first (and only)
    non-empty label in each sample. For counters with multiple labels (e.g.
    ct_llm_calls_total with model+node), values are summed across all label
    combinations — callers must ensure this interpretation is appropriate.

    ``metric_name`` should be the full name including ``_total`` suffix.
    """
    base_name = metric_name.removesuffix("_total")
    result: dict[str, float] = {}
    for m in REGISTRY.collect():
        if m.name == base_name:
            for sample in m.samples:
                if sample.name != f"{base_name}_total":
                    continue
                labels = sample.labels
                if not labels:
                    continue
                # Use the first label value as the grouping key
                key = next(iter(labels.values()))
                result[key] = result.get(key, 0.0) + sample.value
    return {k: int(v) for k, v in result.items()}


def _collect_histogram_buckets(
    metric_name: str,
    filter_labels: dict[str, str] | None,
) -> tuple[dict[float, float], float]:
    """Collect bucket cumulative counts and total count for a Histogram.

    Returns a tuple of (buckets, total_count) where:
    - buckets: {upper_bound: cumulative_count} (finite upper bounds only)
    - total_count: value of the +Inf bucket (total observations)
    """
    buckets: dict[float, float] = {}
    total_count = 0.0
    for m in REGISTRY.collect():
        if m.name != metric_name:
            continue
        for sample in m.samples:
            if sample.name != f"{metric_name}_bucket":
                continue
            if filter_labels and not all(sample.labels.get(k) == v for k, v in filter_labels.items()):
                continue
            le_str = sample.labels.get("le", "")
            if not le_str:
                continue
            if le_str == "+Inf":
                total_count += sample.value
            else:
                upper = float(le_str)
                buckets[upper] = buckets.get(upper, 0.0) + sample.value
    return buckets, total_count


def _interpolate_quantile(buckets: dict[float, float], total_count: float, quantile: float) -> float:
    """Estimate a quantile via linear interpolation within the covering bucket.

    Implements the same algorithm as Prometheus ``histogram_quantile()``.
    Returns 0.0 if total_count is 0 or buckets is empty.
    """
    if total_count == 0.0 or not buckets:
        return 0.0
    target = quantile * total_count
    sorted_bounds = sorted(buckets.keys())
    prev_bound = 0.0
    prev_count = 0.0
    for upper in sorted_bounds:
        cum_count = buckets[upper]
        if cum_count >= target:
            count_in_bucket = cum_count - prev_count
            if count_in_bucket == 0:
                return upper
            fraction = (target - prev_count) / count_in_bucket
            return prev_bound + fraction * (upper - prev_bound)
        prev_bound = upper
        prev_count = cum_count
    # All observations are in the +Inf bucket — return the last finite bound
    return sorted_bounds[-1]


def _histogram_quantile(metric_name: str, quantile: float, *, filter_labels: dict[str, str] | None = None) -> float:
    """Estimate a quantile from a prometheus Histogram using bucket linear interpolation.

    This implements the same algorithm as Prometheus ``histogram_quantile()``:
    find the smallest bucket whose cumulative count covers the target quantile,
    then linearly interpolate within that bucket.

    Args:
        metric_name: Base metric name (without ``_bucket``/``_count`` suffixes).
        quantile: Target quantile in [0, 1], e.g. 0.5 for p50.
        filter_labels: When the histogram has label dimensions (e.g. ``engine``),
            only include samples whose labels are a superset of this dict.
            Pass ``None`` to aggregate all label combinations.

    Returns:
        Estimated quantile value (float). Returns 0.0 when no observations exist.
    """
    buckets, total_count = _collect_histogram_buckets(metric_name, filter_labels)
    return _interpolate_quantile(buckets, total_count, quantile)


# ---------------------------------------------------------------------------
# Existing endpoint (unchanged)
# ---------------------------------------------------------------------------


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Return Prometheus text format metrics."""
    try:
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
    except Exception:
        logger.warning("Failed to generate Prometheus metrics", exc_info=True)
        return Response(content=b"", media_type=CONTENT_TYPE_LATEST, status_code=500)


# ---------------------------------------------------------------------------
# New endpoint: /metrics/summary
# ---------------------------------------------------------------------------


@router.get("/metrics/summary", response_model=MetricsSummaryResponse)
async def metrics_summary() -> MetricsSummaryResponse:
    """Return a JSON snapshot of key metrics for Dashboard consumption.

    Reads current values from the process-local prometheus-client registry.
    Counter values are monotonically increasing since process start.
    Histogram percentiles use linear bucket interpolation (same as Prometheus
    ``histogram_quantile()``).
    """
    try:
        # Ensure the singleton is initialized so all metrics are registered
        get_metrics_collector()

        llm_calls_total = _sum_counter_samples("ct_llm_calls_total")
        debate_skipped_total = _sum_counter_samples("ct_debate_skipped_total")
        risk_rejected_total = _sum_counter_samples("ct_risk_rejected_total")
        trade_executed_total = _sum_counter_samples("ct_trade_executed_total")

        verdict_distribution = _collect_labeled_counter("ct_verdict_total")
        risk_rejected_by_check = _collect_labeled_counter("ct_risk_rejected_total")

        pipeline_p50 = _histogram_quantile("ct_pipeline_duration_ms", 0.50)
        pipeline_p95 = _histogram_quantile("ct_pipeline_duration_ms", 0.95)
        execution_p50 = _histogram_quantile("ct_execution_latency_ms", 0.50)
        execution_p95 = _histogram_quantile("ct_execution_latency_ms", 0.95)

        return MetricsSummaryResponse(
            llm_calls_total=llm_calls_total,
            debate_skipped_total=debate_skipped_total,
            verdict_distribution=verdict_distribution,
            risk_rejected_total=risk_rejected_total,
            risk_rejected_by_check=risk_rejected_by_check,
            trade_executed_total=trade_executed_total,
            pipeline_duration_p50_ms=float(pipeline_p50),
            pipeline_duration_p95_ms=float(pipeline_p95),
            execution_latency_p50_ms=float(execution_p50),
            execution_latency_p95_ms=float(execution_p95),
            snapshot_time=datetime.datetime.now(UTC),
        )
    except Exception:
        logger.warning("Failed to build metrics summary", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Contract endpoint — /api/metrics/summary (FR-808)
# ---------------------------------------------------------------------------


class MetricsCounters(BaseModel):
    trades_total: int
    orders_placed: int
    orders_failed: int
    risk_rejections: int
    debate_skipped_total: int


class MetricsPercentiles(BaseModel):
    pipeline_p50_ms: float
    pipeline_p95_ms: float
    execution_p50_ms: float
    execution_p95_ms: float


class LatencyHistogramBucketOut(BaseModel):
    """One Prometheus histogram bucket for the Metrics page chart."""

    upper_bound_s: float  # upper bound in seconds (+Inf serialised as a large number)
    count: float


class DailyCostPointOut(BaseModel):
    ts: str  # ISO date, e.g. "2026-04-17"
    cost_usd: float


class MetricsSummaryV2Response(BaseModel):
    """Data-model §5 MetricsSummary shape for the React Metrics page."""

    counters: MetricsCounters
    percentiles: MetricsPercentiles
    collected_at: datetime.datetime
    # Alignment with frontend prototype (2026-04-24):
    llm_calls_24h: int = 0
    llm_cost_24h: float = 0.0
    cache_hit_rate: float = 0.0
    decisions_per_day: float = 0.0
    latency_histogram: list[LatencyHistogramBucketOut] = []
    cost_14d: list[DailyCostPointOut] = []


def _pipeline_histogram_buckets() -> list[LatencyHistogramBucketOut]:
    """Export raw Prometheus histogram buckets for the pipeline duration metric."""
    buckets, total = _collect_histogram_buckets("ct_pipeline_duration_ms", filter_labels=None)
    out: list[LatencyHistogramBucketOut] = []
    for upper_ms, count in sorted(buckets.items()):
        out.append(LatencyHistogramBucketOut(upper_bound_s=upper_ms / 1000.0, count=count))
    if total:
        # Append the +Inf bucket as a synthetic bound so front-end can render the long tail.
        out.append(LatencyHistogramBucketOut(upper_bound_s=1e12, count=total))
    return out


async def _llm_accounting_last_24h(database_url: str | None) -> tuple[int, float, float, float]:
    """Aggregate llm_calls / llm_cost / cache_hit_rate / decisions_per_day from journal.

    Returns ``(calls_24h, cost_24h, cache_hit_rate, decisions_per_day_last_30d)``.
    """
    from cryptotrader.journal.store import JournalStore

    try:
        store = JournalStore(database_url)
        commits = await store.log(limit=2000)
    except Exception:
        logger.debug("metrics: journal read failed", exc_info=True)
        return 0, 0.0, 0.0, 0.0

    now = datetime.datetime.now(UTC)
    cutoff_24h = now - datetime.timedelta(hours=24)
    cutoff_30d = now - datetime.timedelta(days=30)

    def _ts(c: Any) -> datetime.datetime | None:
        return _metrics_coerce_ts(c.timestamp)

    calls = 0
    cost = 0.0
    cache_hits = 0
    in_last_24h = 0
    decisions_30d = 0

    for c in commits:
        ts = _ts(c)
        if ts is None:
            continue
        if ts >= cutoff_30d:
            decisions_30d += 1
        if ts < cutoff_24h:
            continue
        in_last_24h += 1
        usage = getattr(c, "token_usage", None) or {}
        if isinstance(usage, dict):
            calls += int(usage.get("calls", 0) or 0)
            cost += float(usage.get("cost_usd", 0.0) or 0.0)
            cache_hits += int(usage.get("cache_hits", 0) or 0)

    # Anthropic prompt cache can emit multi-segment cache_reads per call, so
    # raw cache_hits may exceed calls — clamp to [0, 1] for the contract.
    cache_hit_rate = min(1.0, (cache_hits / calls) if calls > 0 else 0.0)
    decisions_per_day = decisions_30d / 30.0
    # Suppress in_last_24h — not returned (kept as a trace aid).
    _ = in_last_24h
    return calls, round(cost, 4), round(cache_hit_rate, 4), round(decisions_per_day, 2)


async def _cost_14d_series(database_url: str | None) -> list[DailyCostPointOut]:
    """Per-day cost total for the last 14 calendar days (UTC), including zero-fill days."""
    from cryptotrader.journal.store import JournalStore

    try:
        store = JournalStore(database_url)
        commits = await store.log(limit=3000)
    except Exception:
        logger.debug("cost_14d: journal read failed", exc_info=True)
        return []

    now = datetime.datetime.now(UTC)
    daily: dict[str, float] = {}
    for i in range(14):
        day = (now - datetime.timedelta(days=13 - i)).date().isoformat()
        daily[day] = 0.0

    cutoff = now - datetime.timedelta(days=14)
    for c in commits:
        ts_dt = _metrics_coerce_ts(c.timestamp)
        if ts_dt is None or ts_dt < cutoff:
            continue
        day = ts_dt.date().isoformat()
        if day not in daily:
            continue
        usage = getattr(c, "token_usage", None) or {}
        if isinstance(usage, dict):
            daily[day] += float(usage.get("cost_usd", 0.0) or 0.0)

    return [DailyCostPointOut(ts=d, cost_usd=round(v, 4)) for d, v in daily.items()]


from api.routes._utils import coerce_timestamp as _metrics_coerce_ts  # noqa: E402  — shared util


@api_router.get("/summary", response_model=MetricsSummaryV2Response)
async def metrics_summary_v2() -> MetricsSummaryV2Response:
    """Return key metrics in the data-model contract shape (FR-808).

    ``orders_placed`` mirrors ``trades_total`` (every executed trade went
    through OrderManager). ``orders_failed`` is reserved for a future failure
    counter and currently reports 0 — exposing the field keeps the contract
    stable for the frontend.
    """
    get_metrics_collector()

    trades_total = _sum_counter_samples("ct_trade_executed_total")
    risk_rejections = _sum_counter_samples("ct_risk_rejected_total")
    debate_skipped = _sum_counter_samples("ct_debate_skipped_total")

    pipeline_p50 = _histogram_quantile("ct_pipeline_duration_ms", 0.50)
    pipeline_p95 = _histogram_quantile("ct_pipeline_duration_ms", 0.95)
    execution_p50 = _histogram_quantile("ct_execution_latency_ms", 0.50)
    execution_p95 = _histogram_quantile("ct_execution_latency_ms", 0.95)

    from cryptotrader.config import load_config

    cfg = load_config()
    db_url = cfg.infrastructure.database_url
    # Parallel fetch: both helpers scan the journal independently — gather saves ~50% latency.
    (calls_24h, cost_24h, cache_hit_rate, decisions_per_day), cost_14d = await asyncio.gather(
        _llm_accounting_last_24h(db_url),
        _cost_14d_series(db_url),
    )
    latency_hist = _pipeline_histogram_buckets()

    return MetricsSummaryV2Response(
        counters=MetricsCounters(
            trades_total=trades_total,
            orders_placed=trades_total,
            orders_failed=0,
            risk_rejections=risk_rejections,
            debate_skipped_total=debate_skipped,
        ),
        percentiles=MetricsPercentiles(
            pipeline_p50_ms=float(pipeline_p50),
            pipeline_p95_ms=float(pipeline_p95),
            execution_p50_ms=float(execution_p50),
            execution_p95_ms=float(execution_p95),
        ),
        collected_at=datetime.datetime.now(UTC),
        llm_calls_24h=calls_24h,
        llm_cost_24h=cost_24h,
        cache_hit_rate=cache_hit_rate,
        decisions_per_day=decisions_per_day,
        latency_histogram=latency_hist,
        cost_14d=cost_14d,
    )
