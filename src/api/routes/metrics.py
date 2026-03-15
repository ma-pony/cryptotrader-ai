"""Prometheus metrics endpoints.

GET /metrics  — Prometheus text format (existing, unchanged)
GET /metrics/summary — JSON snapshot of key metrics for Dashboard consumption
"""

from __future__ import annotations

import datetime
import logging

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from pydantic import BaseModel

from cryptotrader.metrics import get_metrics_collector

logger = logging.getLogger(__name__)
router = APIRouter()


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
            snapshot_time=datetime.datetime.now(datetime.UTC),
        )
    except Exception:
        logger.warning("Failed to build metrics summary", exc_info=True)
        raise
