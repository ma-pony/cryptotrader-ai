"""Closed, immutable presentation contracts; all strings are plain text."""
# ruff: noqa: RUF001

from __future__ import annotations

import re
from datetime import timedelta
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    model_validator,
)


def plain_text(value: str) -> str:
    if re.search(r"<\s*/?\s*[a-zA-Z][^>]*>|javascript\s*:", value, re.IGNORECASE):
        raise ValueError("presentation accepts plain text only")
    return value


Text = Annotated[StrictStr, AfterValidator(plain_text)]
Scalar = Text | StrictBool | StrictInt | StrictFloat | None


class PresentationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TextBlock(PresentationModel):
    kind: Literal["text"] = "text"
    title: Text
    body: Text


class Metric(PresentationModel):
    key: Text
    value: Scalar
    unit: Text | None = None
    note: Text | None = None


class MetricsBlock(PresentationModel):
    kind: Literal["metrics"] = "metrics"
    title: Text
    metrics: tuple[Metric, ...]


class SeriesPoint(PresentationModel):
    time: AwareDatetime
    value: Decimal | None


class Series(PresentationModel):
    name: Text
    unit: Text | None = None
    points: tuple[SeriesPoint, ...]


class SeriesBlock(PresentationModel):
    kind: Literal["series"] = "series"
    title: Text
    series: tuple[Series, ...]
    forecast_start: AwareDatetime | None = None
    evaluation_target: Literal["candle_close"] | None = None


class TableColumn(PresentationModel):
    key: Text
    label: Text


class TableCell(PresentationModel):
    column_key: Text
    value: Scalar


class TableRow(PresentationModel):
    cells: tuple[TableCell, ...]


class TableBlock(PresentationModel):
    kind: Literal["table"] = "table"
    title: Text
    columns: tuple[TableColumn, ...]
    rows: tuple[TableRow, ...]

    @model_validator(mode="after")
    def valid_cells(self):
        keys = [column.key for column in self.columns]
        if len(keys) != len(set(keys)) or any(
            len(row.cells) != len(keys) or {cell.column_key for cell in row.cells} != set(keys) for row in self.rows
        ):
            raise ValueError("table cells must match unique columns")
        return self


class TimelineEntry(PresentationModel):
    time: AwareDatetime
    actor: Text
    body: Text


class TimelineBlock(PresentationModel):
    kind: Literal["timeline"] = "timeline"
    title: Text
    entries: tuple[TimelineEntry, ...]


ResultBlock = Annotated[
    TextBlock | MetricsBlock | SeriesBlock | TableBlock | TimelineBlock, Field(discriminator="kind")
]
RESULT_BLOCKS = TypeAdapter(tuple[ResultBlock, ...])


class EvaluationReference(PresentationModel):
    reference_time: AwareDatetime
    reference_price: Decimal = Field(gt=0)
    due_at: AwareDatetime
    interval: Text
    market_source_id: Text

    @model_validator(mode="after")
    def valid_times(self):
        if self.due_at <= self.reference_time:
            raise ValueError("evaluation reference requires aware, increasing timestamps")
        return self


class SignalUsage(PresentationModel):
    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)


class PredictionComparisonPoint(PresentationModel):
    time: AwareDatetime
    close_time: AwareDatetime
    predicted: Decimal | None
    actual: Decimal | None = None
    difference: Decimal | None = None
    status: Literal["pending", "matched", "missing_market"] = "pending"


class PredictionComparison(PresentationModel):
    title: Text
    name: Text
    timeframe: Text
    points: tuple[PredictionComparisonPoint, ...]
    matched: int = Field(ge=0)
    total: int = Field(ge=0)
    mae: Decimal | None = None
    rmse: Decimal | None = None


def interval_delta(value: str) -> timedelta:
    match = re.fullmatch(r"([1-9][0-9]*)([mhdw])", value)
    if match is None:
        raise ValueError("周期必须是正整数加 m/h/d/w，例如 1h")
    return timedelta(seconds=int(match[1]) * {"m": 60, "h": 3600, "d": 86400, "w": 604800}[match[2]])


def freeze_evaluation_reference(context, timeframe: str, evaluation_interval: str | None) -> EvaluationReference | None:
    """Use a verifiable candle close; absent timestamps mean not evaluable."""
    import pandas as pd

    interval = evaluation_interval or timeframe
    delta, evaluation_delta = interval_delta(timeframe), interval_delta(interval)
    snapshot = context.snapshots.get(timeframe)
    if snapshot is None:
        return None
    frame = snapshot.market.ohlcv
    if "close" not in frame or frame.empty:
        return None
    if "timestamp" in frame:
        raw = frame["timestamp"]
        opens = pd.to_datetime(
            raw, unit="ms" if pd.api.types.is_numeric_dtype(raw) else None, utc=True, errors="coerce"
        )
    elif isinstance(frame.index, pd.DatetimeIndex):
        opens = pd.to_datetime(frame.index, utc=True, errors="coerce")
    else:
        return None
    candidates = []
    for opened, price in zip(opens, frame["close"], strict=True):
        if pd.isna(opened) or pd.isna(price):
            continue
        closed = opened.to_pydatetime() + delta
        decimal_price = Decimal(str(price))
        if closed <= context.as_of and decimal_price.is_finite() and decimal_price > 0:
            candidates.append((closed, decimal_price))
    if not candidates:
        return None
    reference_time, reference_price = max(candidates, key=lambda item: item[0])
    return EvaluationReference(
        reference_time=reference_time,
        reference_price=reference_price,
        due_at=reference_time + evaluation_delta,
        interval=interval,
        market_source_id=context.market_data_source_id,
    )
