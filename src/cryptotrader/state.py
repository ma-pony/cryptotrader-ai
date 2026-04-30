"""Arena state definition — shared by graph.py and all node modules."""

from __future__ import annotations

import logging
import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage

from cryptotrader.pair import Pair

if TYPE_CHECKING:
    from cryptotrader.config import AppConfig
    from cryptotrader.models import DataSnapshot

logger = logging.getLogger(__name__)

# Spec 013 (Phase 3c) — bumped from implicit v1 (str pair) to v2 (Pair pair).
# Used by checkpoint deserialization to detect legacy state and apply the
# str→Pair compat shim per FR-204.
STATE_SCHEMA_VERSION = 2

# One-time-per-pair WARN cache for the legacy-str compat shim (FR-204).
_legacy_str_warned: set[str] = set()


def get_pair(state: ArenaState) -> Pair:
    """Read ``state.metadata.pair`` as a ``Pair`` instance.

    Backward-compatibility shim per FR-204: if a legacy state checkpoint or
    older caller stored ``pair`` as a str (state schema v1), parse it on
    read and emit one WARN per distinct pair. Saves us a global migration
    of every old checkpoint while making the new contract authoritative.
    """
    raw = state["metadata"]["pair"]
    if isinstance(raw, Pair):
        return raw
    if not isinstance(raw, str):
        raise TypeError(f"state.metadata.pair must be Pair or str, got {type(raw).__name__}")
    if raw not in _legacy_str_warned:
        logger.warning(
            "state.metadata.pair stored as str (%r) — legacy v1 schema; coercing to Pair. "
            "Update producer to pass Pair (spec 013 FR-204).",
            raw,
        )
        _legacy_str_warned.add(raw)
    return Pair.parse(raw)


def merge_dicts(a: dict, b: dict) -> dict:
    result = {**a}
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


class HitlState(TypedDict, total=False):
    approval_id: str
    decision: str
    trigger_reason: str
    skipped: bool


class ArenaState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, Any], merge_dicts]
    metadata: Annotated[dict[str, Any], merge_dicts]
    debate_round: int
    max_debate_rounds: int
    divergence_scores: Annotated[list[float], operator.add]
    hitl: Annotated[dict[str, Any], merge_dicts]


def build_initial_state(
    pair: str | Pair,
    *,
    engine: str = "paper",
    exchange_id: str = "",
    timeframe: str | None = None,
    ohlcv_limit: int | None = None,
    snapshot: DataSnapshot | None = None,
    config: AppConfig | None = None,
    extra_metadata: dict | None = None,
    extra_data: dict | None = None,
) -> dict:
    """Build the initial state dict for graph invocation.

    Args:
        pair: Trading pair as :class:`~cryptotrader.pair.Pair` (preferred) or
            a canonical str (auto-coerced via ``Pair.parse``).
        engine: Execution engine — ``"paper"`` or ``"live"``.
        exchange_id: Exchange identifier, e.g. ``"binance"``.
        timeframe: OHLCV timeframe; falls back to ``config.data.default_timeframe``.
        ohlcv_limit: Number of OHLCV bars; falls back to ``config.data.ohlcv_limit``.
        snapshot: Pre-built :class:`~cryptotrader.models.DataSnapshot` (used in
            backtests where data are already collected before graph invocation).
        config: Loaded :class:`~cryptotrader.config.AppConfig`; resolved via
            :func:`~cryptotrader.config.load_config` when *None*.
        extra_metadata: Additional key/value pairs merged (shallowly) into the
            ``metadata`` dict, allowing callers to override or extend defaults.
        extra_data: Additional key/value pairs merged into the ``data`` dict,
            e.g. ``position_context`` for backtests that track position externally.

    Returns:
        A fully-formed initial state dict ready to pass to ``graph.ainvoke()``.
    """
    if config is None:
        from cryptotrader.config import load_config

        config = load_config()

    pair_obj = pair if isinstance(pair, Pair) else Pair.parse(pair)

    metadata: dict[str, Any] = {
        "pair": pair_obj,
        "_state_schema_version": STATE_SCHEMA_VERSION,
        "engine": engine,
        "exchange_id": exchange_id or config.exchange_id,
        "timeframe": timeframe if timeframe is not None else config.data.default_timeframe,
        "ohlcv_limit": ohlcv_limit if ohlcv_limit is not None else config.data.ohlcv_limit,
        "analysis_model": config.models.analysis,
        "debate_model": config.models.debate,
        "verdict_model": config.models.verdict,
        "models": {
            "tech_agent": config.models.tech_agent,
            "chain_agent": config.models.chain_agent,
            "news_agent": config.models.news_agent,
            "macro_agent": config.models.macro_agent,
        },
        "database_url": config.infrastructure.database_url,
        "redis_url": config.infrastructure.redis_url,
        "convergence_threshold": config.debate.convergence_threshold,
        "max_single_pct": config.risk.position.max_single_pct,
        "schedule_depth": 0,
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    initial_data: dict[str, Any] = {}
    if snapshot is not None:
        initial_data["snapshot"] = snapshot
    if extra_data:
        initial_data.update(extra_data)

    return {
        "messages": [],
        "data": initial_data,
        "metadata": metadata,
        "debate_round": 0,
        "max_debate_rounds": config.debate.max_rounds,
        "divergence_scores": [],
        "hitl": {},
    }
