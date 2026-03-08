"""Arena state definition — shared by graph.py and all node modules."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage

if TYPE_CHECKING:
    from cryptotrader.config import AppConfig
    from cryptotrader.models import DataSnapshot


def merge_dicts(a: dict, b: dict) -> dict:
    result = {**a}
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


class ArenaState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, Any], merge_dicts]
    metadata: Annotated[dict[str, Any], merge_dicts]
    debate_round: int
    max_debate_rounds: int
    divergence_scores: Annotated[list[float], operator.add]


def build_initial_state(
    pair: str,
    *,
    engine: str = "paper",
    exchange_id: str = "binance",
    timeframe: str | None = None,
    ohlcv_limit: int | None = None,
    snapshot: DataSnapshot | None = None,
    config: AppConfig | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """Build the initial state dict for graph invocation.

    Args:
        pair: Trading pair, e.g. ``"BTC/USDT"``.
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

    Returns:
        A fully-formed initial state dict ready to pass to ``graph.ainvoke()``.
    """
    if config is None:
        from cryptotrader.config import load_config

        config = load_config()

    metadata: dict[str, Any] = {
        "pair": pair,
        "engine": engine,
        "exchange_id": exchange_id,
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
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    initial_data: dict[str, Any] = {}
    if snapshot is not None:
        initial_data["snapshot"] = snapshot

    return {
        "messages": [],
        "data": initial_data,
        "metadata": metadata,
        "debate_round": 0,
        "max_debate_rounds": config.debate.max_rounds,
        "divergence_scores": [],
    }
