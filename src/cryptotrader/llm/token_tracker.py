"""Token + cost accounting for LLM calls.

Per-decision ledgers accumulate input/output tokens and USD cost across every
``create_llm`` invocation within the decision pipeline. The ledger is attached
to the active asyncio context via ``ContextVar`` so LangChain callbacks can
update it without explicit plumbing.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


# USD per 1 000 000 tokens — (input_cost, output_cost).
# Values track vendor public pricing as of 2026-04 (Anthropic + OpenAI).
MODEL_COSTS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-opus-4-1-20250805": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    # OpenAI
    "gpt-5": (1.25, 10.0),
    "gpt-5.5": (1.25, 10.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "o3-mini": (1.10, 4.40),
    "o1-mini": (1.10, 4.40),
    # Fallback
    "gpt-3.5-turbo": (0.50, 1.50),
}


# Cache resolved prefix matches so we only walk the sorted list once per unique model name.
_RESOLVED_COSTS: dict[str, tuple[float, float]] = {}


def _config_model_costs() -> dict[str, tuple[float, float]]:
    """Read ``[[llm.model_costs]]`` entries from config, if any.

    Config values override/extend the hardcoded ``MODEL_COSTS`` table so ops can
    adjust prices without a deploy. Returns empty dict when config unavailable.
    """
    try:
        from cryptotrader.config import load_config

        entries = load_config().llm.model_costs or []
    except Exception:
        logger.debug("model_costs: config read failed", exc_info=True)
        return {}
    return {e.name: (float(e.input_usd_per_mtok), float(e.output_usd_per_mtok)) for e in entries if e.name}


def _match_cost(model: str) -> tuple[float, float]:
    """Resolve a model string to (input_cost_per_mtok, output_cost_per_mtok).

    Resolution order (first match wins, longest-prefix inside each layer):
      1. Exact match in config ``[[llm.model_costs]]``
      2. Exact match in hardcoded ``MODEL_COSTS``
      3. Longest-prefix match across (config + hardcoded) combined
      4. Zero cost (logged once per unknown model)

    The longest-prefix rule prevents ``gpt-4o-mini-YYYY-MM-DD`` from matching
    ``gpt-4o`` instead of ``gpt-4o-mini`` (which is 17x cheaper).
    """
    config_costs = _config_model_costs()
    combined = {**MODEL_COSTS, **config_costs}  # config wins on exact conflict

    if model in combined:
        return combined[model]
    if model in _RESOLVED_COSTS:
        return _RESOLVED_COSTS[model]
    for key in sorted(combined, key=len, reverse=True):
        if model.startswith(key):
            _RESOLVED_COSTS[model] = combined[key]
            return combined[key]
    logger.info("Unknown model for cost tracking: %s — cost=$0", model)
    _RESOLVED_COSTS[model] = (0.0, 0.0)
    return (0.0, 0.0)


@dataclass
class TokenLedger:
    """Running totals for a single decision pipeline."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, *, model: str, input_tokens: int, output_tokens: int, cache_hit: bool = False) -> None:
        in_cost_m, out_cost_m = _match_cost(model)
        delta = (input_tokens / 1_000_000.0) * in_cost_m + (output_tokens / 1_000_000.0) * out_cost_m
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1
        self.cost_usd += delta
        if cache_hit:
            self.cache_hits += 1
        stats = self.by_model.setdefault(model, {"input": 0, "output": 0, "calls": 0, "cost_usd": 0.0})
        stats["input"] = stats.get("input", 0) + input_tokens
        stats["output"] = stats.get("output", 0) + output_tokens
        stats["calls"] = stats.get("calls", 0) + 1
        stats["cost_usd"] = stats.get("cost_usd", 0.0) + delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": float(self.input_tokens),
            "output_tokens": float(self.output_tokens),
            "cache_hits": float(self.cache_hits),
            "calls": float(self.calls),
            "cost_usd": round(self.cost_usd, 6),
            "by_model": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in self.by_model.items()},
        }


_ledger_ctx: ContextVar[TokenLedger | None] = ContextVar("token_ledger", default=None)


def start_ledger() -> TokenLedger:
    """Bind a fresh ledger to the current context; replace any existing one."""
    ledger = TokenLedger()
    _ledger_ctx.set(ledger)
    return ledger


def current_ledger() -> TokenLedger | None:
    return _ledger_ctx.get()


def set_ledger(ledger: TokenLedger | None) -> None:
    _ledger_ctx.set(ledger)


def _extract_usage(response: Any) -> tuple[str, int, int, bool]:
    """Pull (model, input, output, cache_hit) out of a LangChain response.

    Handles both ``LLMResult`` (returned by ``generate``) and raw ``AIMessage``
    objects. Falls back to zero when metadata is absent.
    """
    usage_meta: dict[str, Any] | None = None
    model = ""

    if isinstance(response, AIMessage):
        usage_meta = getattr(response, "usage_metadata", None)
        model = (response.response_metadata or {}).get("model_name", "") or ""
    elif isinstance(response, LLMResult):
        gens = response.generations or []
        if gens and gens[0]:
            first = gens[0][0]
            msg = getattr(first, "message", None)
            if isinstance(msg, AIMessage):
                usage_meta = getattr(msg, "usage_metadata", None)
                model = (msg.response_metadata or {}).get("model_name", "") or ""
        llm_output = response.llm_output or {}
        model = model or llm_output.get("model_name", "")
        if not usage_meta:
            token_usage = llm_output.get("token_usage") or {}
            usage_meta = {
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
            }

    if not usage_meta:
        return model, 0, 0, False

    input_tokens = int(usage_meta.get("input_tokens", 0) or 0)
    output_tokens = int(usage_meta.get("output_tokens", 0) or 0)
    cache_read = int((usage_meta.get("input_token_details") or {}).get("cache_read", 0) or 0)
    cache_hit = cache_read > 0
    return model, input_tokens, output_tokens, cache_hit


class TokenTrackerCallback(BaseCallbackHandler):
    """LangChain callback that records usage into the active ledger."""

    raise_error = False

    def on_llm_end(self, response: LLMResult, **_: Any) -> None:  # type: ignore[override]
        ledger = _ledger_ctx.get()
        if ledger is None:
            return
        try:
            model, input_tokens, output_tokens, cache_hit = _extract_usage(response)
            if input_tokens == 0 and output_tokens == 0:
                return
            ledger.record(
                model=model or "unknown",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_hit=cache_hit,
            )
        except Exception:  # pragma: no cover — never break the pipeline
            logger.debug("token tracker on_llm_end failed", exc_info=True)

    def on_chat_model_end(self, response: LLMResult, **_: Any) -> None:
        # Some provider integrations dispatch the chat variant instead.
        self.on_llm_end(response)


_SHARED_CALLBACK = TokenTrackerCallback()


def default_callback() -> TokenTrackerCallback:
    """Single shared instance — safe because callbacks are context-driven."""
    return _SHARED_CALLBACK
