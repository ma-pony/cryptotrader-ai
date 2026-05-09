"""Base agent with LLM-powered analysis.

Two agent types:
- BaseAgent: Single LLM call, for agents with complete pre-computed data (TechAgent, MacroAgent)
- ToolAgent: LangChain create_agent loop, for agents that need to actively query data (ChainAgent, NewsAgent)

All LLM calls go through unified LangChain gateway with SQLiteCache.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from cryptotrader.models import AgentAnalysis, DataSnapshot
from cryptotrader.security import sanitize_input

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cryptotrader.agents.prompt_builder import PromptBuilder

_logger = logging.getLogger(__name__)
logger = _logger
_structlog = structlog.get_logger(__name__)

# Funding rate thresholds shared with debate/verdict.py
FUNDING_RATE_HIGH = 0.0003  # above → crowded long
FUNDING_RATE_LOW = -0.0001  # below → crowded short

# ── LangChain SQLiteCache initialization ──

_cache_initialized = False
_cache_disabled = False


def disable_llm_cache() -> None:
    """Disable LLM cache (backtest mode). Reversible via ``restore_llm_cache()``.

    Prevents cross-contamination between backtest and live LLM responses.
    """
    global _cache_disabled
    _cache_disabled = True
    try:
        from langchain_core.globals import set_llm_cache

        set_llm_cache(None)  # type: ignore[arg-type]
        _logger.info("LLM cache disabled (backtest mode)")
    except Exception:
        _logger.info("Failed to disable LLM cache", exc_info=True)


def restore_llm_cache() -> None:
    """Re-enable SQLiteCache after a backtest run."""
    global _cache_disabled, _cache_initialized
    _cache_disabled = False
    _cache_initialized = False  # allow _init_cache() to run again
    _init_cache()


def _init_cache() -> None:
    """Initialize LangChain SQLiteCache for exact-match LLM caching."""
    global _cache_initialized
    if _cache_initialized:
        return
    _cache_initialized = True
    if _cache_disabled:
        return
    try:
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache

        cache_dir = Path.home() / ".cryptotrader"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_db = cache_dir / "llm_cache.db"
        set_llm_cache(SQLiteCache(database_path=str(cache_db)))
        _logger.info("LLM cache initialized: %s", cache_db)
    except Exception:
        _logger.warning("Failed to initialize LLM cache, continuing without cache", exc_info=True)


# ── Unified LLM factory ──


def _build_llm_kwargs(
    model: str, temperature: float, timeout: int, llm_cfg, *, json_mode: bool = False
) -> dict[str, Any]:
    """Build kwargs dict for ChatOpenAI constructor."""
    kwargs: dict[str, Any] = {"model": model, "temperature": temperature, "timeout": timeout}
    if llm_cfg.base_url:
        kwargs["base_url"] = llm_cfg.base_url
    if llm_cfg.api_key:
        kwargs["api_key"] = llm_cfg.api_key
    if model in llm_cfg.streaming_models:
        kwargs["streaming"] = True
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return kwargs


def _try_manifest_llm(
    role: str,
    cfg,
    temperature: float,
    timeout: int,
    json_mode: bool,
    retry_cfg,
    *,
    track_tokens: bool = True,
) -> ChatOpenAI | None:
    """Attempt to build an LLM via the models.toml manifest for the given role.

    When ``track_tokens`` is True, the resulting runnable is bound with the shared
    :class:`TokenTrackerCallback` so token accounting works identically to the
    direct path (see :func:`create_llm`).
    """
    from cryptotrader.llm.factory import build_resilient_llm
    from cryptotrader.llm.registry import load_manifest

    manifest_path = Path(cfg.models.models_path) if cfg.models.models_path else None
    manifest = load_manifest(manifest_path)
    if manifest is None:
        return None
    role_cfg = manifest.get_role(role)
    if role_cfg is None:
        return None
    resilient = build_resilient_llm(
        role_cfg,
        manifest,
        temperature,
        timeout,
        json_mode,
        retry_cfg,
        role=role,
    )
    if track_tokens and resilient is not None:
        from cryptotrader.llm.token_tracker import default_callback

        # Runnable.with_config({'callbacks': [...]}) binds the callback for every
        # ainvoke/invoke on the chain, including the resilient fallback wrapper.
        try:
            resilient = resilient.with_config({"callbacks": [default_callback()]})
        except Exception:
            logger.info("manifest llm: failed to bind token tracker callback", exc_info=True)
    return resilient


def create_llm(
    model: str = "",
    temperature: float | None = None,
    timeout: int | None = None,
    json_mode: bool = False,
    *,
    with_fallback: bool = True,
    role: str = "",
    track_tokens: bool = True,
) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance with unified config.

    All LLM calls in the project route through this factory to ensure
    consistent configuration, caching, and fallback behavior.

    When ``role`` is provided and ``config/models.toml`` exists, builds a
    multi-provider fallback chain with per-provider retry middleware.

    Args:
        track_tokens: When True (default), attaches the shared TokenTrackerCallback
            so token usage accumulates into the context-bound ledger. Set to False
            for zero-overhead LLMs in test/backtest contexts that manage their own
            accounting.
    """
    from cryptotrader.config import load_config
    from cryptotrader.metrics import get_metrics_collector

    _init_cache()

    cfg = load_config()
    llm_cfg = cfg.llm
    retry_cfg = llm_cfg.retry

    if temperature is None:
        temperature = llm_cfg.default_temperature
    if timeout is None:
        timeout = llm_cfg.timeout

    if role:
        get_metrics_collector().inc_llm_calls(model=role, node="create_llm")
        resilient = _try_manifest_llm(
            role,
            cfg,
            temperature,
            timeout,
            json_mode,
            retry_cfg,
            track_tokens=track_tokens,
        )
        if resilient is not None:
            return resilient

    if not model:
        model = cfg.models.analysis or cfg.models.fallback
    if not model:
        raise ValueError("No LLM model configured — set models.analysis or models.fallback in config/default.toml")

    get_metrics_collector().inc_llm_calls(model=model, node="create_llm")

    kwargs = _build_llm_kwargs(model, temperature, timeout, llm_cfg, json_mode=json_mode)
    if track_tokens:
        # Attach token tracker so each decision pipeline accumulates input/output
        # tokens into the ContextVar-bound ledger (see llm.token_tracker).
        from cryptotrader.llm.token_tracker import default_callback

        existing = kwargs.get("callbacks") or []
        kwargs["callbacks"] = [*existing, default_callback()]
    llm = ChatOpenAI(**kwargs)

    from cryptotrader.llm.factory import _wrap_with_retry

    llm = _wrap_with_retry(llm, retry_cfg)

    if with_fallback:
        fallback_model = cfg.models.fallback
        if fallback_model and fallback_model != model:
            from langchain_core.runnables import RunnableWithFallbacks

            fallback_kwargs = _build_llm_kwargs(fallback_model, temperature, timeout, llm_cfg, json_mode=json_mode)
            fb_llm = _wrap_with_retry(ChatOpenAI(**fallback_kwargs), retry_cfg)
            llm = RunnableWithFallbacks(
                runnable=llm,
                fallbacks=[fb_llm],
                exceptions_to_handle=(Exception,),
            )

    return llm


def _to_langchain_messages(messages: list[dict]) -> list:
    """Convert OpenAI-format message dicts to LangChain message objects."""
    result = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(HumanMessage(content=content))
        else:
            result.append(AIMessage(content=content))
    return result


def extract_content(response: AIMessage | Any) -> str:
    """Extract text content from LLM response.

    Handles both LangChain AIMessage and reasoning models (deepseek-reasoner)
    that may return content in additional_kwargs['reasoning_content'].
    """
    if isinstance(response, AIMessage):
        text = response.content or ""
        if not text:
            # deepseek-reasoner puts answer in additional_kwargs
            text = response.additional_kwargs.get("reasoning_content", "")
        return text
    return str(response)


def log_llm_usage(response: Any, *, caller: str) -> None:
    """记录 LLM 调用的 token 消耗到结构化日志。

    从 AIMessage.usage_metadata 中提取 input_tokens, output_tokens, model_name,
    并通过 structlog 以 llm_usage 事件命名空间记录, 支持后续按时间窗口汇总成本报告。

    Args:
        response: LLM 调用返回的 AIMessage 对象.
        caller: 调用方标识 (如 agent_id 或函数名), 用于日志定位.
    """
    if not isinstance(response, AIMessage):
        return

    usage = response.usage_metadata
    if not usage:
        return

    input_tokens: int = usage.get("input_tokens", 0)
    output_tokens: int = usage.get("output_tokens", 0)
    model_name: str = (response.response_metadata or {}).get("model_name", "unknown")

    cache_read = usage.get("cache_read_input_tokens", 0)
    if not cache_read:
        prompt_details = usage.get("input_token_details") or {}
        cache_read = prompt_details.get("cached", 0) if isinstance(prompt_details, dict) else 0

    _structlog.info(
        "llm_usage",
        caller=caller,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        prompt_cache_hit=cache_read > 0,
        cache_read_input_tokens=cache_read,
    )


async def acompletion_with_fallback(*, model: str, **kwargs) -> AIMessage:
    """Unified LLM call via LangChain with automatic fallback.

    Accepts OpenAI-format kwargs and returns AIMessage.
    """
    messages = kwargs.pop("messages", [])
    temperature = kwargs.pop("temperature", None)
    timeout = kwargs.pop("timeout", None)
    response_format = kwargs.pop("response_format", None)
    json_mode = response_format is not None and response_format.get("type") == "json_object"

    llm = create_llm(model=model, temperature=temperature, timeout=timeout, json_mode=json_mode)
    lc_messages = _to_langchain_messages(messages)
    from cryptotrader.llm.prompt_cache import apply_cache_control, should_cache

    if should_cache(model=model):
        lc_messages = apply_cache_control(lc_messages)
    response = await llm.ainvoke(lc_messages)
    log_llm_usage(response, caller="acompletion_with_fallback")
    return response


class BaseAgent:
    """Single-LLM-call agent using PromptBuilder for prompt assembly (spec 017b)."""

    def __init__(self, *, agent_id: str, prompt_builder: PromptBuilder, model: str = "") -> None:
        self.agent_id = agent_id
        self._prompt_builder = prompt_builder
        self.model = model

    def _resolve_model(self) -> str:
        """Return model name, falling back to config if not set."""
        if self.model:
            return self.model
        from cryptotrader.config import load_config

        cfg = load_config()
        return cfg.models.analysis or cfg.models.fallback

    def _snapshot_to_dict(self, snapshot: DataSnapshot) -> dict:
        """Convert DataSnapshot to dict consumable by render_crypto_snapshot."""
        liq = snapshot.onchain.liquidations_24h if snapshot.onchain else {}
        return {
            "pair": snapshot.pair,
            "timestamp": str(snapshot.timestamp),
            "ticker": snapshot.market.ticker,
            "volatility": snapshot.market.volatility,
            "funding_rate": snapshot.market.funding_rate,
            "onchain": {
                "open_interest": snapshot.onchain.open_interest,
                "exchange_netflow": snapshot.onchain.exchange_netflow,
                "liquidations_24h": liq,
            },
            "news": {
                "headlines": list(snapshot.news.headlines),
            },
            "macro": {
                "fed_rate": snapshot.macro.fed_rate,
                "dxy": snapshot.macro.dxy,
            },
        }

    async def analyze(self, snapshot: DataSnapshot, experience: str = "") -> AgentAnalysis:
        try:
            sys_msg, usr_msg = self._prompt_builder.build(
                snapshot=self._snapshot_to_dict(snapshot),
                portfolio={},
                experience=experience,
            )
            model = self._resolve_model()
            llm = create_llm(model=model)
            messages = [sys_msg, usr_msg]
            from cryptotrader.llm.prompt_cache import apply_cache_control, should_cache

            if should_cache(model=model, role=self.agent_id):
                messages = apply_cache_control(messages)
            response = await llm.ainvoke(messages)
            log_llm_usage(response, caller=self.agent_id)
            text = extract_content(response)
            return await self._parse_response(text, snapshot.pair, llm=llm)
        except Exception as exc:
            from cryptotrader.llm.errors import LLMProvidersExhaustedError

            if isinstance(exc, LLMProvidersExhaustedError):
                _structlog.warning(
                    "llm_all_providers_exhausted",
                    role=exc.role,
                    providers_tried=exc.providers_tried,
                    agent_id=self.agent_id,
                )
            else:
                logger.exception("LLM call failed for %s, returning mock analysis", self.agent_id)
            return AgentAnalysis(
                agent_id=self.agent_id,
                pair=snapshot.pair,
                direction="neutral",
                confidence=0.1,
                reasoning="LLM unavailable - mock analysis",
                is_mock=True,
                data_sufficiency="low",
            )

    @staticmethod
    def _regex_fallback(text: str) -> dict | None:
        """Last-resort extraction of direction/confidence from free-text responses."""
        import re

        text_lower = text.lower()
        direction = "neutral"
        for d in ("bullish", "bearish"):
            if d in text_lower:
                direction = d
                break

        conf_match = re.search(r"confidence[\":\s]+(0\.\d+)", text_lower)
        confidence = float(conf_match.group(1)) if conf_match else 0.3

        # Only use fallback if we found a clear directional signal
        if direction == "neutral" and not conf_match:
            return None

        # Extract reasoning from first meaningful sentence, sanitized + capped
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        raw_reasoning = ". ".join(sentences[:2]) + "." if sentences else "Extracted via regex fallback"
        reasoning = sanitize_input(raw_reasoning, max_chars=500)

        return {
            "direction": direction,
            "confidence": confidence,
            "reasoning": reasoning,
            "data_sufficiency": "low",
            "key_factors": [],
            "risk_flags": [],
        }

    async def _parse_response(self, response_text: str, pair: str, llm=None) -> AgentAnalysis:
        from cryptotrader.llm.json_retry import extract_json_with_retry

        data = await extract_json_with_retry(
            response_text,
            llm=llm,
            schema_hint="direction,confidence,reasoning,key_factors,risk_flags,data_sufficiency",
            max_retries=2,
        )
        if not data:
            data = self._regex_fallback(response_text)
            if data is None:
                logger.warning("Failed to parse LLM response for %s: %.200s", self.agent_id, response_text)
                return AgentAnalysis(
                    agent_id=self.agent_id,
                    pair=pair,
                    direction="neutral",
                    confidence=0.1,
                    reasoning=f"Parse failure — raw: {response_text[:300]}",
                    is_mock=True,
                    data_sufficiency="low",
                )
            data["_regex_fallback"] = True
        # Standard fields
        standard = {
            "direction",
            "confidence",
            "reasoning",
            "key_factors",
            "risk_flags",
            "data_points",
            "data_sufficiency",
        }
        # Everything else goes into data_points for downstream rules engine
        extra = {k: v for k, v in data.items() if k not in standard}
        dp = data.get("data_points", {})
        dp.update(extra)

        # Parse data_sufficiency and enforce confidence cap for low-data agents
        sufficiency = data.get("data_sufficiency", "medium")
        if sufficiency not in ("high", "medium", "low"):
            sufficiency = "medium"
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        if sufficiency == "low":
            confidence = min(confidence, 0.3)

        return AgentAnalysis(
            agent_id=self.agent_id,
            pair=pair,
            direction=data.get("direction", "neutral"),
            confidence=confidence,
            reasoning=data.get("reasoning", ""),
            key_factors=data.get("key_factors", []),
            risk_flags=data.get("risk_flags", []),
            data_points=dp,
            data_sufficiency=sufficiency,
            is_mock=bool(data.get("_regex_fallback")),
        )


def _create_chat_model(model: str, temperature: float = 0.2):
    """Create a LangChain chat model without fallback (for ToolAgent / create_agent)."""
    return create_llm(model=model, temperature=temperature, with_fallback=False)


class ToolAgent(BaseAgent):
    """Agent with tool-calling capability via LangChain create_agent (spec 017b).

    Unlike BaseAgent (single LLM call), ToolAgent runs a model->tool->model loop,
    allowing the AI to actively query data sources during analysis.

    In backtest_mode, skips tool-calling entirely and uses BaseAgent's single LLM call
    to avoid forward-looking bias from real-time API calls and speed up backtesting.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        prompt_builder: PromptBuilder,
        tools: Sequence,
        model: str = "",
        backtest_mode: bool = False,
    ) -> None:
        super().__init__(agent_id=agent_id, prompt_builder=prompt_builder, model=model)
        self.tools = list(tools)
        self.backtest_mode = backtest_mode

    async def analyze(self, snapshot: DataSnapshot, experience: str = "") -> AgentAnalysis:
        # In backtest mode, skip tool-calling to avoid forward-looking bias
        if self.backtest_mode:
            return await super().analyze(snapshot, experience)

        try:
            from langchain.agents import create_agent

            sys_msg, usr_msg = self._prompt_builder.build(
                snapshot=self._snapshot_to_dict(snapshot),
                portfolio={},
                experience=experience,
            )

            llm = _create_chat_model(self.model)
            agent = create_agent(llm, tools=self.tools, system_prompt=sys_msg.content)

            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": usr_msg.content}]},
            )

            # Extract final message text from agent output
            final_msg = result["messages"][-1]
            text = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
            return await self._parse_response(text, snapshot.pair, llm=llm)

        except Exception:
            logger.exception("ToolAgent call failed for %s, falling back to single-call", self.agent_id)
            # Fallback to single LLM call (same as BaseAgent)
            return await super().analyze(snapshot, experience)
