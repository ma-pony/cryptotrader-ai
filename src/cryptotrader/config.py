"""Unified configuration — loaded from config/default.toml."""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.agents.prompt_builder import PromptBuilder

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from cryptotrader.mcp.config import MCPConfig, MCPServerConfig
from cryptotrader.pair import Pair

logger = logging.getLogger(__name__)

# ── LLM gateway ──


@dataclass
class RetryConfig:
    max_attempts: int = 3
    retry_base_delay_s: float = 1.0
    retry_backoff_factor: float = 2.0
    retry_jitter: bool = True


@dataclass
class LLMModelCostConfig:
    """USD cost per 1M tokens for a specific model.

    Populated from ``[[llm.model_costs]]`` TOML blocks. Any model not present in
    this table falls back to the hardcoded table in ``llm.token_tracker.MODEL_COSTS``.
    """

    name: str = ""
    input_usd_per_mtok: float = 0.0
    output_usd_per_mtok: float = 0.0


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = ""
    streaming_models: list[str] = field(default_factory=list)
    default_temperature: float = 0.2
    timeout: int = 120
    prompt_caching: bool = True
    retry: RetryConfig = field(default_factory=RetryConfig)
    model_costs: list[LLMModelCostConfig] = field(default_factory=list)


# ── Model names ──


@dataclass
class ModelConfig:
    analysis: str = "gemini-3-flash"
    debate: str = "gemini-3-flash"
    verdict: str = "gpt-5.4"
    tech_agent: str = "gemini-3-flash"
    chain_agent: str = "gemini-3.1-pro"
    news_agent: str = "gemini-3.1-pro"
    macro_agent: str = "gemini-3.1-pro"
    fallback: str = "deepseek-chat"
    timeout_seconds: int = 90
    models_path: str = ""


# ── Debate ──


@dataclass
class DebateConfig:
    max_rounds: int = 3
    convergence_threshold: float = 0.1
    divergence_hold_threshold: float = 0.7
    skip_debate: bool = True
    consensus_skip_threshold: float = 0.5
    confusion_skip_threshold: float = 0.05
    confusion_max_dispersion: float = 0.2


# ── Data ──


@dataclass
class DataConfig:
    default_timeframe: str = "1h"
    ohlcv_limit: int = 100


# ── Risk ──


@dataclass
class PositionConfig:
    # Max single-position notional as fraction of equity. Forces diversification:
    # 0.50 means no one trade can take more than half the budget on its own.
    max_single_pct: float = 0.50
    # Max sum-of-positions notional as fraction of equity. Caps total price
    # exposure independent of leverage. 1.00 = up to 1x equity in notional;
    # raise to 1.5-2.0 for explicitly leveraged strategies. Was 0.50 in the
    # spot-only era; bumped after dual-cap refactor (2026-05-07) so leverage
    # actually buys capital efficiency.
    max_total_exposure_pct: float = 1.00
    # Max fraction of equity locked as margin. Independent of notional —
    # this is the "free margin buffer" guarantee. With 2x leverage and
    # max_total_exposure_pct=1.00, max_margin_used_pct=0.50 is a tight pair
    # (barely consistent); 0.40 leaves a real buffer for adverse moves +
    # stop-loss execution + funding payments.
    max_margin_used_pct: float = 0.40
    max_correlated_positions: int = 2
    # Cap on simultaneous positions in a single direction (long or short).
    # When 3+ positions already share a direction, the bot is making a single
    # macro bet, not a diversified portfolio — refuse the (N+1)th to prevent
    # synchronous stop-loss cascade exhausting the daily-loss budget. Set to
    # 999 to disable; default 3 of a 5-pair set keeps headroom for one
    # contrarian position.
    max_same_direction_positions: int = 3


@dataclass
class LossConfig:
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    max_cvar_95: float = 0.05
    cvar_min_returns: int = 20


@dataclass
class CooldownConfig:
    same_pair_minutes: int = 60
    post_loss_minutes: int = 120


@dataclass
class VolatilityConfig:
    flash_crash_threshold: float = 0.05
    funding_rate_threshold: float = 0.005
    flash_crash_lookback: int = 10


@dataclass
class ExchangeCheckConfig:
    max_api_latency_ms: int = 2000
    health_check_interval_s: int = 30


@dataclass
class RateLimitConfig:
    max_trades_per_hour: int = 6
    max_trades_per_day: int = 20


@dataclass
class RiskConfig:
    max_stop_loss_pct: float = 0.05
    position: PositionConfig = field(default_factory=PositionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    cooldown: CooldownConfig = field(default_factory=CooldownConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    exchange: ExchangeCheckConfig = field(default_factory=ExchangeCheckConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)


# ── Backtest ──


@dataclass
class BacktestPositionSizingConfig:
    high_confidence_pct: float = 0.20
    medium_confidence_pct: float = 0.12
    low_confidence_pct: float = 0.06


@dataclass
class BacktestConfig:
    initial_capital: float = 10000
    slippage_base: float = 0.0005
    fee_bps: float = 10.0
    sma_fast: int = 20
    sma_slow: int = 50
    lookback: int = 60
    default_position_pct: float = 0.1
    position_sizing: BacktestPositionSizingConfig = field(default_factory=BacktestPositionSizingConfig)


# ── Portfolio (live trading inception baseline) ──


@dataclass
class PortfolioConfig:
    """Live-trading portfolio settings.

    ``initial_capital`` is the baseline for the dashboard's "总收益"
    (inception-to-date return) calculation. ``0.0`` (default) falls back to
    the earliest portfolio_snapshots row — adequate when no deposits or
    withdrawals occur after the system starts tracking. Set explicitly to
    pin the baseline if you fund the account in stages or want a fixed
    reference (e.g. set 100000 if you started with $100K USDT).
    """

    initial_capital: float = 0.0


# ── Regime thresholds ──


@dataclass
class RegimeThresholdsConfig:
    high_funding: float = 0.0003
    negative_funding: float = -0.0001
    high_vol: float = 0.025
    low_vol: float = 0.010
    trending_up: float = 0.05
    trending_down: float = -0.05
    extreme_fear_fng: int = 25
    extreme_greed_fng: int = 75


# ── Experience (replaces Reflection) ──


@dataclass
class ExperienceConfig:
    enabled: bool = True
    every_n_cycles: int = 20
    min_commits_required: int = 10
    lookback_commits: int = 30
    model: str = ""  # empty = use models.analysis
    token_budget_pct: float = 0.30
    verify_win_rate_tolerance: float = 0.15
    min_cases_per_pattern: int = 5
    regime_thresholds: RegimeThresholdsConfig = field(default_factory=RegimeThresholdsConfig)


# ── Execution ──


@dataclass
class ExecutionConfig:
    order_wait_seconds: int = 30
    retry_attempts: int = 3
    graph_timeout_s: int = 300


# ── Evolution Daemon ──


@dataclass
class EvolutionDaemonConfig:
    """spec 022 FR-D4: Evolution reflect daemon configuration."""

    enabled: bool = True
    cron: str = "0 0 * * *"
    actions: list = field(default_factory=lambda: ["pareto", "regime", "skill_proposal"])
    llm_model: str = ""
    propose_threshold: int = 10


# ── Scheduler ──


@dataclass
class SchedulerConfig:
    enabled: bool = False
    # Per spec 013-pair-value-object D4: pairs is list[Pair].
    # Legacy ``pairs = ["BTC/USDT"]`` form in TOML and new
    # ``[[scheduler.pairs]] symbol=... market=... settle=...`` form are both
    # accepted by ``_build_scheduler_config()``; this dataclass stores the
    # parsed result (list of Pair instances). When constructed without
    # arguments (tests, programmatic use), defaults to two spot pairs.
    pairs: list[Pair] = field(default_factory=lambda: [Pair.parse("BTC/USDT"), Pair.parse("ETH/USDT")])
    interval_minutes: int = 240
    exchange_id: str = "binance"
    daily_summary_hour: int = 0


# ── Providers ──


@dataclass
class ProvidersConfig:
    coinglass_api_key: str = ""
    cryptoquant_api_key: str = ""
    whale_alert_api_key: str = ""
    fred_api_key: str = ""
    coinglass_enabled: bool = True
    cryptoquant_enabled: bool = True
    whale_alert_enabled: bool = True
    fred_enabled: bool = True
    defillama_enabled: bool = True
    coingecko_enabled: bool = True
    okx_api_key: str = ""
    okx_secret_key: str = ""
    okx_passphrase: str = ""
    okx_enabled: bool = False
    sosovalue_api_key: str = ""
    sosovalue_enabled: bool = True
    coindesk_api_key: str = ""
    binance_audit_enabled: bool = True
    binance_sentiment_enabled: bool = True
    enforce_token_security: bool = True
    max_acceptable_risk: str = "MEDIUM"
    token_tax_threshold: int = 10

    def has_okx_credentials(self) -> bool:
        return all([self.okx_api_key, self.okx_secret_key, self.okx_passphrase])


# ── Notifications ──


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = False


@dataclass
class NotificationsConfig:
    webhook_url: str = ""
    enabled: bool = True
    webhook_timeout: int = 5
    events: list[str] = field(
        default_factory=lambda: ["trade", "rejection", "circuit_breaker", "reconcile_mismatch", "daily_summary"]
    )
    telegram: TelegramConfig = field(default_factory=TelegramConfig)


# ── Triggers ──


@dataclass
class TriggersConfig:
    enabled: bool = False
    max_rules: int = 50
    ws_reconnect_max_s: int = 60
    funding_rate_poll_interval_minutes: int = 5


# ── Exchanges ──


@dataclass
class ExchangeCredentials:
    api_key: str = ""
    secret: str = ""
    passphrase: str = ""
    sandbox: bool = True
    # Perp leverage applied via `set_leverage` on first contact with each perp
    # symbol. 1 = no-op (keeps OKX default; avoids extra API calls). Spot
    # symbols ignore this.
    leverage: int = 1
    # OKX margin mode for perp positions: "isolated" (per-position margin) or
    # "cross" (shared cross-margin pool). Used only when leverage > 1.
    margin_mode: str = "isolated"


@dataclass
class ExchangesConfig:
    _exchanges: dict[str, ExchangeCredentials] = field(default_factory=dict)

    def get(self, exchange_id: str) -> ExchangeCredentials | None:
        return self._exchanges.get(exchange_id)

    def __iter__(self):
        return iter(self._exchanges)

    def items(self):
        return self._exchanges.items()


# ── Chart Analysis ──


@dataclass
class ChartAnalysisConfig:
    vision_models: list[str] = field(
        default_factory=lambda: [
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
            "claude-opus-4-5",
            "gemini-3.1-pro",
            "gemini-3-flash",
        ]
    )
    fast_model: str = ""
    max_image_bytes: int = 4_718_592
    description_max_tokens: int = 800


@dataclass
class HitlTelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


@dataclass
class HitlConfig:
    enabled: bool = False
    min_position_scale: float = 0.5
    divergence_threshold: float = 0.6
    cold_start_min_trades: int = 5
    approval_timeout_seconds: int = 300
    telegram: HitlTelegramConfig = field(default_factory=HitlTelegramConfig)


@dataclass
class ChatConfig:
    event_buffer_ttl_seconds: int = 300
    max_concurrent_tasks: int = 10
    max_steering_instruction_chars: int = 500
    event_buffer_max_size: int = 500


# ── Infrastructure ──


@dataclass
class InfrastructureConfig:
    database_url: str = ""
    redis_url: str = ""


# ── Agents ──

_BUILTIN_AGENT_IDS = ("tech_agent", "chain_agent", "news_agent", "macro_agent")


@dataclass
class AgentConfig:
    agent_id: str = ""
    model: str = ""
    timeout_seconds: int = 0
    enabled: bool = True
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    regime_skills: dict[str, list[str]] = field(default_factory=dict)


class AgentNotFoundError(KeyError):
    def __init__(self, agent_id: str, registered: list[str]) -> None:
        self.agent_id = agent_id
        self.registered = registered
        super().__init__(f"Agent {agent_id!r} not found. Registered agents: {registered}")


@dataclass
class AgentsConfig:
    _agents: dict[str, AgentConfig] = field(default_factory=dict)

    def get(self, agent_id: str) -> AgentConfig | None:
        cfg = self._agents.get(agent_id)
        if cfg is not None:
            return cfg
        if agent_id in _BUILTIN_AGENT_IDS:
            return AgentConfig(agent_id=agent_id)
        return None

    def list_active(self) -> list[AgentConfig]:
        seen = set(self._agents.keys())
        result = [cfg for cfg in self._agents.values() if cfg.enabled]
        result.extend(AgentConfig(agent_id=bid) for bid in _BUILTIN_AGENT_IDS if bid not in seen)
        return result

    def build(
        self,
        agent_id: str,
        *,
        prompt_builder: PromptBuilder,
        backtest_mode: bool = False,
        model_override: str = "",
    ):
        """Dynamically construct an Agent instance from registry config.

        Returns BaseAgent or ToolAgent depending on agent_id and config.
        """
        from cryptotrader.agents.base import BaseAgent, ToolAgent

        agent_cfg = self._agents.get(agent_id)
        if agent_cfg is None:
            if agent_id in _BUILTIN_AGENT_IDS:
                agent_cfg = AgentConfig(agent_id=agent_id)
            else:
                raise AgentNotFoundError(agent_id, list(self._agents.keys()) + list(_BUILTIN_AGENT_IDS))

        if not agent_cfg.enabled:
            raise AgentNotFoundError(agent_id, [a.agent_id for a in self.list_active()])

        model = model_override or agent_cfg.model

        if agent_id in _BUILTIN_AGENT_IDS and not agent_cfg.tools:
            return self._build_builtin(
                agent_id, prompt_builder=prompt_builder, model=model, backtest_mode=backtest_mode
            )

        if agent_cfg.tools:
            tools = self._resolve_tools(agent_cfg)
            return ToolAgent(
                agent_id=agent_id,
                prompt_builder=prompt_builder,
                tools=tools,
                model=model,
                backtest_mode=backtest_mode,
            )
        return BaseAgent(agent_id=agent_id, prompt_builder=prompt_builder, model=model)

    @staticmethod
    def _build_builtin(
        agent_id: str,
        *,
        prompt_builder: PromptBuilder,
        model: str,
        backtest_mode: bool,
    ):
        from cryptotrader.agents.chain import ChainAgent
        from cryptotrader.agents.macro import MacroAgent
        from cryptotrader.agents.news import NewsAgent
        from cryptotrader.agents.tech import TechAgent

        builders = {
            "tech_agent": lambda: TechAgent(prompt_builder=prompt_builder, model=model),
            "chain_agent": lambda: ChainAgent(prompt_builder=prompt_builder, model=model, backtest_mode=backtest_mode),
            "news_agent": lambda: NewsAgent(prompt_builder=prompt_builder, model=model, backtest_mode=backtest_mode),
            "macro_agent": lambda: MacroAgent(prompt_builder=prompt_builder, model=model),
        }
        return builders[agent_id]()

    @staticmethod
    def _resolve_tools(agent_cfg: AgentConfig) -> list:
        from cryptotrader.agents.data_tools import CHAIN_TOOLS, NEWS_TOOLS

        all_tools = list(CHAIN_TOOLS) + list(NEWS_TOOLS)
        tool_by_name = {t.name: t for t in all_tools}
        result = []
        for name in agent_cfg.tools:
            if name in tool_by_name:
                result.append(tool_by_name[name])
            else:
                logger.warning("unknown tool '%s' in agents.%s.tools", name, agent_cfg.agent_id)
        return result


# ── Top-level ──


@dataclass
class AppConfig:
    mode: str = "standalone"
    engine: str = "paper"
    exchange_id: str = "binance"
    llm: LLMConfig = field(default_factory=LLMConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    experience: ExperienceConfig = field(default_factory=ExperienceConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    triggers: TriggersConfig = field(default_factory=TriggersConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    exchanges: ExchangesConfig = field(default_factory=ExchangesConfig)
    chart_analysis: ChartAnalysisConfig = field(default_factory=ChartAnalysisConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    hitl: HitlConfig = field(default_factory=HitlConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    evolution_daemon: EvolutionDaemonConfig = field(default_factory=EvolutionDaemonConfig)

    @property
    def reflection(self) -> ExperienceConfig:
        """Deprecated alias for experience config."""
        return self.experience


# ── Configuration validation ──


class ConfigurationError(ValueError):
    """Raised when AppConfig contains an invalid field value at startup.

    Attributes:
        field_path: Dot-separated path to the offending field (e.g. "risk.loss.max_daily_loss_pct").
        expected: Human-readable description of the expected constraint (e.g. "value in (0, 1)").
    """

    def __init__(self, *, field_path: str, expected: str) -> None:
        self.field_path = field_path
        self.expected = expected
        super().__init__(f"Configuration error: {field_path!r} — {expected}")


def validate_config(cfg: AppConfig) -> None:
    """Validate critical AppConfig constraints.  Raises ConfigurationError on violation.

    Checked constraints:
      - risk.loss.max_daily_loss_pct ∈ (0, 1)
      - risk.position.max_single_pct ∈ (0, 1)
      - debate.consensus_skip_threshold ∈ (0, 1)
      - models.fallback is non-empty (after strip)
    """
    _check_open_unit_interval(cfg.risk.loss.max_daily_loss_pct, "risk.loss.max_daily_loss_pct")
    _check_open_unit_interval(cfg.risk.position.max_single_pct, "risk.position.max_single_pct")
    _check_open_unit_interval(cfg.debate.consensus_skip_threshold, "debate.consensus_skip_threshold")
    if cfg.hitl.enabled:
        _check_open_unit_interval(cfg.hitl.min_position_scale, "hitl.min_position_scale")
    if not cfg.models.fallback.strip():
        raise ConfigurationError(
            field_path="models.fallback",
            expected="non-empty string (model name must be specified)",
        )
    # Agents validation
    active_agents = cfg.agents.list_active()
    if not active_agents:
        raise ConfigurationError(
            field_path="agents",
            expected="at least 1 enabled agent",
        )
    for ac in cfg.agents._agents.values():
        if ac.timeout_seconds != 0 and ac.timeout_seconds < 1:
            raise ConfigurationError(
                field_path=f"agents.{ac.agent_id}.timeout_seconds",
                expected=f"positive integer, got {ac.timeout_seconds}",
            )


def _check_open_unit_interval(value: float, field_path: str) -> None:
    """Assert that *value* is strictly inside the open interval (0, 1)."""
    if not (0.0 < value < 1.0):
        raise ConfigurationError(
            field_path=field_path,
            expected=f"value in open interval (0, 1), got {value!r}",
        )


# ── Environment variable overrides ──

_ENV_PREFIX = "CRYPTOTRADER_"


def _coerce_value(raw: str) -> bool | int | float | str:
    """Coerce an environment variable string to the most appropriate Python type.

    Rules (in order):
    - "true" / "false" (case-insensitive) -> bool
    - Pure integer string -> int
    - Numeric string with decimal point -> float
    - Anything else -> str (unchanged)
    """
    lower = raw.strip().lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _resolve_nested(data: dict, parts: list[str]) -> tuple[dict, str]:
    """Navigate (and create if missing) nested dict nodes; return (leaf_dict, leaf_key)."""
    node = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]  # type: ignore[assignment]
    return node, parts[-1]


def _adapt_numeric_type(value: bool | int | float | str, original: object) -> bool | int | float | str:
    """Ensure numeric type consistency when the original TOML value is numeric."""
    if isinstance(original, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    orig_is_plain_int = isinstance(original, int) and not isinstance(original, bool)
    if orig_is_plain_int and isinstance(value, float) and value == int(value):
        return int(value)
    return value


def apply_env_overrides(toml_data: dict[str, object]) -> dict[str, object]:
    """Merge CRYPTOTRADER_* environment variables into toml_data and return a new dict.

    The original dict is never mutated (deep-copied first).

    Key-path rules:
      - Strip the CRYPTOTRADER_ prefix.
      - Split the remainder on double-underscore (__) to form the nested path.
      - Convert each path segment to lowercase.
      - Single underscores are allowed inside key names.

    Example:
      CRYPTOTRADER_RISK__LOSS__MAX_DAILY_LOSS_PCT=0.03
      -> result["risk"]["loss"]["max_daily_loss_pct"] = 0.03

    Type conversion (via _coerce_value):
      "true"/"false" -> bool, integer strings -> int, decimal strings -> float, else str.

    If the original TOML value is numeric but the env var string cannot be parsed as a
    number (coercion falls back to str), a logger.warning is emitted and the key is
    skipped (original value preserved).

    Priority: environment variables > local.toml > default.toml.
    """
    result = copy.deepcopy(toml_data)

    for key, raw_value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue

        path_str = key[len(_ENV_PREFIX) :]
        parts = [p.lower() for p in path_str.split("__")]

        value = _coerce_value(raw_value)

        node, leaf_key = _resolve_nested(result, parts)
        original_value = node.get(leaf_key)

        # If the original TOML value is numeric but we could only produce a str,
        # the env var is not parseable as a number -> warn and skip.
        if isinstance(original_value, int | float) and not isinstance(original_value, bool) and isinstance(value, str):
            logger.warning(
                "Env var %s value %r cannot be parsed as %s; skipping key",
                key,
                raw_value,
                type(original_value).__name__,
            )
            continue

        node[leaf_key] = _adapt_numeric_type(value, original_value)

    return result


# ── TOML loading ──

_CONFIG_SEARCH_PATHS = [
    Path("config/default.toml"),
    Path(__file__).resolve().parent.parent.parent.parent / "config" / "default.toml",
]


def _find_config() -> Path | None:
    for p in _CONFIG_SEARCH_PATHS:
        if p.exists():
            return p
    return None


def _merge(target: dict, source: dict) -> dict:
    """Deep-merge source into target."""
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _merge(target[k], v)
        else:
            target[k] = v
    return target


def _build_experience_config(toml_data: dict) -> ExperienceConfig:
    """Build ExperienceConfig from the TOML ``[experience]`` section.

    (Legacy ``[reflection]`` fallback removed 2026-05-13 — the reflection
    subsystem itself is gone and no current toml carries that section.)
    """
    raw = dict(toml_data.get("experience", {}))
    regime_raw = raw.pop("regime_thresholds", {})
    return ExperienceConfig(
        **raw,
        regime_thresholds=RegimeThresholdsConfig(**regime_raw),
    )


def _build_notifications_config(toml_data: dict) -> NotificationsConfig:
    """Build NotificationsConfig, extracting nested [notifications.telegram]."""
    raw = dict(toml_data.get("notifications", {}))
    telegram_raw = raw.pop("telegram", {})
    return NotificationsConfig(**raw, telegram=TelegramConfig(**telegram_raw))


def _build_hitl_config(toml_data: dict) -> HitlConfig:
    """Build HitlConfig, extracting nested [hitl.telegram]."""
    raw = dict(toml_data.get("hitl", {}))
    telegram_raw = raw.pop("telegram", {})
    return HitlConfig(**raw, telegram=HitlTelegramConfig(**telegram_raw))


def _build_agents_config(toml_data: dict) -> AgentsConfig:
    """Build AgentsConfig from TOML [agents.*] sections."""
    agents_raw = toml_data.get("agents", {})
    agents_map: dict[str, AgentConfig] = {}
    for agent_id, agent_data in agents_raw.items():
        if not isinstance(agent_data, dict):
            continue
        raw = dict(agent_data)
        raw.setdefault("agent_id", agent_id)
        regime_skills_raw = raw.pop("regime_skills", {})
        regime_skills: dict[str, list[str]] = {}
        for tag, skill_list in regime_skills_raw.items():
            if isinstance(skill_list, list):
                regime_skills[tag] = [str(s) for s in skill_list]
        agents_map[agent_id] = AgentConfig(**raw, regime_skills=regime_skills)
    return AgentsConfig(_agents=agents_map)


def _build_mcp_config(toml_data: dict) -> MCPConfig:
    mcp_raw = toml_data.get("mcp", {})
    servers_raw = mcp_raw.pop("servers", []) if isinstance(mcp_raw, dict) else []
    servers = [MCPServerConfig(**s) for s in servers_raw if isinstance(s, dict)]
    return MCPConfig(
        enabled=mcp_raw.get("enabled", False),
        transport=mcp_raw.get("transport", "stdio"),
        fallback_on_error=mcp_raw.get("fallback_on_error", True),
        call_timeout_s=mcp_raw.get("call_timeout_s", 5.0),
        servers=servers,
    )


def _build_scheduler_config(toml_data: dict) -> SchedulerConfig:
    """Parse ``[scheduler]`` section into ``SchedulerConfig`` with ``list[Pair]``.

    Two TOML forms are accepted (per spec 013-pair-value-object FR-100):

    - **Legacy** ``pairs = ["BTC/USDT", "ETH/USDT"]`` — all spot
    - **New table-array**::

          [[scheduler.pairs]]
          symbol = "BTC/USDT"
          market = "swap"      # optional, defaults to "spot"
          settle = "USDT"      # required when market != "spot"

    Mixed (some str, some dict) raises ``ConfigurationError``.
    """
    raw = dict(toml_data.get("scheduler", {}))
    raw_pairs = raw.pop("pairs", None)

    if raw_pairs is None:
        # No explicit config -> SchedulerConfig dataclass default kicks in
        return SchedulerConfig(**raw)

    if len(raw_pairs) == 0:
        return SchedulerConfig(pairs=[], **raw)

    if all(isinstance(p, str) for p in raw_pairs):
        pairs = [Pair.parse(p) for p in raw_pairs]
    elif all(isinstance(p, dict) for p in raw_pairs):
        pairs = [_parse_pair_dict(p, idx) for idx, p in enumerate(raw_pairs)]
    else:
        raise ConfigurationError(
            field_path="scheduler.pairs",
            expected="all-strings (legacy) or all-tables ([[scheduler.pairs]]); mixing is not allowed",
        )

    seen: set[str] = set()
    for p in pairs:
        c = p.canonical()
        if c in seen:
            raise ConfigurationError(
                field_path="scheduler.pairs",
                expected=f"unique canonical pairs; duplicate pair: {c}",
            )
        seen.add(c)

    return SchedulerConfig(pairs=pairs, **raw)


def _parse_pair_dict(d: dict, idx: int) -> Pair:
    """Parse a single ``[[scheduler.pairs]]`` table entry into a ``Pair``."""
    symbol = d.get("symbol")
    if not symbol or not isinstance(symbol, str):
        raise ConfigurationError(
            field_path=f"scheduler.pairs[{idx}].symbol",
            expected=f"non-empty string (got {symbol!r})",
        )
    market = d.get("market", "spot")
    if market not in ("spot", "swap", "future"):
        raise ConfigurationError(
            field_path=f"scheduler.pairs[{idx}].market",
            expected=f"one of spot/swap/future (got {market!r})",
        )
    settle = d.get("settle")
    if market != "spot":
        if not settle or not isinstance(settle, str):
            raise ConfigurationError(
                field_path=f"scheduler.pairs[{idx}].settle",
                expected=f"required when market={market!r} (got {settle!r})",
            )
        ccxt_symbol = f"{symbol}:{settle}"
    else:
        if settle:
            raise ConfigurationError(
                field_path=f"scheduler.pairs[{idx}].settle",
                expected="must be omitted when market='spot'",
            )
        ccxt_symbol = symbol
    return Pair.parse(ccxt_symbol)


def _build_config(toml_data: dict) -> AppConfig:
    """Build AppConfig from parsed TOML dict."""
    app = toml_data.get("app", {})
    llm_raw = dict(toml_data.get("llm", {}))
    retry_raw = llm_raw.pop("retry", {})
    llm_cfg = LLMConfig(**llm_raw, retry=RetryConfig(**retry_raw))
    risk_raw = toml_data.get("risk", {})

    risk = RiskConfig(
        max_stop_loss_pct=risk_raw.get("max_stop_loss_pct", 0.05),
        position=PositionConfig(**risk_raw.get("position", {})),
        loss=LossConfig(**risk_raw.get("loss", {})),
        cooldown=CooldownConfig(**risk_raw.get("cooldown", {})),
        volatility=VolatilityConfig(**risk_raw.get("volatility", {})),
        exchange=ExchangeCheckConfig(**risk_raw.get("exchange", {})),
        rate_limit=RateLimitConfig(**risk_raw.get("rate_limit", {})),
    )

    backtest_raw = dict(toml_data.get("backtest", {}))  # copy to avoid mutating original
    backtest_ps_raw = backtest_raw.pop("position_sizing", {})
    backtest = BacktestConfig(
        **backtest_raw,
        position_sizing=BacktestPositionSizingConfig(**backtest_ps_raw),
    )

    providers_raw = toml_data.get("providers", {})
    providers = ProvidersConfig(**providers_raw)
    if providers.has_okx_credentials():
        providers.okx_enabled = True

    # Parse exchanges
    exchanges_raw = toml_data.get("exchanges", {})
    exchange_creds = {}
    for ex_id, ex_data in exchanges_raw.items():
        if isinstance(ex_data, dict):
            exchange_creds[ex_id] = ExchangeCredentials(**ex_data)
    exchanges = ExchangesConfig(_exchanges=exchange_creds)

    return AppConfig(
        mode=app.get("mode", "standalone"),
        engine=app.get("engine", "paper"),
        exchange_id=app.get("exchange_id", "binance"),
        llm=llm_cfg,
        models=ModelConfig(**toml_data.get("models", {})),
        debate=DebateConfig(**toml_data.get("debate", {})),
        data=DataConfig(**toml_data.get("data", {})),
        risk=risk,
        backtest=backtest,
        portfolio=PortfolioConfig(**toml_data.get("portfolio", {})),
        experience=_build_experience_config(toml_data),
        scheduler=_build_scheduler_config(toml_data),
        providers=providers,
        execution=ExecutionConfig(**toml_data.get("execution", {})),
        notifications=_build_notifications_config(toml_data),
        triggers=TriggersConfig(**toml_data.get("triggers", {})),
        infrastructure=InfrastructureConfig(**toml_data.get("infrastructure", {})),
        exchanges=exchanges,
        chart_analysis=ChartAnalysisConfig(**toml_data.get("chart_analysis", {})),
        chat=ChatConfig(**toml_data.get("chat", {})),
        hitl=_build_hitl_config(toml_data),
        agents=_build_agents_config(toml_data),
        mcp=_build_mcp_config(toml_data),
        evolution_daemon=EvolutionDaemonConfig(**toml_data.get("evolution_daemon", {})),
    )


_cached_config: AppConfig | None = None


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from TOML file (cached after first call).

    Loads config/default.toml first, then deep-merges config/local.toml
    on top (for secrets like api_key). local.toml is gitignored.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    path = Path(config_path) if config_path else _find_config()
    if path and path.exists():
        with open(path, "rb") as f:
            toml_data = tomllib.load(f)
        # Merge local.toml overrides (same directory, gitignored)
        local_path = path.parent / "local.toml"
        if local_path.exists():
            with open(local_path, "rb") as f:
                local_data = tomllib.load(f)
            _merge(toml_data, local_data)
        # Apply CRYPTOTRADER_* environment variable overrides (highest priority)
        toml_data = apply_env_overrides(toml_data)
        _cached_config = _build_config(toml_data)
    else:
        _cached_config = AppConfig()

    validate_config(_cached_config)
    return _cached_config
