"""Unified configuration — loaded from config/default.toml."""

from __future__ import annotations

import copy
import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ── LLM gateway ──


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = ""
    streaming_models: list[str] = field(default_factory=list)
    default_temperature: float = 0.2
    timeout: int = 120


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
    timeout_seconds: int = 60


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
    max_single_pct: float = 0.10
    max_total_exposure_pct: float = 0.50
    max_correlated_positions: int = 2


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
    regime_thresholds: RegimeThresholdsConfig = field(default_factory=RegimeThresholdsConfig)


# ── Reflection (deprecated alias) ──

ReflectionConfig = ExperienceConfig


# ── Execution ──


@dataclass
class ExecutionConfig:
    order_wait_seconds: int = 30
    retry_attempts: int = 3
    graph_timeout_s: int = 300


# ── Scheduler ──


@dataclass
class SchedulerConfig:
    enabled: bool = False
    pairs: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
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
class NotificationsConfig:
    webhook_url: str = ""
    enabled: bool = True
    webhook_timeout: int = 5
    events: list[str] = field(
        default_factory=lambda: ["trade", "rejection", "circuit_breaker", "reconcile_mismatch", "daily_summary"]
    )


# ── Exchanges ──


@dataclass
class ExchangeCredentials:
    api_key: str = ""
    secret: str = ""
    passphrase: str = ""
    sandbox: bool = True


@dataclass
class ExchangesConfig:
    _exchanges: dict[str, ExchangeCredentials] = field(default_factory=dict)

    def get(self, exchange_id: str) -> ExchangeCredentials | None:
        return self._exchanges.get(exchange_id)

    def __iter__(self):
        return iter(self._exchanges)

    def items(self):
        return self._exchanges.items()


# ── Infrastructure ──


@dataclass
class InfrastructureConfig:
    database_url: str = ""
    redis_url: str = ""


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
    experience: ExperienceConfig = field(default_factory=ExperienceConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    exchanges: ExchangesConfig = field(default_factory=ExchangesConfig)

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
    if not cfg.models.fallback.strip():
        raise ConfigurationError(
            field_path="models.fallback",
            expected="non-empty string (model name must be specified)",
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
    """Build ExperienceConfig from TOML, supporting both [experience] and legacy [reflection]."""
    raw = dict(toml_data.get("experience", toml_data.get("reflection", {})))
    regime_raw = raw.pop("regime_thresholds", {})
    return ExperienceConfig(
        **raw,
        regime_thresholds=RegimeThresholdsConfig(**regime_raw),
    )


def _build_config(toml_data: dict) -> AppConfig:
    """Build AppConfig from parsed TOML dict."""
    app = toml_data.get("app", {})
    llm = toml_data.get("llm", {})
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
        llm=LLMConfig(**llm),
        models=ModelConfig(**toml_data.get("models", {})),
        debate=DebateConfig(**toml_data.get("debate", {})),
        data=DataConfig(**toml_data.get("data", {})),
        risk=risk,
        backtest=backtest,
        experience=_build_experience_config(toml_data),
        scheduler=SchedulerConfig(**toml_data.get("scheduler", {})),
        providers=providers,
        execution=ExecutionConfig(**toml_data.get("execution", {})),
        notifications=NotificationsConfig(**toml_data.get("notifications", {})),
        infrastructure=InfrastructureConfig(**toml_data.get("infrastructure", {})),
        exchanges=exchanges,
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
