"""Unified configuration — loaded from config/default.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# ── LLM gateway ──


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = ""
    streaming_models: list[str] = field(default_factory=list)


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


# ── Debate ──


@dataclass
class DebateConfig:
    max_rounds: int = 3
    convergence_threshold: float = 0.1
    divergence_hold_threshold: float = 0.7


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


@dataclass
class LossConfig:
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    max_cvar_95: float = 0.05


@dataclass
class CooldownConfig:
    same_pair_minutes: int = 60
    post_loss_minutes: int = 120


@dataclass
class VolatilityConfig:
    flash_crash_threshold: float = 0.05
    funding_rate_threshold: float = 0.005


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
    sma_fast: int = 20
    sma_slow: int = 50
    position_sizing: BacktestPositionSizingConfig = field(default_factory=BacktestPositionSizingConfig)


# ── Reflection ──


@dataclass
class ReflectionConfig:
    enabled: bool = True
    every_n_cycles: int = 20
    min_commits_required: int = 10
    lookback_commits: int = 30
    model: str = ""  # empty = use models.analysis


# ── Scheduler ──


@dataclass
class SchedulerConfig:
    enabled: bool = False
    pairs: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    interval_minutes: int = 240
    exchange_id: str = "binance"


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

    def has_okx_credentials(self) -> bool:
        return all([self.okx_api_key, self.okx_secret_key, self.okx_passphrase])


# ── Notifications ──


@dataclass
class NotificationsConfig:
    webhook_url: str = ""
    enabled: bool = True
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
    llm: LLMConfig = field(default_factory=LLMConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    exchanges: ExchangesConfig = field(default_factory=ExchangesConfig)


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
        llm=LLMConfig(**llm),
        models=ModelConfig(**toml_data.get("models", {})),
        debate=DebateConfig(**toml_data.get("debate", {})),
        data=DataConfig(**toml_data.get("data", {})),
        risk=risk,
        backtest=backtest,
        reflection=ReflectionConfig(**toml_data.get("reflection", {})),
        scheduler=SchedulerConfig(**toml_data.get("scheduler", {})),
        providers=providers,
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
        _cached_config = _build_config(toml_data)
    else:
        _cached_config = AppConfig()

    return _cached_config
