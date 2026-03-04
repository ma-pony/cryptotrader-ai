"""Unified configuration using Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    analysis: str = "gpt-4o-mini"
    debate: str = "gpt-4o-mini"
    verdict: str = "gpt-4o"
    tech_agent: str = "gpt-4o-mini"
    chain_agent: str = "gpt-4o-mini"
    news_agent: str = "gpt-4o-mini"
    macro_agent: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(env_prefix="MODEL_")


class DebateConfig(BaseSettings):
    max_rounds: int = 3
    convergence_threshold: float = 0.1
    divergence_hold_threshold: float = 0.7

    model_config = SettingsConfigDict(env_prefix="DEBATE_")


class DataConfig(BaseSettings):
    default_timeframe: str = "1h"
    ohlcv_limit: int = 100

    model_config = SettingsConfigDict(env_prefix="DATA_")


class PositionConfig(BaseSettings):
    max_single_pct: float = 0.10
    max_total_exposure_pct: float = 0.50

    model_config = SettingsConfigDict(env_prefix="RISK_POSITION_")


class LossConfig(BaseSettings):
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    max_cvar_95: float = 0.05

    model_config = SettingsConfigDict(env_prefix="RISK_LOSS_")


class CooldownConfig(BaseSettings):
    same_pair_minutes: int = 60
    post_loss_minutes: int = 120

    model_config = SettingsConfigDict(env_prefix="RISK_COOLDOWN_")


class VolatilityConfig(BaseSettings):
    flash_crash_threshold: float = 0.05
    funding_rate_threshold: float = 0.001

    model_config = SettingsConfigDict(env_prefix="RISK_VOLATILITY_")


class ExchangeCheckConfig(BaseSettings):
    max_api_latency_ms: int = 2000
    health_check_interval_s: int = 30

    model_config = SettingsConfigDict(env_prefix="RISK_EXCHANGE_")


class RateLimitConfig(BaseSettings):
    max_trades_per_hour: int = 6
    max_trades_per_day: int = 20

    model_config = SettingsConfigDict(env_prefix="RISK_RATE_LIMIT_")


class RiskConfig(BaseSettings):
    position: PositionConfig = Field(default_factory=PositionConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    cooldown: CooldownConfig = Field(default_factory=CooldownConfig)
    volatility: VolatilityConfig = Field(default_factory=VolatilityConfig)
    exchange: ExchangeCheckConfig = Field(default_factory=ExchangeCheckConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


class SchedulerConfig(BaseSettings):
    enabled: bool = False
    pairs: list[str] = ["BTC/USDT", "ETH/USDT"]
    interval_minutes: int = 240
    exchange_id: str = "binance"

    model_config = SettingsConfigDict(env_prefix="SCHEDULER_")


class ProvidersConfig(BaseSettings):
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
    binance_audit_enabled: bool = True
    binance_sentiment_enabled: bool = True
    enforce_token_security: bool = True
    max_acceptable_risk: str = "MEDIUM"

    model_config = SettingsConfigDict(env_prefix="PROVIDER_")

    def has_okx_credentials(self) -> bool:
        return all([self.okx_api_key, self.okx_secret_key, self.okx_passphrase])

    def model_post_init(self, __context) -> None:
        if self.has_okx_credentials():
            self.okx_enabled = True


class NotificationsConfig(BaseSettings):
    webhook_url: str = ""
    enabled: bool = True
    events: list[str] = ["trade", "rejection", "circuit_breaker", "reconcile_mismatch", "daily_summary"]

    model_config = SettingsConfigDict(env_prefix="NOTIFICATION_")


class AppConfig(BaseSettings):
    mode: str = "standalone"
    engine: str = "paper"
    models: ModelConfig = Field(default_factory=ModelConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    return AppConfig()

