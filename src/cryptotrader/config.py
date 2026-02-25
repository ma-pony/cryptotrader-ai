"""Load and validate TOML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli
from pydantic import BaseModel

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


class ModelConfig(BaseModel):
    analysis: str = "gpt-4o-mini"
    debate: str = "gpt-4o-mini"
    verdict: str = "gpt-4o"
    agents: dict[str, str] = {}


class DebateConfig(BaseModel):
    max_rounds: int = 3
    convergence_threshold: float = 0.1
    divergence_hold_threshold: float = 0.7


class DataConfig(BaseModel):
    default_timeframe: str = "1h"
    ohlcv_limit: int = 100


class PositionConfig(BaseModel):
    max_single_pct: float = 0.10
    max_total_exposure_pct: float = 0.50


class LossConfig(BaseModel):
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    max_cvar_95: float = 0.05


class CooldownConfig(BaseModel):
    same_pair_minutes: int = 60
    post_loss_minutes: int = 120


class VolatilityConfig(BaseModel):
    flash_crash_threshold: float = 0.05
    funding_rate_threshold: float = 0.001


class ExchangeCheckConfig(BaseModel):
    max_api_latency_ms: int = 2000
    health_check_interval_s: int = 30


class RateLimitConfig(BaseModel):
    max_trades_per_hour: int = 6
    max_trades_per_day: int = 20


class RiskConfig(BaseModel):
    position: PositionConfig = PositionConfig()
    loss: LossConfig = LossConfig()
    cooldown: CooldownConfig = CooldownConfig()
    volatility: VolatilityConfig = VolatilityConfig()
    exchange: ExchangeCheckConfig = ExchangeCheckConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()


class SchedulerConfig(BaseModel):
    enabled: bool = False
    pairs: list[str] = ["BTC/USDT", "ETH/USDT"]
    interval_minutes: int = 240
    exchange_id: str = "binance"


class ProvidersConfig(BaseModel):
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


class NotificationsConfig(BaseModel):
    webhook_url: str = ""
    enabled: bool = True
    events: list[str] = [
        "trade", "rejection", "circuit_breaker",
        "reconcile_mismatch", "daily_summary",
    ]


class AppConfig(BaseModel):
    mode: str = "standalone"
    engine: str = "paper"
    models: ModelConfig = ModelConfig()
    debate: DebateConfig = DebateConfig()
    data: DataConfig = DataConfig()
    risk: RiskConfig = RiskConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    providers: ProvidersConfig = ProvidersConfig()
    notifications: NotificationsConfig = NotificationsConfig()


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomli.load(f)


def load_config(config_dir: Path | None = None) -> AppConfig:
    d = config_dir or CONFIG_DIR
    raw: dict[str, Any] = {}

    default_path = d / "default.toml"
    if default_path.exists():
        data = _load_toml(default_path)
        raw["mode"] = data.get("mode", {}).get("mode", "standalone")
        raw["engine"] = data.get("execution", {}).get("engine", "paper")
        raw["models"] = data.get("models", {})
        raw["debate"] = data.get("debate", {})
        raw["data"] = data.get("data", {})
        raw["scheduler"] = data.get("scheduler", {})
        raw["providers"] = data.get("providers", {})
        raw["notifications"] = data.get("notifications", {})

    risk_path = d / "risk.toml"
    if risk_path.exists():
        raw["risk"] = _load_toml(risk_path)

    return AppConfig(**raw)
