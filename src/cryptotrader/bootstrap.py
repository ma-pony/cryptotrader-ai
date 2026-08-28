"""Production dependency assembly for the single TradingCycle runtime."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cryptotrader.cycle_events import NullCycleEventSink
from cryptotrader.decision.engine import DecisionEngine
from cryptotrader.decision.exit_policy import AtrExitPolicy
from cryptotrader.execution.order import OrderManager
from cryptotrader.execution.planner import ExecutionPlanner
from cryptotrader.execution.service import ExecutionService
from cryptotrader.hitl.store import ApprovalStore
from cryptotrader.journal.store import CycleJournalStore
from cryptotrader.portfolio.exchange_reader import ExchangePortfolioReader
from cryptotrader.profiles.models import validate_signal_profile
from cryptotrader.profiles.repository import SignalProfileRepository
from cryptotrader.risk.gate import RiskGate
from cryptotrader.risk.state import RedisStateManager
from cryptotrader.signals.context import LiveSignalContextProvider
from cryptotrader.signals.fusion import WeightedSignalFusion
from cryptotrader.signals.models import CandleRequirement, DataRequirements
from cryptotrader.signals.registry import SignalComponentRegistry
from cryptotrader.signals.runner import ComponentRunner
from cryptotrader.trading_cycle import TradingCycle

if TYPE_CHECKING:
    from cryptotrader.config import AppConfig
    from cryptotrader.cycle_events import CycleEventSink
    from cryptotrader.profiles.models import SignalProfile
    from cryptotrader.signals.component import SignalComponent
    from cryptotrader.signals.models import TradingMode


class SeededProfileRepository:
    """Read the persisted profile, seeding defaults on the first cycle."""

    def __init__(self, database_url: str | None, default: SignalProfile) -> None:
        self.default = default
        self.repository = SignalProfileRepository(database_url) if database_url else None
        self._memory = default if default.updated_at is not None else replace(default, updated_at=datetime.now(UTC))

    async def get(self) -> SignalProfile:
        if self.repository is None:
            return self._memory
        return await self.repository.get_or_create(self.default)

    async def replace(self, profile: SignalProfile) -> SignalProfile:
        if self.repository is None:
            self._memory = replace(
                profile,
                revision=self._memory.revision + 1,
                updated_at=datetime.now(UTC),
            )
            return self._memory
        return await self.repository.replace(profile)


async def initialize_trading_cycle(cycle: TradingCycle) -> TradingCycle:
    """Validate the active profile once at the shared runtime boundary."""
    if getattr(cycle, "_startup_validated", False):
        return cycle
    active_profile = await cycle.profiles.get()
    if active_profile is None:
        raise RuntimeError("global signal profile is not initialized")
    validate_signal_profile(active_profile, cycle.registry.ids())
    cycle._startup_validated = True
    return cycle


def _build_exchange(config: AppConfig, mode: TradingMode):
    if mode == "paper":
        from cryptotrader.execution.simulator import PaperExchange

        return PaperExchange(initial_balances={"USDT": config.backtest.initial_capital})
    if mode != "live":
        raise ValueError("BacktestEngine owns historical TradingCycle assembly")

    from cryptotrader.execution.exchange import LiveExchange

    exchange_id = config.scheduler.exchange_id or config.exchange_id
    credentials = config.exchanges.get(exchange_id)
    if credentials is None or not credentials.api_key or not credentials.secret:
        raise RuntimeError(f"No credentials configured for exchange {exchange_id!r}")
    return LiveExchange(
        exchange_id,
        credentials.api_key,
        credentials.secret,
        sandbox=credentials.sandbox,
        passphrase=credentials.passphrase,
        leverage=credentials.leverage,
        margin_mode=credentials.margin_mode,
    )


def build_signal_registry(
    config: AppConfig,
    events: CycleEventSink,
    *,
    custom_components: tuple[SignalComponent, ...] | None = None,
) -> SignalComponentRegistry:
    registry = SignalComponentRegistry()

    from cryptotrader.signals.components.kronos import KronosComponent
    from cryptotrader.signals.components.llm_committee import LLMCommitteeComponent

    registry.register(KronosComponent(config.kronos))
    registry.register(LLMCommitteeComponent(config, sink=events))
    if custom_components is None:
        for factory in config.signal_plugins.factories:
            registry.load_factory(factory)
    else:
        for component in custom_components:
            registry.register(component)
    return registry


def build_trading_cycle(
    config: AppConfig,
    mode: TradingMode,
    event_sink: CycleEventSink | None = None,
    *,
    profile_repository=None,
    approval_store=None,
    journal_store=None,
    custom_components: tuple[SignalComponent, ...] | None = None,
) -> TradingCycle:
    if mode == "backtest":
        raise ValueError("BacktestEngine must provide historical context and executor")
    events = event_sink or NullCycleEventSink()
    registry = build_signal_registry(config, events, custom_components=custom_components)

    default_profile = config.signal_profile_defaults.to_profile()
    validate_signal_profile(default_profile, registry.ids())
    exchange = _build_exchange(config, mode)

    from cryptotrader.data.snapshot import SnapshotAggregator

    aggregator = SnapshotAggregator(config.providers)
    portfolio = ExchangePortfolioReader(
        exchange,
        config.infrastructure.database_url or None,
        ticker_source=aggregator.market,
    )
    contexts = LiveSignalContextProvider(
        aggregator,
        aggregator.market,
        portfolio,
        default_timeframe=config.data.default_timeframe,
        max_single_pct=config.risk.position.max_single_pct,
        kronos_aux_symbol=config.kronos.aux_symbol,
    )
    redis_state = RedisStateManager(config.infrastructure.redis_url or None)
    credentials = config.exchanges.get(config.scheduler.exchange_id or config.exchange_id)
    leverage = credentials.leverage if credentials is not None else 1
    database_url = config.infrastructure.database_url or None
    profiles = (
        profile_repository if profile_repository is not None else SeededProfileRepository(database_url, default_profile)
    )
    approvals = approval_store if approval_store is not None else ApprovalStore(database_url)
    return TradingCycle(
        mode=mode,
        profiles=profiles,
        registry=registry,
        contexts=contexts,
        runner=ComponentRunner(events),
        fusion=WeightedSignalFusion(),
        decisions=DecisionEngine(),
        exits=AtrExitPolicy(),
        approvals=approvals,
        risk=RiskGate(config.risk, redis_state, leverage=leverage),
        execution_planner=ExecutionPlanner(config.risk.position.max_single_pct),
        executor=ExecutionService(OrderManager(), exchange, manage_protection=True),
        journal=journal_store if journal_store is not None else CycleJournalStore(database_url),
        events=events,
        exit_requirement=DataRequirements(
            candles=(CandleRequirement(config.data.default_timeframe, max(20, config.data.ohlcv_limit)),),
        ),
    )
