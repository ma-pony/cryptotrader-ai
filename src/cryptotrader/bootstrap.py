"""Production dependency assembly for the single TradingCycle runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from cryptotrader.cycle_events import NullCycleEventSink
from cryptotrader.decision.engine import DecisionEngine
from cryptotrader.decision.exit_policy import AtrExitPolicy
from cryptotrader.execution.order import OrderManager
from cryptotrader.execution.planner import ExecutionPlanner
from cryptotrader.execution.service import ExecutionService
from cryptotrader.hitl.store import ApprovalStore
from cryptotrader.journal.store import CycleJournalStore
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
    from cryptotrader.decision.models import CycleRequest
    from cryptotrader.profiles.models import SignalProfile


class SeededProfileRepository:
    """Read the persisted profile, seeding defaults on the first cycle."""

    def __init__(self, database_url: str | None, default: SignalProfile) -> None:
        self.default = default
        self.repository = SignalProfileRepository(database_url) if database_url else None
        self._memory = default

    async def get(self) -> SignalProfile:
        if self.repository is None:
            return self._memory
        return await self.repository.get_or_create(self.default)

    async def replace(self, profile: SignalProfile) -> SignalProfile:
        if self.repository is None:
            self._memory = replace(profile, revision=self._memory.revision + 1)
            return self._memory
        return await self.repository.replace(profile)


class ExchangePortfolioReader:
    """Read portfolio and risk facts from the same exchange used for orders."""

    def __init__(self, exchange, database_url: str | None = None) -> None:
        self.exchange = exchange
        self.database_url = database_url

    async def read(self, request: CycleRequest, current_price: float) -> dict[str, Any]:
        balances = await self.exchange.get_balance()
        try:
            free_balances = await self.exchange.get_free_balance()
        except AttributeError:
            free_balances = balances
        try:
            positions = await self.exchange.get_positions(
                current_prices={request.pair.canonical(): current_price},
            )
        except TypeError:
            positions = await self.exchange.get_positions()

        cash = float(balances.get("USDT", 0.0) or 0.0)
        free_cash = float(free_balances.get("USDT", cash) or 0.0)
        total_value = cash
        for pair, position in positions.items():
            amount = float(position.get("amount", 0.0) or 0.0)
            if ":" in pair:
                total_value += float(position.get("unrealized_pnl", 0.0) or 0.0)
            else:
                mark = float(position.get("avg_price", 0.0) or 0.0)
                if pair == request.pair.canonical():
                    mark = current_price
                total_value += amount * mark
        portfolio: dict[str, Any] = {
            "cash": cash,
            "free_cash": free_cash,
            "positions": positions,
            "total_value": total_value,
        }
        if self.database_url:
            from cryptotrader.portfolio.manager import PortfolioManager

            manager = PortfolioManager(self.database_url)
            portfolio["daily_pnl"] = await manager.get_daily_pnl()
            portfolio["drawdown"] = await manager.get_drawdown()
        return portfolio


def _build_exchange(config: AppConfig, mode: str):
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


def build_signal_registry(config: AppConfig, events: CycleEventSink) -> SignalComponentRegistry:
    registry = SignalComponentRegistry()

    from cryptotrader.signals.components.kronos import KronosComponent
    from cryptotrader.signals.components.llm_committee import LLMCommitteeComponent

    registry.register(KronosComponent(config.kronos))
    registry.register(LLMCommitteeComponent(config, sink=events))
    for factory in config.signal_plugins.factories:
        registry.load_factory(factory)
    return registry


def build_trading_cycle(
    config: AppConfig,
    mode: str,
    event_sink: CycleEventSink | None = None,
) -> TradingCycle:
    if mode == "backtest":
        raise ValueError("BacktestEngine must provide historical context and executor")
    events = event_sink or NullCycleEventSink()
    registry = build_signal_registry(config, events)

    default_profile = config.signal_profile_defaults.to_profile()
    exchange = _build_exchange(config, mode)
    portfolio = ExchangePortfolioReader(exchange, config.infrastructure.database_url or None)

    from cryptotrader.data.snapshot import SnapshotAggregator

    aggregator = SnapshotAggregator(config.providers)
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
    return TradingCycle(
        profiles=SeededProfileRepository(database_url, default_profile),
        registry=registry,
        contexts=contexts,
        runner=ComponentRunner(events),
        fusion=WeightedSignalFusion(),
        decisions=DecisionEngine(),
        exits=AtrExitPolicy(),
        approvals=ApprovalStore(database_url),
        risk=RiskGate(config.risk, redis_state, leverage=leverage),
        execution_planner=ExecutionPlanner(config.risk.position.max_single_pct),
        executor=ExecutionService(OrderManager(), exchange, manage_protection=True),
        journal=CycleJournalStore(database_url),
        events=events,
        exit_requirement=DataRequirements(
            candles=(CandleRequirement(config.data.default_timeframe, max(20, config.data.ohlcv_limit)),),
        ),
    )
