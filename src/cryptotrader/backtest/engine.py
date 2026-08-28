"""Historical execution of the same TradingCycle used by paper and live modes."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd

from cryptotrader._compat import UTC
from cryptotrader.backtest.cache import _TF_MS, fetch_historical
from cryptotrader.backtest.result import BacktestResult
from cryptotrader.decision.models import CycleRequest
from cryptotrader.execution.service import ExecutionOrderResult, ExecutionResult
from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData
from cryptotrader.pair import Pair
from cryptotrader.signals.models import CandleRequirement, DataRequirements, PositionSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from cryptotrader.decision.models import CycleOutcome, ExecutionPlan
    from cryptotrader.journal.models import TradingCycleRecord
    from cryptotrader.profiles.models import SignalProfile
    from cryptotrader.signals.component import SignalComponent
    from cryptotrader.signals.context import HistoricalSignalContextProvider

logger = logging.getLogger(__name__)


class FrozenProfileRepository:
    """A point-in-time profile view owned by one backtest run."""

    def __init__(self, profile: SignalProfile) -> None:
        self.profile = replace(profile, hitl_required=False)

    async def get(self) -> SignalProfile:
        return self.profile


class BacktestExecutor:
    """Queue plans at signal close and fill them at the following bar open."""

    def __init__(self, *, initial_capital: float, slippage_bps: float, fee_bps: float) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps
        self._signed_amount = 0.0
        self._entry_price = 0.0
        self._pending: ExecutionPlan | None = None
        self._pair = ""
        self.protection: tuple[float, float] | None = None
        self.trades: list[dict[str, Any]] = []

    @property
    def position(self) -> PositionSnapshot:
        if abs(self._signed_amount) < 1e-12:
            return PositionSnapshot("flat", 0.0, 0.0)
        return PositionSnapshot(
            "long" if self._signed_amount > 0.0 else "short",
            abs(self._signed_amount),
            0.0,
            self._entry_price,
            0.0,
        )

    def position_at(self, price: float) -> PositionSnapshot:
        position = self.position
        if position.side == "flat":
            return position
        direction = 1.0 if position.side == "long" else -1.0
        unrealized = (price - self._entry_price) * position.amount * direction
        return replace(position, unrealized_pnl=unrealized)

    def equity_at(self, price: float) -> float:
        return self.cash + self.position_at(price).unrealized_pnl

    async def execute(self, plan: ExecutionPlan, _context) -> ExecutionResult:
        if self._pending is not None:
            return ExecutionResult(False, (), None, "a backtest execution is already pending")
        self._pending = plan
        if plan.intents:
            self._pair = plan.intents[-1].pair
        orders = tuple(
            ExecutionOrderResult(
                intent=intent,
                status="scheduled",
                exchange_id=None,
                raw={"execution": "next_bar_open"},
            )
            for intent in plan.intents
        )
        return ExecutionResult(True, orders, None, None)

    def execute_pending_at(self, bar: list) -> None:
        if self._pending is None:
            return
        plan = self._pending
        self._pending = None
        open_price = float(bar[1] or bar[4])
        for intent in plan.intents:
            self._fill(intent.side, intent.amount, open_price, int(bar[0]), reason="target_position")
        if abs(self._signed_amount) < 1e-12:
            self.protection = None
        elif plan.stop_loss is not None and plan.take_profit is not None:
            self.protection = (plan.stop_loss, plan.take_profit)
        else:
            raise ValueError("non-flat backtest position requires stop loss and take profit")

    def process_protection(self, bar: list) -> None:
        if self.protection is None or abs(self._signed_amount) < 1e-12:
            return
        stop_loss, take_profit = self.protection
        high, low = float(bar[2]), float(bar[3])
        trigger: float | None = None
        reason = ""
        if self._signed_amount > 0.0:
            if low <= stop_loss:
                trigger, reason = stop_loss, "stop_loss"
            elif high >= take_profit:
                trigger, reason = take_profit, "take_profit"
        else:
            if high >= stop_loss:
                trigger, reason = stop_loss, "stop_loss"
            elif low <= take_profit:
                trigger, reason = take_profit, "take_profit"
        if trigger is None:
            return
        self._fill(
            "sell" if self._signed_amount > 0.0 else "buy",
            abs(self._signed_amount),
            trigger,
            int(bar[0]),
            reason=reason,
        )
        self.protection = None

    def close_at(self, price: float, ts: int) -> None:
        self._pending = None
        if abs(self._signed_amount) < 1e-12:
            return
        self._fill(
            "sell" if self._signed_amount > 0.0 else "buy",
            abs(self._signed_amount),
            price,
            ts,
            reason="backtest_end",
        )
        self.protection = None

    def _fill(self, side: str, amount: float, price: float, ts: int, *, reason: str) -> None:
        slippage = self.slippage_bps / 10_000
        fill_price = price * (1.0 + slippage if side == "buy" else 1.0 - slippage)
        fee = amount * fill_price * self.fee_bps / 10_000
        before = self._signed_amount
        delta = amount if side == "buy" else -amount
        after = before + delta
        closed_amount = min(abs(before), abs(delta)) if before * delta < 0.0 else 0.0
        pnl = 0.0
        if closed_amount > 0.0:
            direction = 1.0 if before > 0.0 else -1.0
            pnl = (fill_price - self._entry_price) * closed_amount * direction
        self.cash += pnl - fee

        if abs(after) < 1e-12:
            self._entry_price = 0.0
            after = 0.0
        elif before == 0.0 or before * after <= 0.0:
            self._entry_price = fill_price
        elif before * delta > 0.0:
            self._entry_price = (self._entry_price * abs(before) + fill_price * abs(delta)) / abs(after)
        self._signed_amount = after
        self.trades.append(
            {
                "pair": self._pair,
                "side": side,
                "amount": amount,
                "price": fill_price,
                "fee": fee,
                "pnl": pnl,
                "reason": reason,
                "ts": ts,
            }
        )


class BacktestEngine:
    def __init__(
        self,
        pair: str,
        start: str,
        end: str,
        interval: str = "4h",
        initial_capital: float | None = None,
        slippage_bps: float | None = None,
        fee_bps: float | None = None,
        lookback: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
        *,
        profile_repository=None,
        cycle_factory: Callable | None = None,
        custom_components: tuple[SignalComponent, ...] | None = None,
        journal_store=None,
        config=None,
    ) -> None:
        from cryptotrader.bootstrap import SeededProfileRepository
        from cryptotrader.config import load_config

        self.config = config or load_config()
        backtest = self.config.backtest
        self.pair = Pair.parse(pair)
        self.start = start
        self.end = end
        self.start_ms = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1000)
        self.end_ms = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1000)
        self.interval = interval
        self.capital = initial_capital if initial_capital is not None else backtest.initial_capital
        self.slippage_bps = slippage_bps if slippage_bps is not None else backtest.slippage_base * 10_000
        self.fee_bps = fee_bps if fee_bps is not None else backtest.fee_bps
        self.lookback = lookback if lookback is not None else backtest.lookback
        self.progress_callback = progress_callback
        self.cycle_factory = cycle_factory
        self.custom_components = custom_components
        self.journal_store = journal_store
        database_url = self.config.infrastructure.database_url or None
        self.profile_repository = profile_repository or SeededProfileRepository(
            database_url,
            self.config.signal_profile_defaults.to_profile(),
        )
        self.first_bar_processed = asyncio.Event()
        self._candles_by_timeframe: dict[str, list[list]] = {}
        self._candles: list[list] = []
        self._fng: dict[str, int] = {}
        self._funding: dict[str, float] = {}
        self._btc_dom: dict[str, float] = {}
        self._fed_rate: dict[str, float] = {}
        self._dxy: dict[str, float] = {}
        self._fut_vol: dict[str, dict] = {}
        self._etf_flows: dict[str, dict] = {}
        self._stablecoin_supply: dict[str, float] = {}
        self._btc_hashrate: dict[str, float] = {}
        self._defi_tvl: dict[str, float] = {}
        self._vix: dict[str, float] = {}
        self._sp500: dict[str, float] = {}
        self._oi: dict[str, dict] = {}
        self._ls_ratio: dict[str, dict] = {}

    async def run(self) -> BacktestResult:
        selected = await self.profile_repository.get()
        if selected is None:
            raise RuntimeError("global signal profile is not initialized")
        frozen_profiles = FrozenProfileRepository(selected)
        registry, requirements = self._dependencies(frozen_profiles.profile)
        await self._fetch_historical_data(requirements)
        if not self._candles:
            return BacktestResult()

        from cryptotrader.journal.store import CycleJournalStore
        from cryptotrader.signals.context import HistoricalSignalContextProvider

        executor = BacktestExecutor(
            initial_capital=self.capital,
            slippage_bps=self.slippage_bps,
            fee_bps=self.fee_bps,
        )
        contexts = HistoricalSignalContextProvider(
            self._snapshot_at,
            default_timeframe=self.config.data.default_timeframe,
            equity=self.capital,
            max_single_pct=self.config.risk.position.max_single_pct,
        )
        journal = self.journal_store if self.journal_store is not None else CycleJournalStore()
        if self.cycle_factory is not None:
            cycle = self.cycle_factory(frozen_profiles, contexts, executor, journal)
        else:
            cycle = self._build_cycle(frozen_profiles, contexts, executor, journal, registry)
        return await self._run_bars(cycle, contexts, executor, journal)

    def _dependencies(self, profile: SignalProfile):
        if self.cycle_factory is not None:
            requirements = DataRequirements(
                candles=(CandleRequirement(self.interval, max(20, self.lookback)),),
            )
            return None, requirements

        from cryptotrader.bootstrap import build_signal_registry
        from cryptotrader.cycle_events import NullCycleEventSink
        from cryptotrader.profiles.models import validate_signal_profile

        registry = build_signal_registry(
            self.config,
            NullCycleEventSink(),
            custom_components=self.custom_components,
        )
        validate_signal_profile(profile, registry.ids())
        components = registry.enabled(profile)
        exit_requirement = DataRequirements(
            candles=(
                CandleRequirement(
                    self.config.data.default_timeframe,
                    max(20, self.config.data.ohlcv_limit),
                ),
            ),
        )
        requirements = DataRequirements.merge(
            *(component.requirements() for component in components),
            exit_requirement,
            DataRequirements(candles=(CandleRequirement(self.interval, self.lookback),)),
        )
        return registry, requirements

    def _build_cycle(
        self,
        profiles: FrozenProfileRepository,
        contexts: HistoricalSignalContextProvider,
        executor: BacktestExecutor,
        journal,
        registry,
    ):
        from cryptotrader.cycle_events import NullCycleEventSink
        from cryptotrader.decision.engine import DecisionEngine
        from cryptotrader.decision.exit_policy import AtrExitPolicy
        from cryptotrader.execution.planner import ExecutionPlanner
        from cryptotrader.hitl.store import ApprovalStore
        from cryptotrader.risk.gate import RiskGate
        from cryptotrader.risk.state import RedisStateManager
        from cryptotrader.signals.fusion import WeightedSignalFusion
        from cryptotrader.signals.runner import ComponentRunner
        from cryptotrader.trading_cycle import TradingCycle

        events = NullCycleEventSink()
        credentials = self.config.exchanges.get(self.config.scheduler.exchange_id or self.config.exchange_id)
        leverage = credentials.leverage if credentials is not None else 1
        return TradingCycle(
            mode="backtest",
            profiles=profiles,
            registry=registry,
            contexts=contexts,
            runner=ComponentRunner(events),
            fusion=WeightedSignalFusion(),
            decisions=DecisionEngine(),
            exits=AtrExitPolicy(),
            approvals=ApprovalStore(),
            risk=RiskGate(self.config.risk, RedisStateManager(None), leverage=leverage),
            execution_planner=ExecutionPlanner(self.config.risk.position.max_single_pct),
            executor=executor,
            journal=journal,
            events=events,
            exit_requirement=DataRequirements(
                candles=(
                    CandleRequirement(
                        self.config.data.default_timeframe,
                        max(20, self.config.data.ohlcv_limit),
                    ),
                ),
            ),
        )

    async def _run_bars(self, cycle, contexts, executor, journal) -> BacktestResult:
        interval_ms = _TF_MS.get(self.interval)
        if interval_ms is None:
            raise ValueError(f"unsupported backtest timeframe {self.interval!r}")
        indexes = [
            index for index, candle in enumerate(self._candles) if self.start_ms <= int(candle[0]) <= self.end_ms
        ]
        if len(indexes) < 2:
            return BacktestResult(equity_curve=[self.capital])

        outcomes: list[CycleOutcome] = []
        curve = [self.capital]
        peak = self.capital
        for step, index in enumerate(indexes[:-1]):
            candle = self._candles[index]
            as_of = datetime.fromtimestamp((int(candle[0]) + interval_ms) / 1000, UTC)
            outcome = await cycle.run(
                CycleRequest(
                    pair=self.pair,
                    mode="backtest",
                    exchange_id=self.config.scheduler.exchange_id or self.config.exchange_id,
                    as_of=as_of,
                )
            )
            outcomes.append(outcome)
            if step == 0:
                self.first_bar_processed.set()

            next_bar = self._candles[indexes[step + 1]]
            executor.execute_pending_at(next_bar)
            executor.process_protection(next_bar)
            close = float(next_bar[4])
            equity = executor.equity_at(close)
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak if peak > 0.0 else 0.0
            contexts.set_execution_state(
                equity=equity,
                cash=executor.cash,
                current_position=executor.position_at(close),
                daily_pnl=equity - self.capital,
                drawdown=drawdown,
            )
            curve.append(equity)
            if self.progress_callback is not None:
                self.progress_callback((step + 1) / len(indexes[:-1]))

        last = self._candles[indexes[-1]]
        executor.close_at(float(last[4]), int(last[0]))
        final_equity = executor.equity_at(float(last[4]))
        curve[-1] = final_equity
        records = []
        for outcome in outcomes:
            record = await journal.get(outcome.cycle_id)
            if record is None:
                raise RuntimeError(f"backtest cycle {outcome.cycle_id!r} is missing from the journal")
            records.append(record)
        return self._compute_result(
            final_equity,
            curve,
            executor.trades,
            records=records,
            outcomes=outcomes,
        )

    async def _fetch_historical_data(self, requirements: DataRequirements) -> None:
        limits = {item.timeframe: item.limit for item in requirements.candles}
        limits[self.interval] = max(limits.get(self.interval, 0), self.lookback)
        for timeframe, limit in limits.items():
            timeframe_ms = _TF_MS.get(timeframe)
            if timeframe_ms is None:
                raise ValueError(f"unsupported backtest timeframe {timeframe!r}")
            self._candles_by_timeframe[timeframe] = await fetch_historical(
                self.pair.canonical(),
                timeframe,
                self.start_ms - limit * timeframe_ms,
                self.end_ms,
            )
        self._candles = self._candles_by_timeframe[self.interval]

        from cryptotrader.backtest.historical_data import (
            fetch_btc_dominance,
            fetch_fear_greed,
            fetch_fred_series,
            fetch_funding_rate,
            fetch_futures_volume,
        )

        symbol = self.pair.base
        start = datetime.fromisoformat(self.start).replace(tzinfo=UTC)
        daily_start = (start - timedelta(days=1)).strftime("%Y-%m-%d")
        self._fng = await fetch_fear_greed(daily_start, self.end)
        self._funding = await fetch_funding_rate(symbol, daily_start, self.end)
        for attribute, loader in (
            ("_btc_dom", lambda: fetch_btc_dominance(daily_start, self.end)),
            ("_fed_rate", lambda: fetch_fred_series("DFF", daily_start, self.end)),
            ("_dxy", lambda: fetch_fred_series("DTWEXBGS", daily_start, self.end)),
            ("_fut_vol", lambda: fetch_futures_volume(symbol, daily_start, self.end)),
        ):
            try:
                setattr(self, attribute, await loader())
            except Exception:
                logger.warning("Historical source %s failed", attribute, exc_info=True)
        self._load_extended_data(daily_start)

    @staticmethod
    def _extract_numeric(data, key: str | None = None) -> float:
        if isinstance(data, dict):
            return float(data.get(key, 0.0)) if key else 0.0
        if isinstance(data, int | float):
            return float(data)
        return 0.0

    @staticmethod
    def _load_dict_range(source: str, start: str, end: str) -> dict:
        from cryptotrader.data.store import get_range

        return {date: value for date, value in get_range(source, start, end).items() if isinstance(value, dict)}

    def _load_extended_data(self, historical_start: str) -> None:
        from cryptotrader.data.store import get_range

        symbol = self.pair.base
        self._etf_flows = self._load_dict_range("sosovalue_etf", historical_start, self.end)
        self._oi = self._load_dict_range(f"binance_oi_{symbol}", historical_start, self.end)
        self._ls_ratio = self._load_dict_range(f"binance_ls_ratio_{symbol}", historical_start, self.end)
        for date, value in get_range("stablecoin_total_supply", historical_start, self.end).items():
            self._stablecoin_supply[date] = self._extract_numeric(value, "total_supply")
        for date, value in get_range("defillama_tvl", historical_start, self.end).items():
            self._defi_tvl[date] = self._extract_numeric(value, "tvl")
        for source, target in (
            ("btc_hashrate", self._btc_hashrate),
            ("fred_VIXCLS", self._vix),
            ("fred_SP500", self._sp500),
        ):
            for date, value in get_range(source, historical_start, self.end).items():
                target[date] = self._extract_numeric(value)

    def _snapshot_at(self, timeframe: str, as_of: datetime) -> DataSnapshot:
        timestamp_ms = int(as_of.timestamp() * 1000)
        interval_ms = _TF_MS.get(timeframe)
        if interval_ms is None:
            raise ValueError(f"unsupported backtest timeframe {timeframe!r}")
        candles = [item for item in self._candles_by_timeframe[timeframe] if int(item[0]) + interval_ms <= timestamp_ms]
        if not candles:
            raise ValueError(f"no {timeframe} candles available at {as_of.isoformat()}")
        frame = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        frame.index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        current = candles[-1]
        completed_day = (as_of - timedelta(days=1)).strftime("%Y-%m-%d")
        futures = self._fut_vol.get(completed_day, {})
        past_volumes = [
            float(value.get("volume", 0.0))
            for date, value in sorted(self._fut_vol.items())
            if date < completed_day and value.get("volume", 0.0) > 0.0
        ][-20:]
        average_volume = (
            sum(past_volumes) / len(past_volumes) if past_volumes else max(float(futures.get("volume", 0.0)), 1.0)
        )
        futures_volume = float(futures.get("volume", 0.0))
        oi = self._oi.get(completed_day, {})
        long_short = self._ls_ratio.get(completed_day, {})
        etf = self._etf_flows.get(completed_day, {})

        from cryptotrader.backtest.historical_data import derive_news_events

        volatility = frame["close"].pct_change().std()
        onchain = OnchainData(
            open_interest=float(oi.get("openInterestValue", 0.0)) if oi else 0.0,
            liquidations_24h={
                "volume_ratio": futures_volume / average_volume if average_volume else 1.0,
                "futures_volume": futures_volume,
                "long_short_ratio": float(long_short.get("longShortRatio", 1.0)) if long_short else 1.0,
            },
            defi_tvl=self._defi_tvl.get(completed_day, 0.0),
            data_quality={
                "has_oi": bool(oi),
                "has_ls_ratio": bool(long_short),
                "has_etf": bool(etf),
            },
        )
        macro = MacroData(
            fear_greed_index=self._fng.get(completed_day, 50),
            btc_dominance=self._btc_dom.get(completed_day, 0.0),
            fed_rate=self._fed_rate.get(completed_day, 0.0),
            dxy=self._dxy.get(completed_day, 0.0),
            etf_daily_net_inflow=float(etf.get("totalNetInflow", 0.0)) if etf else 0.0,
            etf_total_net_assets=float(etf.get("totalNetAssets", 0.0)) if etf else 0.0,
            etf_cum_net_inflow=float(etf.get("cumNetInflow", 0.0)) if etf else 0.0,
            vix=self._vix.get(completed_day, 0.0),
            sp500=self._sp500.get(completed_day, 0.0),
            stablecoin_total_supply=self._stablecoin_supply.get(completed_day, 0.0),
            btc_hashrate=self._btc_hashrate.get(completed_day, 0.0),
        )
        return DataSnapshot(
            timestamp=as_of,
            pair=self.pair.canonical(),
            market=MarketData(
                pair=self.pair.canonical(),
                ohlcv=frame,
                ticker={"last": current[4], "baseVolume": current[5]},
                funding_rate=float(self._funding.get(completed_day, 0.0)),
                orderbook_imbalance=0.0,
                volatility=float(volatility) if not pd.isna(volatility) else 0.0,
            ),
            onchain=onchain,
            news=NewsSentiment(
                key_events=derive_news_events(candles, len(candles) - 1),
                headlines=[
                    f"{self.pair.base} at ${float(current[4]):,.0f}, Fear&Greed={self._fng.get(completed_day, 50)}"
                ],
            ),
            macro=macro,
        )

    def _compute_result(
        self,
        equity: float,
        curve: list[float],
        trades: list[dict],
        *,
        records: list[TradingCycleRecord] | None = None,
        outcomes: list[CycleOutcome] | None = None,
    ) -> BacktestResult:
        returns = [
            (curve[index] - curve[index - 1]) / curve[index - 1]
            for index in range(1, len(curve))
            if curve[index - 1] > 0.0
        ]
        if returns:
            average = sum(returns) / len(returns)
            deviation = math.sqrt(sum((item - average) ** 2 for item in returns) / len(returns))
            periods_per_day = 86_400_000 / _TF_MS.get(self.interval, 3_600_000)
            sharpe = average / deviation * math.sqrt(365 * periods_per_day) if deviation > 0.0 else 0.0
        else:
            sharpe = 0.0
        peak = curve[0] if curve else self.capital
        max_drawdown = 0.0
        for value in curve:
            peak = max(peak, value)
            max_drawdown = min(max_drawdown, (value - peak) / peak if peak > 0.0 else 0.0)
        closed = [trade for trade in trades if trade.get("pnl") is not None and trade.get("pnl") != 0.0]
        wins = sum(1 for trade in closed if trade["pnl"] > 0.0)
        records = records or []
        outcomes = outcomes or []
        decisions = [self._decision_payload(record) for record in records]
        if not decisions:
            decisions = [
                {
                    "cycle_id": outcome.cycle_id,
                    "status": outcome.status,
                    "profile_revision": outcome.profile_revision,
                }
                for outcome in outcomes
            ]
        return BacktestResult(
            total_return=(equity - self.capital) / self.capital,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=wins / len(closed) if closed else 0.0,
            trades=list(trades),
            equity_curve=list(curve),
            decisions=decisions,
            cycle_records=records,
            cycle_ids=[outcome.cycle_id for outcome in outcomes],
            profile_revisions=[outcome.profile_revision for outcome in outcomes],
        )

    @staticmethod
    def _decision_payload(record: TradingCycleRecord) -> dict[str, Any]:
        return {
            "cycle_id": record.cycle_id,
            "ts": record.context_summary.get("as_of"),
            "price": record.context_summary.get("current_price"),
            "status": record.status,
            "profile_revision": record.profile_revision,
            "components": list(record.component_signals),
            "fusion": record.fused_signal,
            "target_position": record.target_position,
            "risk_result": record.risk_result,
            "execution_result": record.execution_result,
        }
