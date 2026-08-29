"""Historical execution of the same TradingCycle used by paper and live modes."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd

from cryptotrader._compat import UTC
from cryptotrader.backtest.cache import _TF_MS, fetch_historical
from cryptotrader.backtest.result import BacktestResult
from cryptotrader.decision.models import CycleRequest
from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData
from cryptotrader.pair import Pair
from cryptotrader.signals.models import CandleRequirement, DataRequirements
from cryptotrader.venues.models import VenueConnection

if TYPE_CHECKING:
    from collections.abc import Callable

    from cryptotrader.decision.models import CycleOutcome
    from cryptotrader.journal.models import MultiVenueCycleRecord
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot
    from cryptotrader.signals.registry import SignalComponentRegistry

logger = logging.getLogger(__name__)


class _FrozenRuntimeRepository:
    """Backtest-local repository that cannot reveal or switch venue configuration."""

    database_url = None

    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot

    async def get_or_create(self):
        return self.snapshot


class BacktestEngine:
    def __init__(
        self,
        pair: str,
        start: str,
        end: str,
        interval: str = "4h",
        initial_capital: float | None = None,
        lookback: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
        *,
        repository=None,
        snapshot: RuntimeConfigSnapshot | None = None,
        signal_registry: SignalComponentRegistry | None = None,
        venue_registry=None,
    ) -> None:
        self.pair = Pair.parse(pair)
        self.start = start
        self.end = end
        self.start_ms = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1000)
        self.end_ms = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1000)
        self.interval = interval
        self.capital = initial_capital if initial_capital is not None else 10_000.0
        self.lookback = lookback if lookback is not None else 512
        self.progress_callback = progress_callback
        self.repository = repository
        self.snapshot = snapshot
        self.signal_registry = signal_registry
        self.venue_registry = venue_registry
        self._as_of: datetime | None = None
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
        from cryptotrader.cycle_events import NullCycleEventSink
        from cryptotrader.journal.store import MultiVenueCycleStore
        from cryptotrader.market_sources.registry import MarketSourceRegistry
        from cryptotrader.runtime import build_runtime
        from cryptotrader.signals.context import HistoricalSignalContextProvider
        from cryptotrader.signals.registry import SignalComponentRegistry
        from cryptotrader.venues.paper import PaperVenueAdapter
        from cryptotrader.venues.registry import VenueAdapterRegistry

        if self.snapshot is not None:
            source_snapshot = self.snapshot
        else:
            if self.repository is None:
                from cryptotrader.bootstrap import BootstrapSettings
                from cryptotrader.runtime_config.repository import RuntimeConfigRepository
                from cryptotrader.runtime_config.secrets import CredentialVault

                settings = BootstrapSettings.from_environment()
                source_repository = RuntimeConfigRepository(
                    settings.database_url,
                    CredentialVault(settings.config_master_key),
                )
            else:
                source_repository = self.repository
            source_snapshot = await source_repository.get_or_create()
        events = NullCycleEventSink()
        registry = self.signal_registry or SignalComponentRegistry.discover(source_snapshot.document, events)
        default_timeframe = str(source_snapshot.document.market_data.parameters.get("timeframe", self.interval))
        limit = int(source_snapshot.document.market_data.parameters.get("limit", self.lookback))
        profile = source_snapshot.document.signals.to_profile(source_snapshot.revision)
        components = registry.enabled(profile)
        requirements = DataRequirements.merge(
            *(component.requirements() for component in components),
            DataRequirements(candles=(CandleRequirement(default_timeframe, max(20, limit)),)),
            DataRequirements(candles=(CandleRequirement(self.interval, self.lookback),)),
        )
        await self._fetch_historical_data(requirements)
        if not self._candles:
            return BacktestResult()
        historical = HistoricalSignalContextProvider(
            self._snapshot_at,
            default_timeframe=default_timeframe,
        )
        frozen = self._backtest_snapshot(source_snapshot)
        frozen_repository = _FrozenRuntimeRepository(frozen)
        paper_registry = self.venue_registry or VenueAdapterRegistry((PaperVenueAdapter(),))
        if set(paper_registry.ids()) != {"paper"}:
            raise ValueError("backtest venue registry must contain only the Paper adapter")
        runtime = await build_runtime(
            repository=frozen_repository,
            snapshot=frozen,
            signal_registry=registry,
            venue_registry=paper_registry,
            market_registry=MarketSourceRegistry((historical,)),
            event_sink=events,
        )
        try:
            async with runtime.cycle_lease() as cycle:
                cycle.clock = self._clock
                cycle.journal = MultiVenueCycleStore()
                return await self._run_bars(cycle, runtime.sessions["backtest-paper"])
        finally:
            await runtime.close()

    def _backtest_snapshot(self, source_snapshot):
        from cryptotrader.runtime_config.models import (
            ExecutionConfig,
            InfrastructureConfig,
            MarketDataConfig,
            RuntimeConfigSnapshot,
            SystemConfig,
        )

        default_timeframe = str(source_snapshot.document.market_data.parameters.get("timeframe", self.interval))
        limit = int(source_snapshot.document.market_data.parameters.get("limit", self.lookback))
        connection = VenueConnection(
            id="backtest-paper",
            label="Backtest Paper",
            adapter_id="paper",
            environment="paper",
            enabled=True,
            credential_ref=None,
            leverage=1,
            margin_mode="isolated",
            canary_only=False,
            parameters={"initial_equity": str(self.capital)},
        )
        book = ExecutionBook(
            id="backtest",
            label="Backtest Paper",
            capital_scope="simulated",
            enabled=True,
            hitl_required=False,
            allocations=(ConnectionAllocation("backtest-paper", True, 1.0),),
        )
        document = source_snapshot.document.model_copy(
            update={
                "system": SystemConfig(active=True),
                "market_data": MarketDataConfig(
                    source_id="historical",
                    parameters={"timeframe": default_timeframe, "limit": limit},
                ),
                "execution": ExecutionConfig(connections=(connection,), books=(book,)),
                # Historical Paper execution never enters Runtime.execution_lease.
                # This isolated, unroutable endpoint keeps the frozen document
                # structurally valid without becoming a production escape hatch.
                "infrastructure": InfrastructureConfig(redis_url="redis://backtest.invalid:6379/0"),
            }
        )
        return RuntimeConfigSnapshot(source_snapshot.revision, document, source_snapshot.updated_at)

    async def _run_bars(self, cycle, session) -> BacktestResult:
        interval_ms = _TF_MS.get(self.interval)
        if interval_ms is None:
            raise ValueError(f"unsupported backtest timeframe {self.interval!r}")
        indexes = [
            index for index, candle in enumerate(self._candles) if self.start_ms <= int(candle[0]) <= self.end_ms
        ]
        if not indexes:
            return BacktestResult(equity_curve=[self.capital])

        outcomes: list[CycleOutcome] = []
        curve = [self.capital]
        for step, index in enumerate(indexes):
            candle = self._candles[index]
            self._as_of = datetime.fromtimestamp((int(candle[0]) + interval_ms) / 1000, UTC)
            await session.set_quote(self.pair, Decimal(str(candle[4])))
            outcome = await cycle.run(CycleRequest(self.pair))
            outcomes.append(outcome)
            portfolio = await session.fetch_portfolio(self.pair)
            curve.append(float(portfolio.equity))
            if self.progress_callback is not None:
                self.progress_callback((step + 1) / len(indexes))

        final_equity = curve[-1]
        records = []
        for outcome in outcomes:
            record = await cycle.journal.get(outcome.cycle_id)
            if record is None:
                raise RuntimeError(f"backtest cycle {outcome.cycle_id!r} is missing from the journal")
            records.append(record)
        return self._compute_result(
            final_equity,
            curve,
            [],
            records=records,
            outcomes=outcomes,
        )

    def _clock(self) -> datetime:
        if self._as_of is None:
            raise RuntimeError("backtest cycle clock is not initialized")
        return self._as_of

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
        records: list[MultiVenueCycleRecord] | None = None,
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
                    "config_revision": outcome.config_revision,
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
            config_revisions=[outcome.config_revision for outcome in outcomes],
        )

    @staticmethod
    def _decision_payload(record: MultiVenueCycleRecord) -> dict[str, Any]:
        return {
            "cycle_id": record.cycle_id,
            "ts": record.created_at.isoformat(),
            "status": record.cycle_status,
            "config_revision": record.config_revision,
            "components": list(record.component_signals),
            "fusion": record.fused_signal,
            "target_position": record.target_position,
            "books": list(record.book_results),
        }
