"""Historical execution of the same TradingCycle used by paper and live modes."""

from __future__ import annotations

import asyncio
import logging
import math
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pandas as pd

from cryptotrader._compat import UTC
from cryptotrader.backtest.cache import _TF_MS
from cryptotrader.backtest.result import BacktestResult, EquityPoint, closed_round_trips
from cryptotrader.decision.exit_policy import exit_candle_requirement
from cryptotrader.decision.models import CycleRequest
from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData
from cryptotrader.pair import Pair
from cryptotrader.signals.models import CandleRequirement, DataRequirements
from cryptotrader.venues.models import BacktestCostModel, VenueConnection

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

    def __init__(self, snapshot, account_store) -> None:
        self.snapshot = snapshot
        self.account_store = account_store

    async def get_or_create(self):
        return self.snapshot


class _HistoricalBookState:
    """One isolated replay's peak; never a fallback for production persistence."""

    def __init__(self):
        self.states = {}
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def book(self, _book_id):
        async with self.lock:
            yield

    async def update(self, book_id, snapshots):
        from cryptotrader.risk.book_state import BookRiskState

        prior = self.states.get(book_id)
        state = BookRiskState.from_snapshots(book_id, snapshots, peak_equity=prior.peak_equity if prior else None)
        self.states[book_id] = state
        return state


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
        cost_model: BacktestCostModel | None = None,
        market_registry=None,
        funding_settlements=None,
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
        self.cost_model = cost_model or BacktestCostModel()
        self.market_registry = market_registry
        self._source_config = None
        self._data_coverage = {}
        self.funding_settlements = tuple(funding_settlements) if funding_settlements is not None else None
        for event in self.funding_settlements or ():
            if event.pair != self.pair.canonical():
                raise ValueError("funding settlement pair does not match replay")
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
        from cryptotrader.market_sources.registry import MarketSourceRegistry
        from cryptotrader.signals.context import HistoricalSignalContextProvider
        from cryptotrader.signals.registry import SignalComponentRegistry

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
        frozen = self._backtest_snapshot(source_snapshot)
        registry = self.signal_registry or SignalComponentRegistry.discover(frozen.document, events)
        self._source_config = source_snapshot.document.market_data
        self.market_registry = self.market_registry or MarketSourceRegistry.discover(self._source_config)
        default_timeframe = self._source_config.timeframe
        exit_candles = exit_candle_requirement(self._source_config)
        profile = source_snapshot.document.signals.to_profile(source_snapshot.revision)
        components = registry.enabled(profile)
        requirements = DataRequirements.merge(
            *(component.requirements() for component in components),
            DataRequirements(candles=(exit_candles,)),
            DataRequirements(candles=(CandleRequirement(default_timeframe, 2),)),
            DataRequirements(candles=(CandleRequirement(self.interval, self.lookback),)),
        )
        await self._fetch_historical_data(requirements)
        historical = HistoricalSignalContextProvider(
            self._snapshot_at,
            default_timeframe=default_timeframe,
            atr_timeframe=exit_candles.timeframe,
            source_id=self._source_config.source_id,
        )
        self._as_of = datetime.fromtimestamp(self.start_ms / 1000, UTC)
        if not self._candles:
            return self._compute_result([EquityPoint(self._clock(), Decimal(str(self.capital)))], [])
        with TemporaryDirectory(prefix="cryptotrader-replay-") as temporary:
            from cryptotrader.accounts.store import AccountStore
            from cryptotrader.db import dispose_engine
            from cryptotrader.migrations.workbench import migrate_workbench_schema

            account_store = AccountStore(f"sqlite+aiosqlite:///{temporary}/account.sqlite", clock=self._clock)
            try:
                await migrate_workbench_schema(account_store.database_url)
                return await self._execute_replay(frozen, registry, historical, events, account_store)
            finally:
                await dispose_engine(account_store.database_url)

    async def _execute_replay(self, frozen, registry, historical, events, account_store):
        from cryptotrader.journal.store import MultiVenueCycleStore
        from cryptotrader.market_sources.registry import MarketSourceRegistry
        from cryptotrader.runtime import build_runtime
        from cryptotrader.venues.paper import PaperVenueAdapter
        from cryptotrader.venues.registry import VenueAdapterRegistry

        frozen_repository = _FrozenRuntimeRepository(frozen, account_store)
        paper_registry = self.venue_registry or VenueAdapterRegistry((PaperVenueAdapter(),))
        if set(paper_registry.ids()) != {"paper"}:
            raise ValueError("backtest venue registry must contain only the Paper adapter")
        paper = paper_registry.require("paper")
        if not isinstance(paper, PaperVenueAdapter):
            raise ValueError("backtest requires the local Paper implementation")
        paper.clock = self._clock
        paper.cost_model = self.cost_model
        runtime = await build_runtime(
            repository=frozen_repository,
            snapshot=frozen,
            signal_registry=registry,
            venue_registry=paper_registry,
            market_registry=MarketSourceRegistry((historical,)),
            event_sink=events,
            recover_unfinished=False,
        )
        try:
            async with runtime.cycle_lease() as cycle:
                historical_state = _HistoricalBookState()
                cycle.ownership = historical_state
                cycle.risk_states = historical_state
                cycle.clock = self._clock
                cycle.journal = MultiVenueCycleStore()
                return await self._run_bars(cycle, runtime.sessions["backtest-paper"])
        finally:
            await runtime.close()

    def _backtest_snapshot(self, source_snapshot):
        from cryptotrader.runtime_config.models import (
            ExecutionConfig,
            InfrastructureConfig,
            RuntimeConfigSnapshot,
        )

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
            index
            for index, candle in enumerate(self._candles)
            if self.start_ms <= int(candle[0]) and int(candle[0]) + interval_ms <= self.end_ms
        ]
        if not indexes:
            return self._compute_result(
                [EquityPoint(datetime.fromtimestamp(self.start_ms / 1000, UTC), Decimal(str(self.capital)))], []
            )

        outcomes: list[CycleOutcome] = []
        curve = [EquityPoint(datetime.fromtimestamp(self.start_ms / 1000, UTC), Decimal(str(self.capital)))]
        settlements = iter(sorted(self.funding_settlements or (), key=lambda item: item.occurred_at))
        pending_funding = next(settlements, None)
        for step, index in enumerate(indexes):
            candle = self._candles[index]
            closed_at = datetime.fromtimestamp((int(candle[0]) + interval_ms) / 1000, UTC)
            while pending_funding is not None and pending_funding.occurred_at <= closed_at:
                if pending_funding.occurred_at >= curve[0].time:
                    self._as_of = pending_funding.occurred_at
                    await session.apply_funding(
                        self.pair,
                        settlement_id=f"{pending_funding.source_id}:{pending_funding.id}",
                        rate=pending_funding.rate,
                        mark_price=pending_funding.mark_price,
                    )
                pending_funding = next(settlements, None)
            self._as_of = closed_at
            from cryptotrader.market_sources.protocol import HistoricalCandle

            await session.advance_bar(
                self.pair,
                HistoricalCandle(
                    open_time=datetime.fromtimestamp(int(candle[0]) / 1000, UTC),
                    **{
                        name: Decimal(str(value))
                        for name, value in zip(("open", "high", "low", "close", "volume"), candle[1:], strict=True)
                    },
                ),
            )
            outcome = await cycle.run(CycleRequest(self.pair, mode="backtest", origin="backtest"))
            outcomes.append(outcome)
            portfolio = await session.fetch_portfolio(self.pair)
            curve.append(EquityPoint(self._clock(), portfolio.equity))
            if self.progress_callback is not None:
                import inspect

                pending = self.progress_callback((step + 1) / len(indexes))
                if inspect.isawaitable(pending):
                    await pending

        records = []
        for outcome in outcomes:
            record = await cycle.journal.get(outcome.cycle_id)
            if record is None:
                raise RuntimeError(f"backtest cycle {outcome.cycle_id!r} is missing from the journal")
            records.append(record)
        fills = await self._read_paper_history(session.fetch_fills, "venue_fill_id")
        funding_entries = await self._read_paper_history(session.fetch_funding, "venue_entry_id")
        return self._compute_result(
            curve,
            fills,
            funding_entries=funding_entries,
            records=records,
            outcomes=outcomes,
        )

    def _clock(self) -> datetime:
        if self._as_of is None:
            raise RuntimeError("backtest cycle clock is not initialized")
        return self._as_of

    async def _read_paper_history(self, fetch, identity):
        items, cursor = {}, None
        while True:
            page = await fetch(cursor)
            items.update((getattr(item, identity), item) for item in page.items)
            if page.coverage_end >= self._clock():
                return list(items.values())
            cursor = page.next_cursor

    async def _fetch_historical_data(self, requirements: DataRequirements) -> None:
        source = self.market_registry.require(self._source_config.source_id)
        limits = {item.timeframe: item.limit for item in requirements.candles}
        limits[self.interval] = max(limits.get(self.interval, 0), self.lookback)
        self._candles_by_timeframe = {}
        self._data_coverage = {
            "market_source_id": self._source_config.source_id,
            "market_parameters": dict(self._source_config.parameters),
            "pair": self.pair.canonical(),
            "market_type": self.pair.market_type,
            "as_of": datetime.fromtimestamp(self.end_ms / 1000, UTC).isoformat(),
            "candles": {},
            "historical_news": "unavailable; never replaced with current news",
            "price_context": "derived from historical OHLCV, not historical news",
            "daily_context": "previous completed day only; unavailable values are neutral placeholders",
            "unavailable_context": [
                "funding_rate",
                "open_interest",
                "long_short_ratio",
                "etf",
                "macro_series",
                "orderbook",
            ],
            "model_limitations": (
                "pretrained models may contain knowledge after as_of; not a strict out-of-sample replay"
            ),
        }
        for timeframe, limit in limits.items():
            timeframe_ms = _TF_MS.get(timeframe)
            if timeframe_ms is None:
                raise ValueError(f"unsupported backtest timeframe {timeframe!r}")
            start = datetime.fromtimestamp((self.start_ms - limit * timeframe_ms) / 1000, UTC)
            end = datetime.fromtimestamp(self.end_ms / 1000, UTC)
            bars = await source.read_candles(self.pair, timeframe, start, end, end)
            bars = sorted(
                (
                    bar
                    for bar in bars
                    if start <= bar.open_time < end and bar.open_time + timedelta(milliseconds=timeframe_ms) <= end
                ),
                key=lambda bar: bar.open_time,
            )
            self._candles_by_timeframe[timeframe] = [
                [
                    int(bar.open_time.timestamp() * 1000),
                    *[getattr(bar, name) for name in ("open", "high", "low", "close", "volume")],
                ]
                for bar in bars
            ]
            expected = (self.end_ms - int(start.timestamp() * 1000)) // timeframe_ms
            self._data_coverage["candles"][timeframe] = {
                "expected": expected,
                "available": len(bars),
                "missing": max(0, expected - len(bars)),
                "first_open": bars[0].open_time.isoformat() if bars else None,
                "last_close": (bars[-1].open_time + timedelta(milliseconds=timeframe_ms)).isoformat() if bars else None,
            }
        self._candles = self._candles_by_timeframe[self.interval]
        # Fear & Greed has dated published historical observations. The old
        # exchange-agnostic daily-average funding and unversioned local macro
        # cache cannot prove settlement events / point-in-time market identity.
        from cryptotrader.backtest.historical_data import fetch_fear_greed

        start = datetime.fromisoformat(self.start).replace(tzinfo=UTC)
        daily_start = (start - timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            self._fng = await fetch_fear_greed(
                daily_start, datetime.fromtimestamp(self.end_ms / 1000, UTC).strftime("%Y-%m-%d")
            )
        except Exception:
            self._fng = {}
            logger.warning("Historical Fear & Greed unavailable")
        self._data_coverage["fear_greed"] = {"source": "alternative.me", "observations": len(self._fng)}

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
        # Components use numeric frames; execution keeps the original Decimal bars.
        for name in ("open", "high", "low", "close", "volume"):
            frame[name] = frame[name].astype(float)
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
                key_events=derive_news_events(candles, len(candles) - 1, pair=self.pair.base, timeframe=timeframe),
                headlines=[
                    "Historical news unavailable. Price-derived context only: "
                    f"{self.pair.base} at ${float(current[4]):,.0f}. "
                    f"Fear&Greed={self._fng.get(completed_day, 'unavailable')}. "
                    "Missing historical macro, funding and orderbook inputs are placeholders, not observations."
                ],
            ),
            macro=macro,
        )

    def _compute_result(
        self,
        curve: list[EquityPoint],
        fills: list,
        *,
        records: list[MultiVenueCycleRecord] | None = None,
        outcomes: list[CycleOutcome] | None = None,
        funding_entries: list | None = None,
    ) -> BacktestResult:
        values = [float(point.equity) for point in curve]
        returns = [
            (values[index] - values[index - 1]) / values[index - 1]
            for index in range(1, len(curve))
            if values[index - 1] > 0.0
        ]
        if returns:
            average = sum(returns) / len(returns)
            deviation = math.sqrt(sum((item - average) ** 2 for item in returns) / len(returns))
            periods_per_day = 86_400_000 / _TF_MS.get(self.interval, 3_600_000)
            sharpe = average / deviation * math.sqrt(365 * periods_per_day) if deviation > 0.0 else 0.0
        else:
            sharpe = 0.0
        peak = values[0] if values else self.capital
        max_drawdown = 0.0
        for value in values:
            peak = max(peak, value)
            max_drawdown = min(max_drawdown, (value - peak) / peak if peak > 0.0 else 0.0)
        funding_entries = funding_entries or []
        closed = closed_round_trips(fills, funding_entries)
        wins = sum(1 for trade in closed if trade.net_pnl > 0)
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
            total_return=float((curve[-1].equity - Decimal(str(self.capital))) / Decimal(str(self.capital)))
            if curve
            else 0.0,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=wins / len(closed) if closed else None,
            fills=list(fills),
            closed_trades=closed,
            fees=sum((fill.fee.amount for fill in fills), Decimal("0")),
            funding=sum((entry.amount.amount for entry in funding_entries), Decimal("0")),
            funding_entries=funding_entries,
            cost_assumptions={
                "fee_rate": str(self.cost_model.fee_rate),
                "slippage_bps": str(self.cost_model.slippage_bps),
                "funding_enabled": self.cost_model.funding_enabled,
                "funding": (
                    "only supplied settlement points; preceding-close position, before current-bar protections; "
                    "no invented rate or settlement"
                ),
                "execution": (
                    "closed-bar simulation, not tick execution; existing protection first, "
                    "then close signal/market fill; "
                    "new protection next bar; protection fills timestamped at bar close"
                ),
                "protection": (
                    "stop first if both touched; stop gap uses worse opening price; take-profit at target; "
                    "directional slippage and fees on every fill"
                ),
            },
            unmodeled_costs=["market impact and intrabar path"]
            + (
                ["historical funding settlements unavailable outside supplied points"]
                if self.cost_model.funding_enabled and self.pair.market_type != "spot"
                else []
            ),
            data_coverage={
                **self._data_coverage,
                "funding": {
                    "status": "not_applicable"
                    if self.pair.market_type == "spot"
                    else "disabled"
                    if not self.cost_model.funding_enabled
                    else "partial"
                    if self.funding_settlements
                    else "unavailable",
                    "provided_settlements": len(self.funding_settlements or ()),
                    "applied_settlements": len(funding_entries),
                    "source_ids": sorted({entry.source_id for entry in self.funding_settlements or ()}),
                },
            },
            equity_curve=list(curve),
            decisions=decisions,
            decision_ids=[record.cycle_id for record in records],
            cycle_records=records,
            cycle_ids=[outcome.cycle_id for outcome in outcomes],
            config_revisions=[outcome.config_revision for outcome in outcomes],
        )

    @staticmethod
    def _decision_payload(record: MultiVenueCycleRecord) -> dict[str, Any]:
        from cryptotrader.decision.read_service import decision_out

        return decision_out(record).model_dump(mode="json")
