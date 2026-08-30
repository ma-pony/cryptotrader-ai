"""Built-in public-market source backed by the existing data collectors."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from cryptotrader.agents._indicators import atr
from cryptotrader.configuration.catalog import PluginConfiguration, configured_factory
from cryptotrader.configuration.fields import LocalizedText
from cryptotrader.configuration.parameters import DefaultMarketSourceParameters
from cryptotrader.data.market import clip_ohlcv_at
from cryptotrader.data.snapshot import SnapshotAggregator
from cryptotrader.signals.models import DataRequirements, SignalContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from cryptotrader.models import DataSnapshot, MarketData
    from cryptotrader.pair import Pair
    from cryptotrader.runtime_config.models import MarketDataConfig


def _materialize_snapshot(base: DataSnapshot, market: MarketData, as_of: datetime, limit: int) -> DataSnapshot:
    clipped = replace(market, ohlcv=clip_ohlcv_at(market.ohlcv, as_of, limit))
    return replace(base, timestamp=as_of, market=clipped)


def _price(snapshot: DataSnapshot) -> float:
    ticker_price = float(snapshot.market.ticker.get("last", 0.0) or 0.0)
    if ticker_price > 0.0:
        return ticker_price
    if snapshot.market.ohlcv.empty:
        raise ValueError("market snapshot has no current price")
    return float(snapshot.market.ohlcv["close"].iloc[-1])


def _atr(snapshot: DataSnapshot) -> float:
    frame = snapshot.market.ohlcv
    values = atr(frame["high"], frame["low"], frame["close"], length=14).dropna()
    return float(values.iloc[-1]) if not values.empty else 0.0


class DefaultMarketDataSource:
    id = "default"
    _MAX_AS_OF_SKEW = timedelta(seconds=60)

    def __init__(
        self,
        config: MarketDataConfig,
        *,
        aggregator=None,
        news_provider_key: str = "",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        parameters = DefaultMarketSourceParameters.model_validate(dict(config.parameters))
        self.market_adapter_id = parameters.market_adapter_id
        if not self.market_adapter_id:
            raise ValueError("default market source requires market_adapter_id")
        self.kronos_aux_symbol = parameters.kronos_aux_symbol
        self.aggregator = aggregator or SnapshotAggregator(coindesk_api_key=news_provider_key)
        self.market = self.aggregator.market
        self._clock = clock or (lambda: datetime.now(UTC))

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def collect(
        self,
        pair: Pair,
        as_of: datetime,
        requirements: DataRequirements,
    ) -> SignalContext:
        now = self._clock()
        if as_of.tzinfo is None or now.tzinfo is None:
            raise ValueError("default market source is live-only and requires timezone-aware as_of")
        if abs(now - as_of) > self._MAX_AS_OF_SKEW:
            raise ValueError("default market source is live-only; historical or delayed as_of is not supported")
        if not requirements.candles:
            raise ValueError("market source requires at least one candle timeframe")
        primary = requirements.candles[0]
        base = await self.aggregator.collect(
            pair=pair.canonical(),
            market_adapter_id=self.market_adapter_id,
            timeframe=primary.timeframe,
            limit=primary.limit,
            backtest_mode=False,
            kronos_aux=requirements.kronos_aux,
            kronos_aux_symbol=self.kronos_aux_symbol,
        )
        snapshots = {
            primary.timeframe: _materialize_snapshot(base, base.market, as_of, primary.limit),
        }
        for requirement in requirements.candles[1:]:
            market = await self.market.collect(
                pair.canonical(),
                self.market_adapter_id,
                requirement.timeframe,
                requirement.limit,
            )
            snapshots[requirement.timeframe] = _materialize_snapshot(base, market, as_of, requirement.limit)

        primary_snapshot = snapshots[primary.timeframe]
        return SignalContext(
            pair=pair,
            as_of=as_of,
            market_data_source_id=self.id,
            market_type=pair.market_type,
            current_price=_price(primary_snapshot),
            atr=_atr(primary_snapshot),
            snapshots=snapshots,
        )


@configured_factory(
    PluginConfiguration(
        id="default",
        label=LocalizedText(zh_CN="默认市场数据", en_US="Default market data"),
        description=LocalizedText(
            zh_CN="采集公开市场、链上和宏观数据以构建交易上下文。",
            en_US="Collects public market, on-chain, and macro data for the trading context.",
        ),
        parameter_model=DefaultMarketSourceParameters,
    )
)
def create_source(config: MarketDataConfig, *, news_provider_key: str = "") -> DefaultMarketDataSource:
    return DefaultMarketDataSource(config, news_provider_key=news_provider_key)
