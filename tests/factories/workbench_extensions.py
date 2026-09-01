"""Code-registered extensions with deterministic, local-only behavior."""

from pydantic import Field, SecretStr

from cryptotrader.configuration.parameters import PluginParameters
from cryptotrader.signals.models import ComponentSignal, DataRequirements
from cryptotrader.signals.presentation import Metric, MetricsBlock, TableBlock, TableCell, TableColumn, TableRow
from cryptotrader.venues.paper import PaperVenueSession


class SampleParameters(PluginParameters):
    account_code: str


class SampleCredentials(PluginParameters):
    access_token: SecretStr = Field(min_length=1, title="Access token")
    tenant_pin: SecretStr | None = Field(default=None, title="Tenant PIN")


class SampleSignalParameters(PluginParameters):
    window: int = Field(default=12, ge=1)


class SampleSignal:
    id = "sample_signal"
    display_name = "Sample signal"
    description = "Local fixture"

    def __init__(self, context):
        self.context = context
        configured = next(item for item in context.document.signals.components if item.component_id == self.id)
        self.parameters = SampleSignalParameters.model_validate(dict(configured.parameters))

    def requirements(self):
        return DataRequirements()

    async def evaluate(self, context):
        return ComponentSignal(
            self.id,
            "neutral",
            0.0,
            "sample",
            details={"window": self.parameters.window, "rows": [[1, 2]]},
            blocks=(
                MetricsBlock(title="测试指标", metrics=(Metric(key="观察窗口", value=self.parameters.window),)),
                TableBlock(
                    title="测试表格",
                    columns=(TableColumn(key="window", label="观察窗口"),),
                    rows=(TableRow(cells=(TableCell(column_key="window", value=self.parameters.window),)),),
                ),
            ),
        )


class SampleVenue:
    adapter_id = "sample_venue"

    def capabilities(self, environment):
        from cryptotrader.venues.models import ACCOUNT_READS, VenueCapabilities

        return VenueCapabilities(frozenset({"spot"}), False, False, True, frozenset({"market"}), ACCOUNT_READS)

    async def connect(self, connection, credentials):
        from decimal import Decimal

        from cryptotrader.venues.paper import _PaperAccount

        self.received_values = {key: value.get_secret_value() for key, value in credentials.values.items()}
        return SampleVenueSession(
            connection, _PaperAccount(Decimal("100"), connection.leverage), self.capabilities(connection.environment)
        )


class SampleVenueSession(PaperVenueSession):
    """Concrete local execution engine plus canned external-account read facts."""

    def __init__(self, connection, account, capabilities):
        super().__init__(connection, account, capabilities)
        self.write_calls = []

    async def place_order(self, intent):
        self.write_calls.append(("place_order", intent))
        return await super().place_order(intent)

    async def cancel_order(self, order_id, pair):
        self.write_calls.append(("cancel_order", order_id))
        return await super().cancel_order(order_id, pair)

    async def list_instruments(self):
        from cryptotrader.accounts.models import Instrument
        from cryptotrader.pair import Pair

        return (
            Instrument("BTCUSDT", Pair.parse("BTC/USDT"), "spot", True),
            Instrument("ETHUSDT", Pair.parse("ETH/USDT"), "spot", True),
            Instrument("UNKNOWN", None, "unknown", False, "unmapped_venue_instrument"),
        )

    async def fetch_account(self):
        from datetime import UTC, datetime
        from decimal import Decimal

        from cryptotrader.accounts.models import AccountOrder, AccountPosition, AccountSnapshot, Money

        instruments = await self.list_instruments()
        now = datetime.now(UTC)
        unknown = Money(None, "USD", "not_provided_by_sample")
        positions = tuple(AccountPosition(item, Decimal("1"), None, unknown, None, unknown) for item in instruments)
        orders = tuple(
            AccountOrder(
                self.connection_id,
                f"sample-{index}",
                instrument,
                "sell",
                "limit",
                Decimal("1"),
                Decimal("0"),
                None,
                "open",
                protection,
                protection,
                None,
                now,
                unknown,
            )
            for index, (instrument, protection) in enumerate(zip(instruments[:2], (False, True), strict=True))
        )
        return AccountSnapshot(
            self.connection_id,
            now,
            "simulated",
            Money(Decimal("100"), "USD"),
            (Money(Decimal("100"), "USD"),),
            positions,
            orders,
            unknown,
            unknown,
            ("margin:not_provided_by_sample",),
        )

    async def fetch_fills(self, cursor):
        from datetime import UTC, datetime
        from decimal import Decimal

        from cryptotrader.accounts.models import Fill, FillPage, Money

        instrument = (await self.list_instruments())[0]
        fill = Fill(
            self.connection_id,
            "sample-fill",
            "sample-order",
            instrument,
            "buy",
            Decimal("1"),
            Decimal("10"),
            datetime.now(UTC),
            Money(Decimal("0.1"), "USDT"),
            Money(None, "USDT", "opening_cost_unknown"),
            "platform",
        )
        return FillPage((fill,) if cursor is None else (), "sample-end", True)

    async def fetch_funding(self, cursor):
        from datetime import UTC, datetime
        from decimal import Decimal

        from cryptotrader.accounts.models import FundingEntry, FundingPage, Money

        item = FundingEntry(
            self.connection_id,
            "sample-funding",
            (await self.list_instruments())[0],
            Money(Decimal("-0.2"), "USDT"),
            datetime.now(UTC),
        )
        return FundingPage((item,) if cursor is None else (), "sample-end", True)


def sample_registry():
    from cryptotrader.configuration.catalog import EnvironmentDefinition, PluginConfiguration
    from cryptotrader.configuration.fields import LocalizedText
    from cryptotrader.configuration.registry import ExtensionRegistration, ExtensionRegistry, get_extension_registry

    calls = []

    def create_signal(context):
        calls.append("signal")
        return SampleSignal(context)

    def create_venue():
        calls.append("venue")
        return SampleVenue()

    label = LocalizedText("测试", "Sample")
    venue = PluginConfiguration(
        "sample_venue",
        label,
        label,
        SampleParameters,
        environments=(EnvironmentDefinition("sandbox", label, "simulated"),),
        credential_model=SampleCredentials,
        margin_modes=("cross",),
        capabilities=SampleVenue().capabilities("sandbox"),
    )
    signal = PluginConfiguration("sample_signal", label, label, SampleSignalParameters)
    builtins = get_extension_registry()
    return ExtensionRegistry(
        components={**builtins.components, "sample_signal": ExtensionRegistration(signal, create_signal)},
        venues={**builtins.venues, "sample_venue": ExtensionRegistration(venue, create_venue)},
        market_sources=builtins.market_sources,
    ), calls
