"""Structural protocols implemented by every executable venue."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from decimal import Decimal

    from cryptotrader.accounts.models import AccountSnapshot, FillPage, FundingPage, Instrument
    from cryptotrader.pair import Pair
    from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot
    from cryptotrader.runtime_config.secrets import CredentialPayload
    from cryptotrader.venues.models import (
        ConnectionEnvironment,
        NormalizedOrder,
        OpenVenueState,
        OrderIntent,
        ProtectionSpec,
        ProtectionState,
        VenueCapabilities,
        VenueConnection,
        VenueQuote,
    )


class VenueOperationError(RuntimeError):
    """A credential-safe failure raised by a normalized venue operation."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


@runtime_checkable
class VenueAdapter(Protocol):
    """A code-owned platform implementation that opens isolated sessions."""

    adapter_id: str

    def capabilities(self, environment: ConnectionEnvironment) -> VenueCapabilities: ...

    async def connect(
        self,
        connection: VenueConnection,
        credentials: CredentialPayload | None,
    ) -> VenueSession: ...


@runtime_checkable
class VenueSession(Protocol):
    """All venue operations available after connecting one account."""

    connection_id: str

    @property
    def capabilities(self) -> VenueCapabilities: ...

    async def check_connection(self) -> None: ...

    async def list_instruments(self) -> tuple[Instrument, ...]: ...

    async def fetch_account(self) -> AccountSnapshot: ...

    async def fetch_fills(self, cursor: str | None) -> FillPage: ...

    async def fetch_funding(self, cursor: str | None) -> FundingPage: ...

    async def fetch_portfolio(self, pair: Pair) -> ConnectionPortfolioSnapshot: ...

    async def fetch_quote(self, pair: Pair) -> VenueQuote: ...

    async def normalize_amount(self, pair: Pair, base_amount: Decimal) -> Decimal: ...

    async def minimum_amount(
        self, pair: Pair, reference_price: Decimal, minimum_quote_notional: Decimal
    ) -> Decimal: ...

    async def place_order(self, intent: OrderIntent) -> NormalizedOrder: ...

    async def cancel_order(self, order_id: str, pair: Pair) -> None: ...

    async def find_order(
        self, pair: Pair, *, order_id: str | None = None, client_order_id: str | None = None
    ) -> NormalizedOrder | None: ...

    async def normalize_protection(self, spec: ProtectionSpec) -> ProtectionSpec: ...

    async def replace_protection(self, spec: ProtectionSpec) -> ProtectionState: ...

    async def cancel_protection(self, protection_ids: tuple[str, ...]) -> None: ...

    async def list_open_state(self, pair: Pair) -> OpenVenueState: ...

    async def close(self) -> None: ...
