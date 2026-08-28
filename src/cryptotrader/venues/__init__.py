"""Platform-neutral venue contracts and domain objects."""

from cryptotrader.venues.models import (
    ConnectionEnvironment,
    ConnectionPosition,
    NormalizedOrder,
    OpenVenueState,
    OrderIntent,
    ProtectionSpec,
    ProtectionState,
    VenueCapabilities,
    VenueConnection,
    VenueQuote,
)
from cryptotrader.venues.protocol import VenueAdapter, VenueSession
from cryptotrader.venues.registry import VenueAdapterRegistry

__all__ = [
    "ConnectionEnvironment",
    "ConnectionPosition",
    "NormalizedOrder",
    "OpenVenueState",
    "OrderIntent",
    "ProtectionSpec",
    "ProtectionState",
    "VenueAdapter",
    "VenueAdapterRegistry",
    "VenueCapabilities",
    "VenueConnection",
    "VenueQuote",
    "VenueSession",
]
