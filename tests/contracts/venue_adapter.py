"""Reusable structural assertions for venue adapter implementations."""

from __future__ import annotations

from inspect import iscoroutinefunction
from typing import TYPE_CHECKING

from cryptotrader.venues.models import ConnectionEnvironment, VenueCapabilities
from cryptotrader.venues.protocol import VenueAdapter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def assert_venue_contract(
    adapter_factory: Callable[[], VenueAdapter],
    environments: Iterable[ConnectionEnvironment],
) -> None:
    """Check the adapter surface shared by Paper and CCXT implementations."""
    adapter = adapter_factory()
    assert isinstance(adapter, VenueAdapter)
    assert adapter.adapter_id.strip()
    assert iscoroutinefunction(adapter.connect), "connect must be async"
    for environment in environments:
        capabilities = adapter.capabilities(environment)
        assert isinstance(capabilities, VenueCapabilities)
