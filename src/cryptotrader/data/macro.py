"""Macro data collector placeholder."""

from __future__ import annotations

from cryptotrader.models import MacroData


class MacroCollector:

    async def collect(self) -> MacroData:
        return MacroData()
