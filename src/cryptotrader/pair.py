"""Pair value object — single source of truth for trading pair semantics.

ccxt's unified symbol is the canonical form throughout the project:
- Spot: ``BASE/QUOTE`` (e.g. ``BTC/USDT``)
- Linear perp swap: ``BASE/QUOTE:SETTLE`` where settle = quote (e.g. ``BTC/USDT:USDT``)
- Inverse perp swap: ``BASE/QUOTE:SETTLE`` where settle = base (e.g. ``BTC/USD:BTC``)
- Futures (delivery): ``BASE/QUOTE:SETTLE-YYMMDD`` (e.g. ``BTC/USDT:USDT-241227``)
- Options: out of scope for spec 013 (raises ``NotImplementedError`` from ``from_ccxt``)

Spec: ``specs/013-pair-value-object/spec.md``
Contract: ``specs/013-pair-value-object/contracts/pair_api.md``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

MarketType = Literal["spot", "swap", "future", "option"]


@dataclass(frozen=True)
class Pair:
    """A trading pair keyed by ccxt unified symbol.

    Attributes:
        base: Base currency (e.g. ``"BTC"``).
        quote: Quote currency (e.g. ``"USDT"``).
        ccxt_symbol: ccxt unified symbol; spot has no suffix, derivatives use
            ``BASE/QUOTE:SETTLE`` form. Equal to ``canonical()``.

    Frozen dataclass: instances are immutable and hashable, suitable as dict
    keys (e.g. ``positions: dict[Pair, dict]``).

    Examples:
        >>> Pair.parse("BTC/USDT")
        Pair(base='BTC', quote='USDT', ccxt_symbol='BTC/USDT')
        >>> Pair.parse("BTC/USDT:USDT").market_type
        'swap'
        >>> Pair.parse("BTC/USDT:USDT").settle
        'USDT'
        >>> Pair.parse("BTC/USDT").canonical() == "BTC/USDT"
        True
    """

    base: str
    quote: str
    ccxt_symbol: str

    def __post_init__(self) -> None:
        if not self.base:
            raise ValueError("Pair.base must be non-empty")
        if not self.quote:
            raise ValueError("Pair.quote must be non-empty")
        if "/" not in self.ccxt_symbol:
            raise ValueError(f"Pair.ccxt_symbol must contain '/'; got {self.ccxt_symbol!r}")
        prefix = f"{self.base}/{self.quote}"
        if not self.ccxt_symbol.startswith(prefix):
            raise ValueError(f"Pair.ccxt_symbol must start with {prefix!r}; got {self.ccxt_symbol!r}")

    # ── Constructors ──────────────────────────────────────────────────────

    @classmethod
    def parse(cls, s: str) -> Pair:
        """Parse a canonical (ccxt unified) string into a ``Pair``.

        Examples:
            >>> Pair.parse("BTC/USDT")
            Pair(base='BTC', quote='USDT', ccxt_symbol='BTC/USDT')
            >>> Pair.parse("ETH/USDT:USDT").market_type
            'swap'

        Raises:
            ValueError: if ``s`` lacks ``/`` or has malformed structure.
        """
        if not s or "/" not in s:
            raise ValueError(f"Pair.parse: missing '/' in {s!r}")
        # Spot: "BTC/USDT" → base="BTC", quote="USDT"
        # Swap: "BTC/USDT:USDT" → base="BTC", quote="USDT", suffix="USDT"
        # Future: "BTC/USDT:USDT-241227" → quote tail before ':'
        head, _, _ = s.partition(":")
        try:
            base, quote = head.split("/", 1)
        except ValueError as exc:
            raise ValueError(f"Pair.parse: malformed head {head!r} in {s!r}") from exc
        if not base or not quote:
            raise ValueError(f"Pair.parse: empty base/quote in {s!r}")
        return cls(base=base, quote=quote, ccxt_symbol=s)

    @classmethod
    def from_ccxt(cls, exchange: Any, symbol: str) -> Pair:
        """Build a ``Pair`` from a ccxt exchange's market metadata.

        Falls back to ``parse(symbol)`` when ``exchange.market(symbol)`` raises
        or returns an empty/missing dict.

        Raises:
            NotImplementedError: when the market type is ``option``.
        """
        try:
            m = exchange.market(symbol)
        except Exception:  # pragma: no cover — narrow ccxt exception types vary
            logger.debug("Pair.from_ccxt: market() lookup failed for %r", symbol, exc_info=True)
            return cls.parse(symbol)
        if not m:
            return cls.parse(symbol)
        if m.get("option"):
            raise NotImplementedError(f"Option markets are out of scope for spec 013-pair-value-object; got {symbol!r}")
        base = m.get("base") or ""
        quote = m.get("quote") or ""
        if not (base and quote):
            return cls.parse(symbol)
        return cls(base=base, quote=quote, ccxt_symbol=symbol)

    # ── Serialization ─────────────────────────────────────────────────────

    def to_ccxt(self) -> str:
        """Return the ccxt unified symbol (≡ ``self.ccxt_symbol``)."""
        return self.ccxt_symbol

    def canonical(self) -> str:
        """Project canonical string form (≡ ``to_ccxt()``)."""
        return self.ccxt_symbol

    def display(self) -> str:
        """Human/AI friendly form.

        Examples:
            >>> Pair.parse("BTC/USDT").display()
            'BTC/USDT'
            >>> Pair.parse("BTC/USDT:USDT").display()
            'BTC/USDT (perp)'
        """
        mt = self.market_type
        bq = f"{self.base}/{self.quote}"
        if mt == "spot":
            return bq
        if mt == "swap":
            return f"{bq} (perp)"
        if mt == "future":
            # ``BTC/USDT:USDT-241227`` → expiry = ``241227``
            _, _, tail = self.ccxt_symbol.partition(":")
            _, _, expiry = tail.partition("-")
            return f"{bq} (futures{' ' + expiry if expiry else ''})"
        # option not constructable via from_ccxt; defensive fallback
        return self.ccxt_symbol  # pragma: no cover

    def __str__(self) -> str:
        return self.canonical()

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def market_type(self) -> MarketType:
        """Inferred from ``ccxt_symbol`` suffix shape.

        - No ``:`` → spot
        - ``:SETTLE`` (no further dash) → swap (perpetual)
        - ``:SETTLE-...`` → future (delivery)

        Options are not produced here; ``from_ccxt`` raises before reaching this.
        """
        if ":" not in self.ccxt_symbol:
            return "spot"
        _, _, tail = self.ccxt_symbol.partition(":")
        if "-" in tail:
            return "future"
        return "swap"

    @property
    def settle(self) -> str | None:
        """Settlement currency for derivatives; ``None`` for spot."""
        if ":" not in self.ccxt_symbol:
            return None
        _, _, tail = self.ccxt_symbol.partition(":")
        settle, _, _ = tail.partition("-")
        return settle or None


def market_type_for(pair: str) -> str:
    """Derive market_type from a pair str. Defaults to ``"spot"`` on parse failure
    so unknown legacy data never blocks a write — matches DB column DEFAULTs.

    Shared helper to avoid the duplicate definitions previously in
    portfolio/manager.py and journal/store.py (review-code A1).
    """
    try:
        return Pair.parse(pair).market_type
    except (ValueError, NotImplementedError):
        return "spot"
