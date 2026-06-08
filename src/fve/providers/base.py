"""The `OddsProvider` interface.

This is the single contract every data source satisfies, whether it is an
aggregator client (sharp + closing odds), The Odds API (soft books), or a
direct Kalshi / Polymarket client. Pricing logic depends on this interface
and on normalized `fve.types` — never on a provider's raw schema. Each
provider normalizes at its own boundary.

Design note (flagged, not silently decided): this interface is SYNCHRONOUS
for v0. A direct Kalshi/Polymarket client using WebSocket order books, or any
high-fan-out polling, will likely want an async variant. When we add a
streaming provider we should decide between (a) an async sibling Protocol,
(b) a sync facade over an async core, or (c) making the whole interface async.
Deferred until the first live provider — it does not affect the pricing core.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from fve.types import Market, MarketSnapshot, Venue


@runtime_checkable
class OddsProvider(Protocol):
    """A source of normalized markets and prices for one venue."""

    @property
    def venue(self) -> Venue:
        """The venue this provider reports for (carries sharp/soft classification)."""
        ...

    def list_markets(self, sport: str) -> Sequence[Market]:
        """Markets currently offered for a sport, normalized into `Market`.

        Implementations own pagination, rate-limit handling, and the mapping
        from the provider's event/market schema to stable `Market.key`s.
        """
        ...

    def fetch_quotes(self, market: Market) -> MarketSnapshot | None:
        """Current prices for one market, or None if the venue does not price it.

        The returned snapshot must cover every selection of `market`.
        """
        ...

    def fetch_closing(self, market: Market) -> MarketSnapshot | None:
        """The closing snapshot for a settled/closed market, if available.

        Closing odds are the CLV reference. Aggregators that retain closing
        lines return them here; venues without history return None.
        """
        ...
