"""Normalized, provider-agnostic domain types.

Every provider normalizes into these types at its boundary. Pricing, signal,
sizing, and tracking code depends ONLY on these — never on a provider's schema.

Canonical odds representation is decimal odds (payout multiplier incl. stake,
strictly > 1.0). Helpers convert to/from American odds and probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# --------------------------------------------------------------------------- #
# Odds conversions (pure)
# --------------------------------------------------------------------------- #
def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds. e.g. -110 -> 1.9091, +150 -> 2.5."""
    if american == 0:
        raise ValueError("American odds cannot be 0")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds (> 1) to American odds."""
    if decimal <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal}")
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def decimal_to_prob(decimal: float) -> float:
    """Raw implied probability of decimal odds (includes the vig)."""
    if decimal <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal}")
    return 1.0 / decimal


def prob_to_decimal(prob: float) -> float:
    """Fair decimal odds for a probability in (0, 1)."""
    if not 0.0 < prob < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {prob}")
    return 1.0 / prob


# --------------------------------------------------------------------------- #
# Venues
# --------------------------------------------------------------------------- #
class VenueKind(Enum):
    """How a venue is treated when forming fair value.

    SHARP and EXCHANGE / deep PREDICTION_MARKET books anchor consensus;
    SOFT books are down-weighted (they often follow the sharps).
    MODEL is a synthetic "venue" produced by an independent model (e.g. a
    polling aggregator). It is weighted below PREDICTION_MARKET and layered
    in only where sharp consensus is weak or absent.
    """

    SHARP = "sharp"  # e.g. Pinnacle
    EXCHANGE = "exchange"  # e.g. Betfair (peer-to-peer, low margin)
    PREDICTION_MARKET = "prediction_market"  # e.g. Kalshi, Polymarket
    SOFT = "soft"  # recreational sportsbooks
    MODEL = "model"  # independent model estimate (polling, power ratings, etc.)


@dataclass(frozen=True, slots=True)
class Venue:
    key: str  # stable machine id, e.g. "pinnacle"
    name: str  # display name
    kind: VenueKind

    @property
    def is_sharp(self) -> bool:
        """Whether this venue should anchor fair value (vs. merely inform it)."""
        return self.kind in (VenueKind.SHARP, VenueKind.EXCHANGE)


# --------------------------------------------------------------------------- #
# Markets & selections
# --------------------------------------------------------------------------- #
class MarketType(Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    PROP = "prop"
    OUTRIGHT = "outright"
    BINARY = "binary"  # prediction-market yes/no contract


@dataclass(frozen=True, slots=True)
class Selection:
    """One mutually-exclusive outcome within a market."""

    key: str  # stable id within the market, e.g. "home", "over", "yes"
    name: str  # display name, e.g. "Lakers", "Over 220.5"


@dataclass(frozen=True, slots=True)
class Market:
    """A set of mutually-exclusive, collectively-exhaustive selections.

    The selections of a single market are what devigging operates over.
    """

    key: str  # stable id, e.g. "nba:2026-06-06:LAL@BOS:moneyline"
    sport: str
    event_key: str
    type: MarketType
    selections: tuple[Selection, ...]
    line: float | None = None  # spread/total line; None for moneyline/binary

    def __post_init__(self) -> None:
        if len(self.selections) < 2:
            raise ValueError("a market needs at least 2 selections")
        keys = [s.key for s in self.selections]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate selection keys in market {self.key}")


# --------------------------------------------------------------------------- #
# Prices & snapshots
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Price:
    """A single quoted price for a selection at a venue.

    `decimal_odds` is canonical. Order-book venues may also carry best
    bid/ask and the size available at the top of book (used to weight
    deep prediction-market books toward the sharp anchor).
    """

    decimal_odds: float
    bid: float | None = None  # exchange/order-book best back/bid (decimal)
    ask: float | None = None  # exchange/order-book best lay/ask (decimal)
    size: float | None = None  # liquidity at top of book, venue currency/contracts

    def __post_init__(self) -> None:
        if self.decimal_odds <= 1.0:
            raise ValueError(f"decimal_odds must be > 1.0, got {self.decimal_odds}")

    @property
    def implied_prob(self) -> float:
        return 1.0 / self.decimal_odds

    @classmethod
    def from_american(cls, american: float, **kw: float | None) -> Price:
        return cls(decimal_odds=american_to_decimal(american), **kw)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """All prices for one market at one venue at one instant.

    `prices` maps selection.key -> Price and must cover every selection of
    the market. `is_closing` marks the snapshot taken at market close, the
    reference used for CLV.
    """

    market: Market
    venue: Venue
    timestamp: datetime
    prices: dict[str, Price] = field(default_factory=dict)
    is_closing: bool = False

    def __post_init__(self) -> None:
        market_keys = {s.key for s in self.market.selections}
        price_keys = set(self.prices)
        if price_keys != market_keys:
            missing = market_keys - price_keys
            extra = price_keys - market_keys
            raise ValueError(
                f"snapshot prices must match market selections "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )

    def decimal_odds(self) -> tuple[float, ...]:
        """Decimal odds in market selection order (the order devig expects)."""
        return tuple(self.prices[s.key].decimal_odds for s in self.market.selections)
