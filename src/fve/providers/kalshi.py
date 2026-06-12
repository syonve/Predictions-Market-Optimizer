"""Kalshi prediction-market provider.

Implements `OddsProvider` against the Kalshi REST API v2.
API base: https://external-api.kalshi.com/trade-api/v2

Price convention (Kalshi v2)
----------------------------
All prices are `FixedPointDollars` strings in [0, 1] representing the dollar
cost of a YES contract that pays $1 at settlement. This is equivalent to an
implied probability:

    yes_bid_dollars = "0.5400"  →  56% ← NO: 54% bid for YES
    yes_ask_dollars = "0.5600"  →  56% ask for YES

Decimal-odds conversion: odds = 1 / price.

Canonical price for the snapshot is the mid-point of the YES bid/ask spread:

    mid = (yes_bid + yes_ask) / 2

If the order book is empty (bid = ask = 0), we fall back to `last_price_dollars`.
The `Price.bid` / `Price.ask` fields store the best-available execution prices:

    Price.bid (best back)  = 1 / yes_ask   ← what a YES buyer pays
    Price.ask (best lay)   = 1 / yes_bid   ← what a YES seller receives

The NO selection mirrors the YES side using no_bid / no_ask.

Authentication
--------------
Market-data endpoints are **public** (no API key required). An optional
`api_key` is accepted for future authenticated endpoints (order placement,
portfolio).

HTTP injection
--------------
The `_http_get` constructor argument replaces the default urllib transport.
Pass a callable `(url: str, params: dict[str, str]) -> dict` to inject
a mock in tests — no monkey-patching required.

Flagged decisions
-----------------
1. **Closing snapshot uses last_price_dollars.** This is the most recent trade
   price and the closest available proxy for "where the market closed." An
   alternative is a volume-weighted average of the final N trades, but Kalshi
   does not expose this directly; last_price is sufficient for CLV calculation.

2. **sport → series_ticker mapping.** The `list_markets(sport)` interface maps
   `sport` directly to Kalshi's `series_ticker` query parameter. Callers must
   pass the Kalshi series ticker (e.g. "KXELECTIONS", "KXFED"). A named-
   constant mapping can be added once the target categories are confirmed.

3. **Binary markets only.** Scalar markets (continuous outcomes) are skipped
   in `list_markets`. They require a different selection model and are out of
   scope for the current binary-EV engine.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from fve.providers.base import OddsProvider
from fve.types import (
    Market,
    MarketSnapshot,
    MarketType,
    Price,
    Selection,
    Venue,
    VenueKind,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

KALSHI_VENUE = Venue(
    key="kalshi",
    name="Kalshi",
    kind=VenueKind.PREDICTION_MARKET,
)

# Statuses that indicate the market has stopped trading and a closing
# snapshot can be built from last_price_dollars.
_TERMINAL_STATUSES = frozenset({"determined", "finalized", "settled"})


# --------------------------------------------------------------------------- #
# Pure normalization helpers (module-level for direct testability)
# --------------------------------------------------------------------------- #
def _fp_to_float(price_str: str) -> float | None:
    """Parse a Kalshi FixedPointDollars string to float, or None if zero/empty."""
    try:
        v = float(price_str)
    except (ValueError, TypeError):
        return None
    return v if v > 0.0 else None


def _price_to_decimal_odds(price_str: str) -> float:
    """Convert a Kalshi price string to decimal odds (1 / price)."""
    v = _fp_to_float(price_str)
    if v is None or v <= 0.0 or v >= 1.0:
        raise ValueError(f"price_str {price_str!r} is not a valid probability in (0,1)")
    return 1.0 / v


def _mid_decimal_odds(bid_str: str, ask_str: str) -> float:
    """Decimal odds for the mid-price of a bid/ask pair."""
    bid = _fp_to_float(bid_str) or 0.0
    ask = _fp_to_float(ask_str) or 0.0
    mid = (bid + ask) / 2.0
    return 1.0 / mid


def _parse_ts(ts_str: str | None) -> datetime:
    """Parse an ISO-8601 timestamp string to a UTC datetime."""
    if not ts_str:
        return datetime.now(tz=timezone.utc)
    # Python 3.11+ handles 'Z' suffix natively; strip for 3.10 compat
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def _build_price_for_side(
    bid_str: str,
    ask_str: str,
    size_str: str | None,
    fallback_str: str,
) -> Price:
    """Build a Price for one side (YES or NO) using mid as canonical odds.

    Falls back to last_price / (1-last_price) when the book is empty.
    bid field  = 1/ask_str  (best odds to BUY this side)
    ask field  = 1/bid_str  (best odds to SELL this side)
    """
    bid_f = _fp_to_float(bid_str)
    ask_f = _fp_to_float(ask_str)

    if bid_f and ask_f:
        mid = (bid_f + ask_f) / 2.0
        back_odds = 1.0 / ask_f   # buyer pays ask → decimal odds = 1/ask
        lay_odds  = 1.0 / bid_f   # seller receives bid → decimal odds = 1/bid
    elif bid_f:
        mid = bid_f
        back_odds = 1.0 / bid_f
        lay_odds  = 1.0 / bid_f
    elif ask_f:
        mid = ask_f
        back_odds = 1.0 / ask_f
        lay_odds  = 1.0 / ask_f
    else:
        # Empty book — fall back to last traded price
        fb = _fp_to_float(fallback_str)
        if fb is None or fb <= 0.0 or fb >= 1.0:
            raise ValueError(
                f"Empty order book and unusable fallback price {fallback_str!r}"
            )
        mid = fb
        back_odds = 1.0 / fb
        lay_odds  = 1.0 / fb

    size = float(size_str) if size_str else None

    return Price(
        decimal_odds=1.0 / mid,
        bid=back_odds,
        ask=lay_odds,
        size=size,
    )


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #
class KalshiProvider:
    """OddsProvider for the Kalshi prediction market REST API v2.

    Parameters
    ----------
    api_key:
        Optional API key for authenticated endpoints. Not required for
        market-data (read-only) access.
    base_url:
        Override the API base URL (e.g. for the demo environment:
        ``https://external-api.demo.kalshi.co/trade-api/v2``).
    _http_get:
        Injectable HTTP callable ``(url, params) -> dict`` for testing.
        If None, uses the default urllib implementation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _BASE_URL,
        _http_get: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http_get = _http_get if _http_get is not None else self._urllib_get

    # --- OddsProvider interface ---

    @property
    def venue(self) -> Venue:
        return KALSHI_VENUE

    def list_markets(self, sport: str) -> Sequence[Market]:
        """Return all open binary markets for a Kalshi series ticker.

        Parameters
        ----------
        sport:
            Kalshi `series_ticker` (e.g. ``"KXELECTIONS"``, ``"KXFED"``).
            Pass an empty string to list all open markets (may be slow).
        """
        markets: list[Market] = []
        # NOTE: the query filter value is "open", but returned market objects
        # report status "active" — verified against the live API 2026-06-09.
        # Passing status=active silently returns zero markets.
        params: dict[str, str] = {"status": "open", "limit": "1000", "mve_filter": "exclude"}
        if sport:
            params["series_ticker"] = sport

        while True:
            data = self._get("/markets", params)
            for raw in data.get("markets", []):
                if raw.get("market_type") != "binary":
                    continue  # skip scalar markets
                try:
                    markets.append(self._normalize_market(raw, sport))
                except Exception:
                    pass  # skip malformed entries
            cursor = data.get("cursor", "")
            if not cursor:
                break
            params["cursor"] = cursor

        return markets

    def fetch_quotes(self, market: Market) -> MarketSnapshot | None:
        """Fetch live prices for one market. Returns None if no longer active."""
        data = self._get(f"/markets/{market.key}")
        raw = data.get("market", data)
        if raw.get("status") not in ("active", "unopened", "paused"):
            return None
        return self._normalize_snapshot(raw, market, is_closing=False)

    def fetch_closing(self, market: Market) -> MarketSnapshot | None:
        """Fetch the closing snapshot for a settled market.

        Returns None if the market has not yet reached a terminal status.
        Uses `last_price_dollars` as the closing price (see module docstring).
        """
        data = self._get(f"/markets/{market.key}")
        raw = data.get("market", data)
        if raw.get("status") not in _TERMINAL_STATUSES:
            return None
        return self._normalize_snapshot(raw, market, is_closing=True)

    # --- Normalization ---

    def _normalize_market(self, raw: dict, sport: str) -> Market:
        ticker       = raw["ticker"]
        event_ticker = raw["event_ticker"]
        yes_sub      = raw.get("yes_sub_title") or "Yes"
        no_sub       = raw.get("no_sub_title") or "No"

        return Market(
            key=ticker,
            sport=sport,
            event_key=event_ticker,
            type=MarketType.BINARY,
            selections=(
                Selection("yes", yes_sub),
                Selection("no",  no_sub),
            ),
        )

    def _normalize_snapshot(
        self,
        raw: dict,
        market: Market,
        is_closing: bool,
    ) -> MarketSnapshot:
        ts_field = "settlement_ts" if is_closing else "updated_time"
        ts_str   = raw.get(ts_field) or raw.get("updated_time")
        timestamp = _parse_ts(ts_str)

        last = raw.get("last_price_dollars", "0.0000")

        if is_closing:
            # Use last traded price as the closing probability reference
            last_f = _fp_to_float(last) or 0.5
            last_f = max(0.0001, min(0.9999, last_f))  # guard against 0/1 extremes
            yes_price = Price(decimal_odds=1.0 / last_f)
            no_price  = Price(decimal_odds=1.0 / (1.0 - last_f))
        else:
            yes_price = _build_price_for_side(
                bid_str=raw.get("yes_bid_dollars", "0.0000"),
                ask_str=raw.get("yes_ask_dollars", "0.0000"),
                size_str=raw.get("yes_bid_size_fp"),
                fallback_str=last,
            )
            # NO side: bid/ask are no_bid/no_ask; fallback is (1 - last_price)
            last_f = _fp_to_float(last) or 0.5
            no_fallback = str(round(1.0 - last_f, 4))
            no_price = _build_price_for_side(
                bid_str=raw.get("no_bid_dollars", "0.0000"),
                ask_str=raw.get("no_ask_dollars", "0.0000"),
                size_str=None,
                fallback_str=no_fallback,
            )

        return MarketSnapshot(
            market=market,
            venue=KALSHI_VENUE,
            timestamp=timestamp,
            prices={"yes": yes_price, "no": no_price},
            is_closing=is_closing,
        )

    # --- HTTP ---

    def _get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        return self._http_get(url, params or {})

    def _urllib_get(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
