"""Tests for the Kalshi provider.

All tests use an injected mock HTTP callable — no live network calls.
This tests the normalization boundary: raw Kalshi API dicts → clean fve types.

Kalshi price conventions (verified against v2 OpenAPI spec):
  yes_bid_dollars  — highest bid to buy YES  (string, e.g. "0.5400")
  yes_ask_dollars  — lowest ask to sell YES  (string, e.g. "0.5600")
  no_bid_dollars   — highest bid to buy NO   (string, e.g. "0.4200")
  no_ask_dollars   — lowest ask to sell NO   (string, e.g. "0.4400")
  last_price_dollars — last traded YES price (string, e.g. "0.5500")

All are in [0, 1] (probability space). Decimal odds = 1 / price.
Mid-price = (bid + ask) / 2 is used as the canonical price for the snapshot.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest

from fve.providers.kalshi import (
    KalshiProvider,
    _fp_to_float,
    _mid_decimal_odds,
    _price_to_decimal_odds,
)
from fve.types import MarketType, MarketSnapshot, VenueKind


# --------------------------------------------------------------------------- #
# Shared raw API responses (representative slices of real Kalshi v2 schema)
# --------------------------------------------------------------------------- #
def _raw_market(
    ticker: str = "KXELECT-25NOV-DEMA",
    event_ticker: str = "KXELECT-25NOV",
    status: str = "active",
    yes_bid: str = "0.5400",
    yes_ask: str = "0.5600",
    no_bid: str = "0.4200",
    no_ask: str = "0.4400",
    yes_bid_size: str = "120.00",
    yes_ask_size: str = "80.00",
    last_price: str = "0.5500",
    yes_sub: str = "Democrat wins",
    no_sub: str = "Republican wins",
    result: str = "",
    settlement_value: str | None = None,
    settlement_ts: str | None = None,
) -> dict:
    m = {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "market_type": "binary",
        "yes_sub_title": yes_sub,
        "no_sub_title": no_sub,
        "status": status,
        "yes_bid_dollars": yes_bid,
        "yes_ask_dollars": yes_ask,
        "no_bid_dollars": no_bid,
        "no_ask_dollars": no_ask,
        "yes_bid_size_fp": yes_bid_size,
        "yes_ask_size_fp": yes_ask_size,
        "last_price_dollars": last_price,
        "volume_fp": "1000.00",
        "open_interest_fp": "500.00",
        "result": result,
        "close_time": "2025-11-04T23:00:00Z",
        "rules_primary": "Market resolves YES if Democrat wins.",
        "rules_secondary": "",
    }
    if settlement_value is not None:
        m["settlement_value_dollars"] = settlement_value
    if settlement_ts is not None:
        m["settlement_ts"] = settlement_ts
    return m


def _make_provider(*raw_markets: dict, events: list | None = None) -> KalshiProvider:
    """Provider with a mock HTTP callable returning the given market(s)."""
    responses: list[dict] = []
    for rm in raw_markets:
        responses.append({"market": rm})  # single-market GET /markets/{ticker}
    # Also build a list response for list_markets
    list_resp = {"markets": list(raw_markets), "cursor": ""}

    call_count = [0]

    def mock_get(url: str, params: dict) -> dict:
        if "/markets/" in url:
            idx = call_count[0] % len(raw_markets)
            call_count[0] += 1
            return {"market": raw_markets[idx]}
        else:  # /markets or /events
            call_count[0] += 1
            return list_resp

    return KalshiProvider(_http_get=mock_get)


# =========================================================================== #
# Price conversion helpers
# =========================================================================== #

class TestPriceHelpers:
    def test_fp_to_float_normal(self):
        assert math.isclose(_fp_to_float("0.5600"), 0.56, abs_tol=1e-9)

    def test_fp_to_float_zero(self):
        assert _fp_to_float("0.0000") is None

    def test_fp_to_float_one(self):
        assert math.isclose(_fp_to_float("1.0000"), 1.0, abs_tol=1e-9)

    def test_price_to_decimal_odds(self):
        # 0.56 probability → decimal odds 1/0.56 ≈ 1.7857
        assert math.isclose(_price_to_decimal_odds("0.5600"), 1.0 / 0.56, abs_tol=1e-9)

    def test_price_to_decimal_odds_round_trip(self):
        for p_str, p in [("0.3000", 0.30), ("0.7500", 0.75), ("0.1000", 0.10)]:
            d = _price_to_decimal_odds(p_str)
            assert math.isclose(d, 1.0 / p, abs_tol=1e-9)

    def test_mid_decimal_odds(self):
        # bid=0.54, ask=0.56 → mid=0.55 → decimal odds=1/0.55
        assert math.isclose(_mid_decimal_odds("0.5400", "0.5600"), 1.0 / 0.55, abs_tol=1e-9)

    def test_mid_decimal_odds_equal_bid_ask(self):
        assert math.isclose(_mid_decimal_odds("0.5000", "0.5000"), 2.0, abs_tol=1e-9)


# =========================================================================== #
# venue property
# =========================================================================== #

class TestVenue:
    def test_venue_is_prediction_market(self):
        p = _make_provider(_raw_market())
        assert p.venue.key == "kalshi"
        assert p.venue.kind == VenueKind.PREDICTION_MARKET
        assert not p.venue.is_sharp


# =========================================================================== #
# fetch_quotes normalization
# =========================================================================== #

class TestFetchQuotes:
    def test_returns_snapshot_for_active_market(self):
        p = _make_provider(_raw_market(status="active"))
        from fve.types import Market, MarketType, Selection
        market = Market(
            key="KXELECT-25NOV-DEMA",
            sport="KXELECT",
            event_key="KXELECT-25NOV",
            type=MarketType.BINARY,
            selections=(Selection("yes", "Democrat wins"), Selection("no", "Republican wins")),
        )
        snap = p.fetch_quotes(market)
        assert snap is not None
        assert isinstance(snap, MarketSnapshot)

    def test_returns_none_for_settled_market(self):
        p = _make_provider(_raw_market(status="finalized", result="yes"))
        from fve.types import Market, MarketType, Selection
        market = Market(
            key="KXELECT-25NOV-DEMA",
            sport="KXELECT",
            event_key="KXELECT-25NOV",
            type=MarketType.BINARY,
            selections=(Selection("yes", "Democrat wins"), Selection("no", "Republican wins")),
        )
        assert p.fetch_quotes(market) is None

    def test_yes_price_uses_mid(self):
        # bid=0.54, ask=0.56 → mid=0.55 → decimal odds=1/0.55
        p = _make_provider(_raw_market(yes_bid="0.5400", yes_ask="0.5600"))
        from fve.types import Market, MarketType, Selection
        market = Market(
            key="KXELECT-25NOV-DEMA",
            sport="KXELECT",
            event_key="KXELECT-25NOV",
            type=MarketType.BINARY,
            selections=(Selection("yes", "Democrat wins"), Selection("no", "Republican wins")),
        )
        snap = p.fetch_quotes(market)
        assert snap is not None
        yes_price = snap.prices["yes"]
        assert math.isclose(yes_price.decimal_odds, 1.0 / 0.55, abs_tol=1e-9)

    def test_no_price_uses_mid(self):
        # no_bid=0.42, no_ask=0.44 → mid=0.43 → decimal odds=1/0.43
        p = _make_provider(_raw_market(no_bid="0.4200", no_ask="0.4400"))
        from fve.types import Market, MarketType, Selection
        market = Market(
            key="KXELECT-25NOV-DEMA",
            sport="KXELECT",
            event_key="KXELECT-25NOV",
            type=MarketType.BINARY,
            selections=(Selection("yes", "Democrat wins"), Selection("no", "Republican wins")),
        )
        snap = p.fetch_quotes(market)
        assert snap is not None
        no_price = snap.prices["no"]
        assert math.isclose(no_price.decimal_odds, 1.0 / 0.43, abs_tol=1e-9)

    def test_yes_bid_ask_stored_on_price(self):
        # bid=1/yes_ask (best back odds for YES buyer), ask=1/yes_bid
        p = _make_provider(_raw_market(yes_bid="0.5400", yes_ask="0.5600"))
        from fve.types import Market, MarketType, Selection
        market = Market(
            key="KXELECT-25NOV-DEMA", sport="KXELECT", event_key="KXELECT-25NOV",
            type=MarketType.BINARY,
            selections=(Selection("yes", "Democrat wins"), Selection("no", "Republican wins")),
        )
        snap = p.fetch_quotes(market)
        yes_p = snap.prices["yes"]
        # bid (best back) = 1/yes_ask = 1/0.56; ask (best lay) = 1/yes_bid = 1/0.54
        assert math.isclose(yes_p.bid, 1.0 / 0.56, abs_tol=1e-9)
        assert math.isclose(yes_p.ask, 1.0 / 0.54, abs_tol=1e-9)

    def test_liquidity_size_stored(self):
        p = _make_provider(_raw_market(yes_bid_size="120.00", yes_ask_size="80.00"))
        from fve.types import Market, MarketType, Selection
        market = Market(
            key="KXELECT-25NOV-DEMA", sport="KXELECT", event_key="KXELECT-25NOV",
            type=MarketType.BINARY,
            selections=(Selection("yes", "Democrat wins"), Selection("no", "Republican wins")),
        )
        snap = p.fetch_quotes(market)
        yes_p = snap.prices["yes"]
        assert math.isclose(yes_p.size, 120.0, abs_tol=1e-9)

    def test_snapshot_covers_both_selections(self):
        p = _make_provider(_raw_market())
        from fve.types import Market, MarketType, Selection
        market = Market(
            key="KXELECT-25NOV-DEMA", sport="KXELECT", event_key="KXELECT-25NOV",
            type=MarketType.BINARY,
            selections=(Selection("yes", "Democrat wins"), Selection("no", "Republican wins")),
        )
        snap = p.fetch_quotes(market)
        assert "yes" in snap.prices
        assert "no" in snap.prices

    def test_snapshot_not_marked_closing(self):
        p = _make_provider(_raw_market(status="active"))
        from fve.types import Market, MarketType, Selection
        market = Market(
            key="KXELECT-25NOV-DEMA", sport="KXELECT", event_key="KXELECT-25NOV",
            type=MarketType.BINARY,
            selections=(Selection("yes", "Democrat wins"), Selection("no", "Republican wins")),
        )
        snap = p.fetch_quotes(market)
        assert snap is not None
        assert snap.is_closing is False

    def test_falls_back_to_last_price_when_book_empty(self):
        # Zero bid/ask → fall back to last_price_dollars for the mid
        p = _make_provider(_raw_market(
            yes_bid="0.0000", yes_ask="0.0000", last_price="0.5500"
        ))
        from fve.types import Market, MarketType, Selection
        market = Market(
            key="KXELECT-25NOV-DEMA", sport="KXELECT", event_key="KXELECT-25NOV",
            type=MarketType.BINARY,
            selections=(Selection("yes", "Democrat wins"), Selection("no", "Republican wins")),
        )
        snap = p.fetch_quotes(market)
        assert snap is not None
        assert math.isclose(snap.prices["yes"].decimal_odds, 1.0 / 0.55, abs_tol=1e-9)


# =========================================================================== #
# fetch_closing normalization
# =========================================================================== #

class TestFetchClosing:
    def _settled_market(self):
        from fve.types import Market, MarketType, Selection
        return Market(
            key="KXELECT-25NOV-DEMA", sport="KXELECT", event_key="KXELECT-25NOV",
            type=MarketType.BINARY,
            selections=(Selection("yes", "Democrat wins"), Selection("no", "Republican wins")),
        )

    def test_returns_none_for_active_market(self):
        p = _make_provider(_raw_market(status="active"))
        assert p.fetch_closing(self._settled_market()) is None

    def test_returns_closing_snapshot_for_settled(self):
        p = _make_provider(_raw_market(
            status="finalized", result="yes",
            last_price="0.8200",
            settlement_ts="2025-11-05T00:00:00Z",
        ))
        snap = p.fetch_closing(self._settled_market())
        assert snap is not None
        assert snap.is_closing is True

    def test_closing_uses_last_price(self):
        # last_price=0.82 → YES decimal odds=1/0.82, NO=1/(1-0.82)=1/0.18
        p = _make_provider(_raw_market(
            status="finalized", result="yes", last_price="0.8200",
        ))
        snap = p.fetch_closing(self._settled_market())
        assert snap is not None
        assert math.isclose(snap.prices["yes"].decimal_odds, 1.0 / 0.82, abs_tol=1e-9)
        assert math.isclose(snap.prices["no"].decimal_odds, 1.0 / 0.18, abs_tol=1e-9)

    def test_closing_snapshot_marked_is_closing(self):
        p = _make_provider(_raw_market(status="determined", result="no", last_price="0.3000"))
        snap = p.fetch_closing(self._settled_market())
        assert snap is not None
        assert snap.is_closing is True

    def test_all_terminal_statuses_return_closing(self):
        from fve.types import Market, MarketType, Selection
        market = self._settled_market()
        for status in ("determined", "finalized", "settled"):
            p = _make_provider(_raw_market(status=status, result="yes", last_price="0.7000"))
            snap = p.fetch_closing(market)
            assert snap is not None, f"status={status} should return closing snapshot"
            assert snap.is_closing is True


# =========================================================================== #
# list_markets normalization
# =========================================================================== #

class TestListMarkets:
    def test_returns_market_objects(self):
        p = _make_provider(_raw_market())
        markets = p.list_markets("KXELECT")
        assert len(markets) == 1
        assert markets[0].key == "KXELECT-25NOV-DEMA"

    def test_market_type_is_binary(self):
        p = _make_provider(_raw_market())
        market = p.list_markets("KXELECT")[0]
        assert market.type == MarketType.BINARY

    def test_market_has_yes_and_no_selections(self):
        p = _make_provider(_raw_market(yes_sub="Dem wins", no_sub="Rep wins"))
        market = p.list_markets("KXELECT")[0]
        sel_keys = {s.key for s in market.selections}
        assert sel_keys == {"yes", "no"}

    def test_selection_names_from_subtitles(self):
        p = _make_provider(_raw_market(yes_sub="Democrat wins", no_sub="Republican wins"))
        market = p.list_markets("KXELECT")[0]
        sel_map = {s.key: s.name for s in market.selections}
        assert sel_map["yes"] == "Democrat wins"
        assert sel_map["no"] == "Republican wins"

    def test_market_event_key(self):
        p = _make_provider(_raw_market(event_ticker="KXELECT-25NOV"))
        market = p.list_markets("KXELECT")[0]
        assert market.event_key == "KXELECT-25NOV"

    def test_sport_stored_on_market(self):
        p = _make_provider(_raw_market())
        market = p.list_markets("KXELECT")[0]
        assert market.sport == "KXELECT"

    def test_multiple_markets_returned(self):
        raw1 = _raw_market(ticker="MARKET-A", event_ticker="EVENT-A")
        raw2 = _raw_market(ticker="MARKET-B", event_ticker="EVENT-B")
        # Override mock to return both in list
        list_resp = {"markets": [raw1, raw2], "cursor": ""}
        def mock_get(url, params):
            return list_resp
        p = KalshiProvider(_http_get=mock_get)
        markets = p.list_markets("KXELECT")
        assert len(markets) == 2
        keys = {m.key for m in markets}
        assert keys == {"MARKET-A", "MARKET-B"}
