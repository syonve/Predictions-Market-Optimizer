"""Tests for PollingModelProvider and ModelEstimate (models/polling.py, models/base.py)."""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone

from fve.models.base import MODEL_VENUE, ModelEstimate
from fve.models.poll import AggregationConfig, Methodology, Poll, Population
from fve.models.polling import PollingModelProvider
from fve.types import Market, MarketType, Selection, VenueKind

TODAY = date(2026, 1, 15)
REF_DT = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _binary_market(key: str = "KXELECT-2026-PRES-YES") -> Market:
    return Market(
        key=key,
        sport="KXELECTIONS",
        event_key="ELECT-2026-PRES",
        type=MarketType.BINARY,
        selections=(
            Selection("yes", "Candidate A wins"),
            Selection("no", "Candidate A does not win"),
        ),
    )


def _poll(
    market_key: str = "KXELECT-2026-PRES-YES",
    selection_key: str = "yes",
    prob: float = 0.52,
    n: int = 1000,
    field_end: date = TODAY,
    pollster: str = "Generic",
) -> Poll:
    return Poll(
        market_key=market_key,
        selection_key=selection_key,
        prob=prob,
        n=n,
        field_end=field_end,
        pollster=pollster,
        methodology=Methodology.LIVE_PHONE,
        population=Population.LIKELY_VOTERS,
    )


# --------------------------------------------------------------------------- #
# PollingModelProvider.estimate()
# --------------------------------------------------------------------------- #
def test_estimate_returns_none_for_unknown_market():
    """No polls for the market → estimate returns None."""
    provider = PollingModelProvider(polls=[_poll(market_key="OTHER")])
    market = _binary_market("KXELECT-2026-PRES-YES")
    result = provider.estimate(market)
    assert result is None


def test_estimate_returns_model_estimate():
    """Single poll → returns a ModelEstimate."""
    market = _binary_market()
    provider = PollingModelProvider(
        polls=[_poll(market_key=market.key, prob=0.55)],
        _reference_date=TODAY,
    )
    result = provider.estimate(market)
    assert result is not None
    assert isinstance(result, ModelEstimate)


def test_estimate_probs_sum_to_one():
    """YES and NO probabilities sum exactly to 1.0."""
    market = _binary_market()
    provider = PollingModelProvider(
        polls=[_poll(market_key=market.key, prob=0.55)],
        _reference_date=TODAY,
    )
    result = provider.estimate(market)
    assert result is not None
    total = sum(result.selection_probs.values())
    assert abs(total - 1.0) < 1e-9


def test_estimate_yes_prob_from_single_poll():
    """Single poll at 0.55 → YES prob is 0.55, NO is 0.45 after normalization.

    For a binary market with a single YES poll, the YES prob is the raw poll
    value; NO is computed as residual (1 - 0.55 = 0.45). After normalization
    both sum to 1.0 with no change.
    """
    market = _binary_market()
    provider = PollingModelProvider(
        polls=[_poll(market_key=market.key, prob=0.55)],
        _reference_date=TODAY,
    )
    result = provider.estimate(market)
    assert result is not None
    assert abs(result.prob("yes") - 0.55) < 1e-9
    assert abs(result.prob("no") - 0.45) < 1e-9


def test_estimate_model_name():
    """Model name is 'polling_aggregate'."""
    market = _binary_market()
    provider = PollingModelProvider(
        polls=[_poll(market_key=market.key)],
        _reference_date=TODAY,
    )
    result = provider.estimate(market)
    assert result is not None
    assert result.model_name == "polling_aggregate"


def test_estimate_n_samples_counts_matching_polls():
    """n_samples counts polls that matched this market key."""
    market = _binary_market()
    polls = [
        _poll(market_key=market.key, prob=0.50),
        _poll(market_key=market.key, prob=0.54),
        _poll(market_key="OTHER", prob=0.60),
    ]
    provider = PollingModelProvider(polls=polls, _reference_date=TODAY)
    result = provider.estimate(market)
    assert result is not None
    assert result.n_samples == 2


def test_estimate_confidence_increases_with_more_polls():
    """More polls of equal size → higher effective_n → higher confidence."""
    market = _binary_market()
    polls_1 = [_poll(market_key=market.key)]
    polls_5 = [_poll(market_key=market.key) for _ in range(5)]

    r1 = PollingModelProvider(polls=polls_1, _reference_date=TODAY).estimate(market)
    r5 = PollingModelProvider(polls=polls_5, _reference_date=TODAY).estimate(market)
    assert r1 is not None and r5 is not None
    assert r5.confidence > r1.confidence


def test_estimate_confidence_in_unit_interval():
    """Confidence is always in [0, 1]."""
    market = _binary_market()
    polls = [_poll(market_key=market.key) for _ in range(20)]
    result = PollingModelProvider(polls=polls, _reference_date=TODAY).estimate(market)
    assert result is not None
    assert 0.0 <= result.confidence <= 1.0


def test_estimate_weighted_average_two_polls():
    """Two polls with known weights → weighted YES prob is correct.

    Poll A: n=4000 (size_weight=2.0), prob=0.60, same-day (time_weight=1.0)
    Poll B: n=1000 (size_weight=1.0), prob=0.40, same-day

    Composite weights (both LV, quality=1.0): w_A=2.0, w_B=1.0
    Weighted YES = (2.0*0.60 + 1.0*0.40) / 3.0 = 1.60/3.0 ≈ 0.5333
    NO = 1 - YES (residual imputation).
    After normalization: YES≈0.5333/(0.5333+0.4667)=0.5333 (already sums to 1)
    """
    market = _binary_market()
    polls = [
        _poll(market_key=market.key, prob=0.60, n=4000),
        _poll(market_key=market.key, prob=0.40, n=1000),
    ]
    result = PollingModelProvider(polls=polls, _reference_date=TODAY).estimate(market)
    assert result is not None
    expected_yes = (2.0 * 0.60 + 1.0 * 0.40) / 3.0
    assert abs(result.prob("yes") - expected_yes) < 1e-6
    assert abs(result.prob("no") - (1.0 - expected_yes)) < 1e-6


def test_estimate_house_effect_applied():
    """House effect is forwarded from AggregationConfig."""
    market = _binary_market()
    polls = [_poll(market_key=market.key, prob=0.55, pollster="BiasedPollster")]
    config = AggregationConfig(house_effects={"BiasedPollster": 0.05})
    result = PollingModelProvider(polls=polls, config=config, _reference_date=TODAY).estimate(market)
    assert result is not None
    # 0.55 - 0.05 = 0.50 YES; 0.50 NO → after normalization both 0.50
    assert abs(result.prob("yes") - 0.50) < 1e-9
    assert abs(result.prob("no") - 0.50) < 1e-9


def test_estimate_reference_date_injectable():
    """Injected reference date produces deterministic time-weights."""
    market = _binary_market()
    old_date = date(2025, 11, 1)  # polls will appear very stale
    recent_date = date(2026, 1, 14)  # polls will appear fresh

    # Both runs use the same polls with field_end = 2026-01-14
    polls = [_poll(market_key=market.key, prob=0.60, n=1000, field_end=date(2026, 1, 14))]

    r_old = PollingModelProvider(polls=polls, _reference_date=old_date).estimate(market)
    r_recent = PollingModelProvider(polls=polls, _reference_date=recent_date).estimate(market)
    assert r_old is not None and r_recent is not None
    # Confidence should be lower when reference date makes polls look stale
    assert r_recent.confidence >= r_old.confidence


# --------------------------------------------------------------------------- #
# ModelEstimate.as_snapshot()
# --------------------------------------------------------------------------- #
def test_as_snapshot_returns_market_snapshot():
    """as_snapshot() returns a MarketSnapshot with prices for all selections."""
    from fve.types import MarketSnapshot
    market = _binary_market()
    result = PollingModelProvider(
        polls=[_poll(market_key=market.key, prob=0.60)],
        _reference_date=TODAY,
    ).estimate(market)
    assert result is not None
    snap = result.as_snapshot()
    assert isinstance(snap, MarketSnapshot)
    assert set(snap.prices.keys()) == {"yes", "no"}


def test_as_snapshot_venue_kind_is_model():
    """Snapshot venue kind is VenueKind.MODEL."""
    market = _binary_market()
    result = PollingModelProvider(
        polls=[_poll(market_key=market.key, prob=0.60)],
        _reference_date=TODAY,
    ).estimate(market)
    assert result is not None
    snap = result.as_snapshot()
    assert snap.venue.kind == VenueKind.MODEL


def test_as_snapshot_decimal_odds_consistent_with_probs():
    """Snapshot decimal_odds == 1 / prob for each selection (round-trip)."""
    market = _binary_market()
    result = PollingModelProvider(
        polls=[_poll(market_key=market.key, prob=0.60)],
        _reference_date=TODAY,
    ).estimate(market)
    assert result is not None
    snap = result.as_snapshot()
    for sel in market.selections:
        odds = snap.prices[sel.key].decimal_odds
        prob_from_odds = 1.0 / odds
        assert abs(prob_from_odds - result.prob(sel.key)) < 1e-6


def test_as_snapshot_is_not_closing():
    """Model snapshot is never marked as closing."""
    market = _binary_market()
    result = PollingModelProvider(
        polls=[_poll(market_key=market.key)],
        _reference_date=TODAY,
    ).estimate(market)
    assert result is not None
    assert result.as_snapshot().is_closing is False


def test_as_snapshot_decimal_odds_gt_1():
    """All decimal_odds in snapshot are strictly > 1.0 (Price invariant)."""
    market = _binary_market()
    # Use a near-certain poll (0.99) to stress-test the clamping
    result = PollingModelProvider(
        polls=[_poll(market_key=market.key, prob=0.99)],
        _reference_date=TODAY,
    ).estimate(market)
    assert result is not None
    snap = result.as_snapshot()
    for price in snap.prices.values():
        assert price.decimal_odds > 1.0


# --------------------------------------------------------------------------- #
# ModelEstimate.prob()
# --------------------------------------------------------------------------- #
def test_estimate_prob_key_error_on_missing_key():
    """ModelEstimate.prob() raises KeyError for unknown selection key."""
    market = _binary_market()
    result = PollingModelProvider(
        polls=[_poll(market_key=market.key)],
        _reference_date=TODAY,
    ).estimate(market)
    assert result is not None
    try:
        result.prob("draw")
        assert False, "expected KeyError"
    except KeyError:
        pass


# --------------------------------------------------------------------------- #
# Multi-selection market (3 candidates)
# --------------------------------------------------------------------------- #
def test_estimate_three_way_market_probs_sum_to_one():
    """Three-candidate market: probs from polls sum to 1.0 after normalization."""
    market = Market(
        key="KXELECT-PRIMARY-3WAY",
        sport="KXELECTIONS",
        event_key="PRIMARY-3WAY",
        type=MarketType.BINARY,
        selections=(
            Selection("a", "Candidate A"),
            Selection("b", "Candidate B"),
            Selection("c", "Candidate C"),
        ),
    )
    polls = [
        Poll(
            market_key=market.key, selection_key="a", prob=0.45, n=1000,
            field_end=TODAY, pollster="P",
            methodology=Methodology.LIVE_PHONE, population=Population.LIKELY_VOTERS,
        ),
        Poll(
            market_key=market.key, selection_key="b", prob=0.35, n=1000,
            field_end=TODAY, pollster="P",
            methodology=Methodology.LIVE_PHONE, population=Population.LIKELY_VOTERS,
        ),
        # "c" has no polls → imputed as residual
    ]
    result = PollingModelProvider(polls=polls, _reference_date=TODAY).estimate(market)
    assert result is not None
    total = sum(result.selection_probs.values())
    assert abs(total - 1.0) < 1e-9
    # "c" should receive the residual: 1 - (0.45 + 0.35) = 0.20 → normalized
    assert result.prob("c") > 0.0


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:
                print(f"  FAIL  {name}: {exc}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                failed += 1
    print(f"\n{'All tests passed.' if failed == 0 else f'{failed} test(s) failed.'}")
    sys.exit(failed)
