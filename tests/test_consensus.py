"""Known-answer tests for the consensus engine.

Each test verifies the weighted fair-value calculation against independently
derived references, not against the implementation itself.

Design decisions encoded here:
  - Sharp=1.0, Exchange=1.0, PredictionMarket=0.5, Soft=0.1 (default weights)
  - Liquidity scaling: weight *= sqrt(avg_size) for EXCHANGE/PREDICTION_MARKET
    snapshots when size is available
  - No sharp/exchange anchor → NoSharpAnchorError
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from fve.pricing.consensus import (
    ConsensusResult,
    NoSharpAnchorError,
    _DEFAULT_WEIGHTS,
    _snapshot_weight,
    consensus,
)
from fve.pricing.devig import proportional, implied_probs
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
# Fixtures
# --------------------------------------------------------------------------- #
TS = datetime(2026, 6, 7, 12, 0, 0, tzinfo=timezone.utc)

SEL_HOME = Selection("home", "Home")
SEL_AWAY = Selection("away", "Away")
MARKET = Market(
    key="test:event:moneyline",
    sport="test",
    event_key="test:event",
    type=MarketType.MONEYLINE,
    selections=(SEL_HOME, SEL_AWAY),
)

SHARP_VENUE = Venue("pinnacle", "Pinnacle", VenueKind.SHARP)
EXCHANGE_VENUE = Venue("betfair", "Betfair", VenueKind.EXCHANGE)
PM_VENUE = Venue("kalshi", "Kalshi", VenueKind.PREDICTION_MARKET)
SOFT_VENUE = Venue("draftkings", "DraftKings", VenueKind.SOFT)


def make_snap(
    venue: Venue,
    home_odds: float,
    away_odds: float,
    home_size: float | None = None,
    away_size: float | None = None,
) -> MarketSnapshot:
    return MarketSnapshot(
        market=MARKET,
        venue=venue,
        timestamp=TS,
        prices={
            "home": Price(decimal_odds=home_odds, size=home_size),
            "away": Price(decimal_odds=away_odds, size=away_size),
        },
    )


# --------------------------------------------------------------------------- #
# Single-snapshot: consensus == devigged result directly
# --------------------------------------------------------------------------- #
def test_single_sharp_snapshot_passes_through():
    """With one sharp snapshot, consensus = that snapshot's devigged probs."""
    snap = make_snap(SHARP_VENUE, 1.6, 2.4)
    result = consensus([snap], method="proportional")

    expected = proportional(implied_probs([1.6, 2.4]))
    assert result.fair_probs == pytest.approx(expected, abs=1e-9)
    assert result.n_sharp == 1
    assert result.n_soft == 0


# --------------------------------------------------------------------------- #
# Sharp dominates soft (known-answer computation)
# --------------------------------------------------------------------------- #
def test_sharp_dominates_soft():
    """
    Sharp odds [1.6, 2.4] → proportional fair_p = (0.6, 0.4)
    Soft odds  [1.8, 2.1] → q = [1/1.8, 1/2.1], B = sum
                           → proportional fair_p = (q_h/B, q_a/B)

    With default weights sharp=1.0, soft=0.1, no size:
      w_sharp = 1.0,  w_soft = 0.1
      consensus[i] = (1.0 * p_sharp[i] + 0.1 * p_soft[i]) / 1.1
    """
    snap_sharp = make_snap(SHARP_VENUE, 1.6, 2.4)
    snap_soft = make_snap(SOFT_VENUE, 1.8, 2.1)

    # Independently compute expected
    p_sharp = proportional(implied_probs([1.6, 2.4]))     # (0.6, 0.4)
    p_soft = proportional(implied_probs([1.8, 2.1]))
    w_s, w_f = 1.0, 0.1
    total = w_s + w_f
    expected_home = (w_s * p_sharp[0] + w_f * p_soft[0]) / total
    expected_away = (w_s * p_sharp[1] + w_f * p_soft[1]) / total

    result = consensus([snap_sharp, snap_soft], method="proportional")

    assert result.fair_probs == pytest.approx((expected_home, expected_away), abs=1e-9)
    assert math.isclose(sum(result.fair_probs), 1.0, abs_tol=1e-9)
    # Consensus must be closer to sharp than to soft
    assert abs(result.fair_probs[0] - p_sharp[0]) < abs(result.fair_probs[0] - p_soft[0])


# --------------------------------------------------------------------------- #
# Two equal-weight snapshots → arithmetic mean of their fair probs
# --------------------------------------------------------------------------- #
def test_two_sharp_snapshots_are_averaged():
    """Two SHARP snapshots (same base weight, no size) → simple arithmetic mean."""
    snap1 = make_snap(SHARP_VENUE, 1.6, 2.4)
    snap2 = make_snap(Venue("sharp2", "Sharp2", VenueKind.SHARP), 1.7, 2.2)

    p1 = proportional(implied_probs([1.6, 2.4]))
    p2 = proportional(implied_probs([1.7, 2.2]))
    expected = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    result = consensus([snap1, snap2], method="proportional")
    assert result.fair_probs == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# Symmetry: equal odds → (0.5, 0.5) regardless of how many venues
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["proportional", "shin", "power"])
def test_symmetric_book_gives_even_split(method):
    snaps = [
        make_snap(SHARP_VENUE, 1.91, 1.91),
        make_snap(SOFT_VENUE, 2.0, 2.0),
    ]
    result = consensus(snaps, method=method)
    assert result.fair_probs == pytest.approx((0.5, 0.5), abs=1e-9)
    assert math.isclose(sum(result.fair_probs), 1.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# Liquidity scaling for EXCHANGE / PREDICTION_MARKET
# --------------------------------------------------------------------------- #
def test_liquidity_weights_exchange_snapshots():
    """
    Two EXCHANGE snapshots with different size:
      snap1: size=1  → weight = 1.0 * sqrt(avg(1,1)) = 1.0
      snap2: size=100 → weight = 1.0 * sqrt(avg(100,100)) = 10.0

    Expected consensus[i] = (1.0*p1[i] + 10.0*p2[i]) / 11.0
    """
    snap1 = make_snap(EXCHANGE_VENUE, 1.6, 2.4, home_size=1.0, away_size=1.0)
    snap2 = make_snap(
        Venue("betfair2", "Betfair2", VenueKind.EXCHANGE),
        1.7, 2.2,
        home_size=100.0, away_size=100.0,
    )

    p1 = proportional(implied_probs([1.6, 2.4]))
    p2 = proportional(implied_probs([1.7, 2.2]))
    w1, w2 = 1.0 * math.sqrt(1.0), 1.0 * math.sqrt(100.0)
    total = w1 + w2
    expected = (
        (w1 * p1[0] + w2 * p2[0]) / total,
        (w1 * p1[1] + w2 * p2[1]) / total,
    )

    result = consensus([snap1, snap2], method="proportional")
    assert result.fair_probs == pytest.approx(expected, abs=1e-9)


def test_snapshot_without_size_uses_base_weight():
    """No size data → weight = base weight (sqrt(size) scaling is skipped)."""
    snap_no_size = make_snap(EXCHANGE_VENUE, 1.91, 1.91)
    w = _snapshot_weight(snap_no_size, _DEFAULT_WEIGHTS)
    assert w == pytest.approx(1.0, abs=1e-9)  # Exchange base weight


def test_snapshot_with_size_scales_weight():
    snap = make_snap(PM_VENUE, 1.91, 1.91, home_size=4.0, away_size=4.0)
    w = _snapshot_weight(snap, _DEFAULT_WEIGHTS)
    # PM base=0.5, sqrt(avg(4,4))=2.0 → 0.5 * 2.0 = 1.0
    assert w == pytest.approx(0.5 * math.sqrt(4.0), abs=1e-9)


# --------------------------------------------------------------------------- #
# No sharp anchor → exception
# --------------------------------------------------------------------------- #
def test_no_sharp_raises():
    snaps = [make_snap(SOFT_VENUE, 1.9, 1.9)]
    with pytest.raises(NoSharpAnchorError):
        consensus(snaps)


def test_pm_only_raises():
    snaps = [make_snap(PM_VENUE, 1.9, 1.9, home_size=50.0, away_size=50.0)]
    with pytest.raises(NoSharpAnchorError):
        consensus(snaps)


# --------------------------------------------------------------------------- #
# venue_probs diagnostic: every input venue appears in result
# --------------------------------------------------------------------------- #
def test_venue_probs_populated():
    snap_s = make_snap(SHARP_VENUE, 1.6, 2.4)
    snap_f = make_snap(SOFT_VENUE, 1.8, 2.1)
    result = consensus([snap_s, snap_f], method="proportional")

    assert "pinnacle" in result.venue_probs
    assert "draftkings" in result.venue_probs
    for vp in result.venue_probs.values():
        assert math.isclose(sum(vp), 1.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# ConsensusResult.prob() accessor
# --------------------------------------------------------------------------- #
def test_result_prob_accessor():
    snap = make_snap(SHARP_VENUE, 1.6, 2.4)
    result = consensus([snap], method="proportional")
    assert result.prob("home") == pytest.approx(result.fair_probs[0], abs=1e-12)
    assert result.prob("away") == pytest.approx(result.fair_probs[1], abs=1e-12)


# --------------------------------------------------------------------------- #
# Custom weight overrides
# --------------------------------------------------------------------------- #
def test_custom_weights_override_defaults():
    """Caller can supply custom weights that change the result."""
    snap_sharp = make_snap(SHARP_VENUE, 1.6, 2.4)
    snap_soft = make_snap(SOFT_VENUE, 1.8, 2.1)

    # Give soft equal weight to sharp — result should differ from default
    equal_weights = {
        VenueKind.SHARP: 1.0,
        VenueKind.EXCHANGE: 1.0,
        VenueKind.PREDICTION_MARKET: 1.0,
        VenueKind.SOFT: 1.0,
    }
    result_default = consensus([snap_sharp, snap_soft], method="proportional")
    result_equal = consensus([snap_sharp, snap_soft], method="proportional", weights=equal_weights)

    # With equal weights, soft drags consensus further from sharp
    p_sharp = proportional(implied_probs([1.6, 2.4]))
    assert abs(result_equal.fair_probs[0] - p_sharp[0]) > abs(result_default.fair_probs[0] - p_sharp[0])


# --------------------------------------------------------------------------- #
# Empty input
# --------------------------------------------------------------------------- #
def test_empty_snapshots_raises():
    with pytest.raises(ValueError, match="at least one"):
        consensus([])
