"""Known-answer tests for the EV and arb signal modules.

EV tests verify the core edge formula and the scan that maps it over
snapshots. Arb tests verify the ROI formula, stake fractions, and the
scan that finds the best available price per outcome across venues.

References for hand-computed values are included inline.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from fve.pricing.consensus import ConsensusResult
from fve.signals.arb import ArbSignal, arb_roi, scan_arb, stake_fractions
from fve.signals.ev import EVSignal, edge, scan_ev
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
# Shared fixtures
# --------------------------------------------------------------------------- #
TS = datetime(2026, 6, 7, 12, 0, 0, tzinfo=timezone.utc)

SEL_HOME = Selection("home", "Home")
SEL_AWAY = Selection("away", "Away")
MARKET = Market(
    key="test:event:ml",
    sport="test",
    event_key="test:event",
    type=MarketType.MONEYLINE,
    selections=(SEL_HOME, SEL_AWAY),
)

SHARP = Venue("pinnacle", "Pinnacle", VenueKind.SHARP)
SOFT_A = Venue("bookA", "BookA", VenueKind.SOFT)
SOFT_B = Venue("bookB", "BookB", VenueKind.SOFT)


def make_snap(venue: Venue, home_odds: float, away_odds: float) -> MarketSnapshot:
    return MarketSnapshot(
        market=MARKET,
        venue=venue,
        timestamp=TS,
        prices={
            "home": Price(decimal_odds=home_odds),
            "away": Price(decimal_odds=away_odds),
        },
    )


def make_consensus(home_prob: float) -> ConsensusResult:
    """Build a ConsensusResult with the given home probability directly."""
    away_prob = 1.0 - home_prob
    return ConsensusResult(
        selection_keys=("home", "away"),
        fair_probs=(home_prob, away_prob),
        n_sharp=1,
        n_soft=0,
        n_pm=0,
        venue_probs={"pinnacle": (home_prob, away_prob)},
    )


# =========================================================================== #
# EV SIGNAL TESTS
# =========================================================================== #

class TestEdge:
    def test_positive_edge(self):
        # fair_prob=0.55, odds=2.0 → edge = 0.55*2.0 - 1 = 0.10
        assert math.isclose(edge(0.55, 2.0), 0.10, abs_tol=1e-12)

    def test_negative_edge(self):
        # fair_prob=0.50, odds=1.90 → edge = 0.50*1.90 - 1 = -0.05
        assert math.isclose(edge(0.50, 1.90), -0.05, abs_tol=1e-12)

    def test_zero_edge_at_fair_price(self):
        # quoted at fair price → edge = 0
        for p in (0.3, 0.5, 0.7):
            assert math.isclose(edge(p, 1.0 / p), 0.0, abs_tol=1e-12)

    def test_edge_difference_proportional_to_prob_difference(self):
        # edge(p1, d) - edge(p2, d) = (p1 - p2) * d
        # The -1 constant cancels; only the p*d term differs.
        d = 2.0
        assert math.isclose(edge(0.6, d) - edge(0.3, d), (0.6 - 0.3) * d, abs_tol=1e-12)

    def test_longshot_at_fair_price(self):
        # 10% chance at 10.0 decimal odds → zero edge
        assert math.isclose(edge(0.10, 10.0), 0.0, abs_tol=1e-12)


class TestScanEV:
    def test_returns_positive_ev_signals(self):
        # fair_home=0.60, quoted home at 2.0 → edge = 0.60*2.0-1 = 0.20
        result = make_consensus(0.60)
        snap = make_snap(SHARP, home_odds=2.0, away_odds=1.8)
        signals = scan_ev(result, [snap])
        home_sigs = [s for s in signals if s.selection_key == "home"]
        assert len(home_sigs) == 1
        assert math.isclose(home_sigs[0].edge, 0.20, abs_tol=1e-9)

    def test_negative_ev_excluded_by_default(self):
        # fair_home=0.50, all venues price home at 1.90 → edge = -0.05
        result = make_consensus(0.50)
        snaps = [make_snap(SOFT_A, 1.90, 1.90)]
        signals = scan_ev(result, snaps)
        home_sigs = [s for s in signals if s.selection_key == "home"]
        assert len(home_sigs) == 0

    def test_min_edge_filter(self):
        # edge on home = 0.10; require min_edge=0.15 → excluded
        result = make_consensus(0.55)
        snap = make_snap(SOFT_A, home_odds=2.0, away_odds=1.8)
        assert len(scan_ev(result, [snap], min_edge=0.15)) == 0
        assert len(scan_ev(result, [snap], min_edge=0.05)) > 0

    def test_correct_venue_attribution(self):
        # Two venues; only BookB prices home attractively
        result = make_consensus(0.60)
        snap_a = make_snap(SOFT_A, home_odds=1.60, away_odds=2.30)  # home edge = 0.60*1.60-1 = -0.04
        snap_b = make_snap(SOFT_B, home_odds=2.10, away_odds=1.70)  # home edge = 0.60*2.10-1 = 0.26
        signals = scan_ev(result, [snap_a, snap_b])
        venues = {s.venue.key for s in signals if s.selection_key == "home"}
        assert "bookB" in venues
        assert "bookA" not in venues

    def test_edge_pct_property(self):
        result = make_consensus(0.55)
        snap = make_snap(SHARP, home_odds=2.0, away_odds=1.8)
        sig = next(s for s in scan_ev(result, [snap]) if s.selection_key == "home")
        assert math.isclose(sig.edge_pct, sig.edge * 100.0, abs_tol=1e-12)

    def test_fair_decimal_property(self):
        result = make_consensus(0.60)
        snap = make_snap(SHARP, home_odds=2.0, away_odds=1.8)
        sig = next(s for s in scan_ev(result, [snap]) if s.selection_key == "home")
        assert math.isclose(sig.fair_decimal, 1.0 / 0.60, abs_tol=1e-12)

    def test_all_below_min_edge_returns_empty(self):
        result = make_consensus(0.50)
        snaps = [make_snap(SHARP, 1.85, 1.85)]
        assert scan_ev(result, snaps, min_edge=0.10) == []


# =========================================================================== #
# ARB SIGNAL TESTS
# =========================================================================== #

class TestArbRoi:
    def test_known_arb_roi(self):
        # arb_sum = 1/2.2 + 1/2.1 = 0.93074..., roi = (1-S)/S
        s = 1 / 2.2 + 1 / 2.1
        expected_roi = (1.0 - s) / s
        assert math.isclose(arb_roi(s), expected_roi, abs_tol=1e-12)

    def test_zero_margin_is_zero_roi(self):
        # sum = 1.0 exactly → no profit, roi = 0
        assert math.isclose(arb_roi(1.0), 0.0, abs_tol=1e-12)

    def test_symmetric_arb(self):
        # Both outcomes at 2.10: S = 2/2.10 = 0.9524, roi ≈ 0.05
        s = 2.0 / 2.10
        assert arb_roi(s) > 0.0
        assert math.isclose(arb_roi(s), (1.0 - s) / s, abs_tol=1e-12)


class TestStakeFractions:
    def test_fractions_sum_to_one(self):
        for odds in ([2.2, 2.1], [1.5, 3.0], [2.0, 4.0, 4.0]):
            fracs = stake_fractions(odds)
            assert math.isclose(sum(fracs), 1.0, abs_tol=1e-9), f"odds={odds} sum={sum(fracs)}"

    def test_equal_return_across_outcomes(self):
        # Each leg must pay the same return: frac_i * odds_i == constant
        odds = [2.2, 2.1]
        fracs = stake_fractions(odds)
        returns = [f * d for f, d in zip(fracs, odds)]
        assert math.isclose(returns[0], returns[1], abs_tol=1e-9), returns

    def test_equal_return_three_way(self):
        odds = [2.5, 3.2, 3.4]
        fracs = stake_fractions(odds)
        returns = [f * d for f, d in zip(fracs, odds)]
        assert math.isclose(returns[0], returns[1], abs_tol=1e-9)
        assert math.isclose(returns[1], returns[2], abs_tol=1e-9)

    def test_known_fractions(self):
        # Home 2.2, Away 2.1: S = 1/2.2 + 1/2.1
        # f_home = 1 / (2.2 * S),  f_away = 1 / (2.1 * S)
        odds = [2.2, 2.1]
        s = sum(1.0 / d for d in odds)
        expected = (1.0 / (2.2 * s), 1.0 / (2.1 * s))
        fracs = stake_fractions(odds)
        assert fracs == pytest.approx(expected, abs=1e-12)

    def test_return_equals_one_plus_roi(self):
        # Each leg return = 1 + roi (when total stake = 1)
        odds = [2.2, 2.1]
        s = sum(1.0 / d for d in odds)
        expected_return = 1.0 / s  # = 1 + roi * (1 - arb_margin)
        fracs = stake_fractions(odds)
        for f, d in zip(fracs, odds):
            assert math.isclose(f * d, expected_return, abs_tol=1e-9)


class TestScanArb:
    def test_no_arb_returns_none(self):
        # Pinnacle -110/-110 line: S = 2/1.909 ≈ 1.048 > 1 → no arb
        snap = make_snap(SHARP, 1.909, 1.909)
        assert scan_arb([snap]) is None

    def test_arb_detected_across_two_venues(self):
        # BookA prices home at 2.20, BookB prices away at 2.10
        # S = 1/2.2 + 1/2.1 = 0.9307 < 1 → arb
        snap_a = make_snap(SOFT_A, home_odds=2.20, away_odds=1.70)
        snap_b = make_snap(SOFT_B, home_odds=1.65, away_odds=2.10)
        result = scan_arb([snap_a, snap_b])
        assert result is not None
        s = 1 / 2.2 + 1 / 2.1
        assert math.isclose(result.roi, (1.0 - s) / s, abs_tol=1e-9)

    def test_arb_legs_use_best_odds_per_outcome(self):
        # Home: BookA=2.20 > BookB=1.65  →  leg must use BookA
        # Away: BookB=2.10 > BookA=1.70  →  leg must use BookB
        snap_a = make_snap(SOFT_A, home_odds=2.20, away_odds=1.70)
        snap_b = make_snap(SOFT_B, home_odds=1.65, away_odds=2.10)
        result = scan_arb([snap_a, snap_b])
        assert result is not None
        leg_map = {leg.selection_key: leg for leg in result.legs}
        assert leg_map["home"].venue.key == "bookA"
        assert leg_map["away"].venue.key == "bookB"
        assert math.isclose(leg_map["home"].decimal_odds, 2.20, abs_tol=1e-12)
        assert leg_map["away"].decimal_odds == pytest.approx(2.10, abs=1e-12)

    def test_arb_stake_fractions_sum_to_one(self):
        snap_a = make_snap(SOFT_A, home_odds=2.20, away_odds=1.70)
        snap_b = make_snap(SOFT_B, home_odds=1.65, away_odds=2.10)
        result = scan_arb([snap_a, snap_b])
        assert result is not None
        total_stake = sum(leg.stake_fraction for leg in result.legs)
        assert math.isclose(total_stake, 1.0, abs_tol=1e-9)

    def test_single_venue_no_arb(self):
        # A single book with standard vig is never an arb
        snap = make_snap(SHARP, home_odds=1.95, away_odds=1.95)
        assert scan_arb([snap]) is None

    def test_arb_roi_positive(self):
        snap_a = make_snap(SOFT_A, home_odds=2.20, away_odds=1.70)
        snap_b = make_snap(SOFT_B, home_odds=1.65, away_odds=2.10)
        result = scan_arb([snap_a, snap_b])
        assert result is not None
        assert result.roi > 0.0

    def test_arb_sum_attribute(self):
        # arb_sum should be < 1 when arb detected
        snap_a = make_snap(SOFT_A, home_odds=2.20, away_odds=1.70)
        snap_b = make_snap(SOFT_B, home_odds=1.65, away_odds=2.10)
        result = scan_arb([snap_a, snap_b])
        assert result is not None
        assert result.arb_sum < 1.0
        assert math.isclose(result.arb_sum, 1 / 2.2 + 1 / 2.1, abs_tol=1e-9)
