"""Known-answer tests for the election-fundamentals model.

Every numeric expectation is derived by hand before being checked against the
implementation. The model works in log-odds space:

    logit(p_out) = logit(prior) + incumbency_bump + fundraising_bump + primary_bump

Hand reference values (sigmoid(x) = 1 / (1 + e^-x)):

    sigmoid(0.40) = 0.598688     (incumbent only, default coef)
    sigmoid(-0.40) = 0.401312    (challenger only)
    sigmoid(0.48) = 0.617747     (LOCAL fundraising 80/20: 0.80 * 0.6)
    sigmoid(0.30) = 0.574443     (STATE fundraising 80/20: 0.50 * 0.6)
    sigmoid(0.15) = 0.537430     (FEDERAL fundraising 80/20: 0.25 * 0.6)
    sigmoid(0.09) = 0.522485     (primary margin +30: 0.30 * 0.30)
    sigmoid(0.97) = 0.725120     (all three combined: 0.40 + 0.48 + 0.09)
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from fve.models.fundamentals import (
    FundamentalsConfig,
    FundamentalsModelProvider,
    Incumbency,
    RaceFundamentals,
    RaceLevel,
    fundamentals_probability,
    fundraising_bump,
    incumbency_bump,
    logit,
    primary_bump,
    sigmoid,
)
from fve.types import Market, MarketType, Selection

MARKET = Market(
    key="KXSEN-AZ-DEM",
    sport="elections",
    event_key="KXSEN-AZ",
    type=MarketType.BINARY,
    selections=(Selection("yes", "Yes"), Selection("no", "No")),
)


def make_race(**kw) -> RaceFundamentals:
    defaults = dict(
        market_key=MARKET.key,
        selection_key="yes",
        race_level=RaceLevel.STATE,
    )
    defaults.update(kw)
    return RaceFundamentals(**defaults)


# --------------------------------------------------------------------------- #
# logit / sigmoid primitives
# --------------------------------------------------------------------------- #
class TestLogitSigmoid:
    def test_sigmoid_inverts_logit(self):
        for p in (0.01, 0.25, 0.5, 0.75, 0.99):
            assert math.isclose(sigmoid(logit(p)), p, abs_tol=1e-12)

    def test_logit_half_is_zero(self):
        assert math.isclose(logit(0.5), 0.0, abs_tol=1e-12)

    def test_sigmoid_known_value(self):
        # sigmoid(0.4) = 1 / (1 + e^-0.4) = 0.598688
        assert math.isclose(sigmoid(0.4), 0.598688, abs_tol=1e-6)

    def test_logit_rejects_boundaries(self):
        with pytest.raises(ValueError):
            logit(0.0)
        with pytest.raises(ValueError):
            logit(1.0)


# --------------------------------------------------------------------------- #
# Individual bumps
# --------------------------------------------------------------------------- #
class TestIncumbencyBump:
    def test_incumbent_positive(self):
        assert math.isclose(incumbency_bump(Incumbency.INCUMBENT, 0.40), 0.40)

    def test_challenger_negative(self):
        assert math.isclose(incumbency_bump(Incumbency.CHALLENGER, 0.40), -0.40)

    def test_open_seat_zero(self):
        assert incumbency_bump(Incumbency.OPEN_SEAT, 0.40) == 0.0


class TestFundraisingBump:
    def test_even_money_no_bump(self):
        assert fundraising_bump(500_000, 500_000, 0.80) == 0.0

    def test_eighty_twenty_local(self):
        # share = 0.8, centered = 2*0.8 - 1 = 0.6, bump = 0.80 * 0.6 = 0.48
        assert math.isclose(fundraising_bump(800_000, 200_000, 0.80), 0.48, abs_tol=1e-12)

    def test_money_disadvantage_is_negative(self):
        assert fundraising_bump(200_000, 800_000, 0.80) < 0.0

    def test_no_data_returns_zero(self):
        assert fundraising_bump(None, None, 0.80) == 0.0
        assert fundraising_bump(0.0, 0.0, 0.80) == 0.0

    def test_one_sided_money_is_full_coef(self):
        # Opponent raised nothing: share = 1.0, centered = 1.0 → full coef
        assert math.isclose(fundraising_bump(100_000, 0.0, 0.80), 0.80, abs_tol=1e-12)

    def test_negative_totals_rejected(self):
        with pytest.raises(ValueError):
            fundraising_bump(-1.0, 100.0, 0.80)


class TestPrimaryBump:
    def test_known_value(self):
        # margin +0.30 (won primary by 30 points), coef 0.30 → bump 0.09
        assert math.isclose(primary_bump(0.30, 0.30), 0.09, abs_tol=1e-12)

    def test_none_margin_is_zero(self):
        assert primary_bump(None, 0.30) == 0.0

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            primary_bump(1.5, 0.30)


# --------------------------------------------------------------------------- #
# Combined probability — hand-derived known answers
# --------------------------------------------------------------------------- #
class TestFundamentalsProbability:
    def test_no_signals_returns_prior(self):
        race = make_race()
        prob, n_signals = fundamentals_probability(race, FundamentalsConfig())
        assert math.isclose(prob, 0.5, abs_tol=1e-12)
        assert n_signals == 0

    def test_incumbent_only(self):
        race = make_race(incumbency=Incumbency.INCUMBENT)
        prob, n_signals = fundamentals_probability(race, FundamentalsConfig())
        assert math.isclose(prob, 0.598688, abs_tol=1e-6)
        assert n_signals == 1

    def test_challenger_only(self):
        race = make_race(incumbency=Incumbency.CHALLENGER)
        prob, _ = fundamentals_probability(race, FundamentalsConfig())
        assert math.isclose(prob, 0.401312, abs_tol=1e-6)

    def test_fundraising_levels_ordered_local_gt_state_gt_federal(self):
        # Same 80/20 money split must matter most for LOCAL, least for FEDERAL.
        probs = {}
        for level in (RaceLevel.LOCAL, RaceLevel.STATE, RaceLevel.FEDERAL):
            race = make_race(
                race_level=level,
                fundraising_own=800_000,
                fundraising_opponent=200_000,
            )
            probs[level], _ = fundamentals_probability(race, FundamentalsConfig())
        assert probs[RaceLevel.LOCAL] > probs[RaceLevel.STATE] > probs[RaceLevel.FEDERAL]
        # Known answers from default coefs (0.80 / 0.50 / 0.25 at centered 0.6)
        assert math.isclose(probs[RaceLevel.LOCAL], 0.617747, abs_tol=1e-6)
        assert math.isclose(probs[RaceLevel.STATE], 0.574443, abs_tol=1e-6)
        assert math.isclose(probs[RaceLevel.FEDERAL], 0.537430, abs_tol=1e-6)

    def test_primary_only(self):
        race = make_race(primary_margin=0.30)
        prob, _ = fundamentals_probability(race, FundamentalsConfig())
        assert math.isclose(prob, 0.522485, abs_tol=1e-6)

    def test_all_three_signals_combined(self):
        # bumps: incumbency 0.40 + local fundraising 0.48 + primary 0.09 = 0.97
        race = make_race(
            race_level=RaceLevel.LOCAL,
            incumbency=Incumbency.INCUMBENT,
            fundraising_own=800_000,
            fundraising_opponent=200_000,
            primary_margin=0.30,
        )
        prob, n_signals = fundamentals_probability(race, FundamentalsConfig())
        assert math.isclose(prob, 0.725120, abs_tol=1e-6)
        assert n_signals == 3

    def test_non_even_prior(self):
        # logit(0.4) = ln(2/3) = -0.405465; +0.40 incumbent → -0.005465
        # sigmoid(-0.005465) = 0.498634
        race = make_race(incumbency=Incumbency.INCUMBENT, prior_prob=0.4)
        prob, _ = fundamentals_probability(race, FundamentalsConfig())
        assert math.isclose(prob, 0.498634, abs_tol=1e-6)

    def test_signals_are_symmetric(self):
        # Swapping the candidate's situation for its mirror image must give 1-p.
        race_strong = make_race(
            race_level=RaceLevel.STATE,
            incumbency=Incumbency.INCUMBENT,
            fundraising_own=800_000,
            fundraising_opponent=200_000,
            primary_margin=0.30,
        )
        race_weak = make_race(
            race_level=RaceLevel.STATE,
            incumbency=Incumbency.CHALLENGER,
            fundraising_own=200_000,
            fundraising_opponent=800_000,
            primary_margin=-0.30,
        )
        p_strong, _ = fundamentals_probability(race_strong, FundamentalsConfig())
        p_weak, _ = fundamentals_probability(race_weak, FundamentalsConfig())
        assert math.isclose(p_strong + p_weak, 1.0, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_prior_must_be_open_interval(self):
        with pytest.raises(ValueError):
            make_race(prior_prob=0.0)
        with pytest.raises(ValueError):
            make_race(prior_prob=1.0)

    def test_primary_margin_range(self):
        with pytest.raises(ValueError):
            make_race(primary_margin=1.01)

    def test_negative_fundraising_rejected(self):
        with pytest.raises(ValueError):
            make_race(fundraising_own=-5.0)

    def test_config_rejects_negative_coefs(self):
        with pytest.raises(ValueError):
            FundamentalsConfig(incumbency_coef=-0.1)
        with pytest.raises(ValueError):
            FundamentalsConfig(max_confidence=1.5)


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #
class TestFundamentalsModelProvider:
    def test_unknown_market_returns_none(self):
        provider = FundamentalsModelProvider(races=[])
        assert provider.estimate(MARKET) is None

    def test_zero_signal_race_returns_none(self):
        # A race with no incumbency, money, or primary data carries no
        # information — the provider must stay silent, not inject 50/50.
        provider = FundamentalsModelProvider(races=[make_race()])
        assert provider.estimate(MARKET) is None

    def test_estimate_probs_sum_to_one(self):
        race = make_race(incumbency=Incumbency.INCUMBENT)
        provider = FundamentalsModelProvider(races=[race])
        est = provider.estimate(MARKET)
        assert est is not None
        assert math.isclose(sum(est.selection_probs.values()), 1.0, abs_tol=1e-12)
        assert math.isclose(est.selection_probs["yes"], 0.598688, abs_tol=1e-6)
        assert math.isclose(est.selection_probs["no"], 1 - 0.598688, abs_tol=1e-6)

    def test_estimate_for_no_side_race(self):
        # Fundamentals can describe the NO selection; YES gets the complement.
        race = make_race(selection_key="no", incumbency=Incumbency.INCUMBENT)
        provider = FundamentalsModelProvider(races=[race])
        est = provider.estimate(MARKET)
        assert est is not None
        assert math.isclose(est.selection_probs["no"], 0.598688, abs_tol=1e-6)

    def test_non_binary_market_returns_none(self):
        multi = Market(
            key="KXSEN-AZ-DEM",
            sport="elections",
            event_key="KXSEN-AZ",
            type=MarketType.OUTRIGHT,
            selections=(
                Selection("a", "A"), Selection("b", "B"), Selection("c", "C"),
            ),
        )
        race = make_race(incumbency=Incumbency.INCUMBENT)
        provider = FundamentalsModelProvider(races=[race])
        assert provider.estimate(multi) is None

    def test_confidence_scales_with_signal_count(self):
        # Default max_confidence = 0.5: 1 signal → 1/6, 3 signals → 0.5
        one = make_race(incumbency=Incumbency.INCUMBENT)
        three = make_race(
            incumbency=Incumbency.INCUMBENT,
            fundraising_own=800_000,
            fundraising_opponent=200_000,
            primary_margin=0.30,
        )
        est_one = FundamentalsModelProvider(races=[one]).estimate(MARKET)
        est_three = FundamentalsModelProvider(races=[three]).estimate(MARKET)
        assert est_one is not None and est_three is not None
        assert math.isclose(est_one.confidence, 0.5 / 3, abs_tol=1e-12)
        assert math.isclose(est_three.confidence, 0.5, abs_tol=1e-12)
        assert est_one.n_samples == 1
        assert est_three.n_samples == 3

    def test_snapshot_venue_is_distinct_from_polling(self):
        race = make_race(incumbency=Incumbency.INCUMBENT)
        est = FundamentalsModelProvider(races=[race]).estimate(MARKET)
        assert est is not None
        snap = est.as_snapshot()
        assert snap.venue.key == "fundamentals_model"
        assert snap.venue.key != "polling_model"

    def test_snapshot_prices_carry_confidence_as_size(self):
        # Consensus scales MODEL weight by Price.size = confidence.
        race = make_race(incumbency=Incumbency.INCUMBENT)  # 1 signal → 0.5/3
        est = FundamentalsModelProvider(races=[race]).estimate(MARKET)
        assert est is not None
        snap = est.as_snapshot()
        for price in snap.prices.values():
            assert price.size == pytest.approx(0.5 / 3, abs=1e-12)
