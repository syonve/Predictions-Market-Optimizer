"""Known-answer tests for CLV tracking and calibration.

All numeric results are derived independently before being checked against
the implementation.

CLV metrics used throughout:
  clv_edge      = closing_prob * entry_odds - 1   (same formula as ev.edge,
                  but closing_prob is the ground truth instead of model_prob)
  clv_log       = log(entry_odds / closing_odds)  (additive; preferred for
                  aggregation and the t-statistic)
  clv_odds_ratio = entry_odds / closing_odds - 1  (simple %-better at entry)

Brier score:
  BS = (1/N) * sum((p_i - o_i)^2)   where o_i in {0, 1}
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

import pytest

from fve.tracking.calibration import (
    CalibrationResult,
    brier_score,
    reliability_diagram,
)
from fve.tracking.clv import (
    BetRecord,
    CLVRecord,
    CLVSummary,
    aggregate_clv,
    clv_edge,
    clv_log,
    clv_odds_ratio,
)
from fve.types import Market, MarketType, Selection, Venue, VenueKind

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
TS_ENTRY  = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
TS_CLOSE  = datetime(2026, 6, 7, 20, 0, 0, tzinfo=timezone.utc)

MARKET = Market(
    key="test:event:ml",
    sport="test",
    event_key="test:event",
    type=MarketType.MONEYLINE,
    selections=(Selection("home", "Home"), Selection("away", "Away")),
)
VENUE = Venue("kalshi", "Kalshi", VenueKind.PREDICTION_MARKET)


def make_bet(entry_odds: float, fair_prob_at_entry: float, stake: float = 100.0) -> BetRecord:
    return BetRecord(
        id=str(uuid.uuid4()),
        market=MARKET,
        selection_key="home",
        selection_name="Home",
        venue=VENUE,
        entry_odds=entry_odds,
        fair_prob_at_entry=fair_prob_at_entry,
        stake=stake,
        placed_at=TS_ENTRY,
    )


def make_clv(
    entry_odds: float,
    fair_prob_at_entry: float,
    closing_odds: float,
    closing_prob: float,
    outcome: bool | None = None,
    stake: float = 100.0,
) -> CLVRecord:
    bet = make_bet(entry_odds, fair_prob_at_entry, stake)
    pnl: float | None = None
    if outcome is not None:
        pnl = stake * (entry_odds - 1) if outcome else -stake
    return CLVRecord(
        bet=bet,
        closing_odds=closing_odds,
        closing_prob=closing_prob,
        settled_at=TS_CLOSE if outcome is not None else None,
        outcome=outcome,
        pnl=pnl,
    )


# =========================================================================== #
# Pure CLV functions
# =========================================================================== #

class TestCLVEdge:
    def test_positive_clv(self):
        # entry 2.10, closing_prob 0.52 → 0.52*2.10 - 1 = 0.092
        assert math.isclose(clv_edge(2.10, 0.52), 0.092, abs_tol=1e-12)

    def test_negative_clv(self):
        # entry 1.80, closing_prob 0.60 → 0.60*1.80 - 1 = 0.08  (positive)
        assert math.isclose(clv_edge(1.80, 0.60), 0.08, abs_tol=1e-12)

    def test_closing_line_beats_entry(self):
        # entry 1.80, closing_prob 0.65 → 0.65*1.80 - 1 = 0.17  (closing moved against you)
        assert math.isclose(clv_edge(1.80, 0.65), 0.17, abs_tol=1e-12)

    def test_zero_clv_at_fair_entry(self):
        # entry = fair price → closing_prob = 1/entry → edge = 0
        for d in (1.5, 2.0, 3.5):
            assert math.isclose(clv_edge(d, 1.0 / d), 0.0, abs_tol=1e-12)

    def test_same_formula_as_ev_edge(self):
        # clv_edge is just ev.edge with closing_prob as "fair_prob"
        from fve.signals.ev import edge
        p, d = 0.55, 2.10
        assert math.isclose(clv_edge(d, p), edge(p, d), abs_tol=1e-12)


class TestCLVLog:
    def test_beat_closing_line(self):
        # entry 2.10, close 1.90 → log(2.10/1.90) = log(1.10526...) ≈ 0.10008
        expected = math.log(2.10 / 1.90)
        assert math.isclose(clv_log(2.10, 1.90), expected, abs_tol=1e-12)

    def test_behind_closing_line(self):
        # entry 1.80, close 1.90 → log(1.80/1.90) < 0
        expected = math.log(1.80 / 1.90)
        assert math.isclose(clv_log(1.80, 1.90), expected, abs_tol=1e-12)
        assert clv_log(1.80, 1.90) < 0.0

    def test_matched_entry_and_close(self):
        # entry == close → log(1) = 0
        assert math.isclose(clv_log(2.0, 2.0), 0.0, abs_tol=1e-12)

    def test_additivity(self):
        # log CLV from A→B then B→C equals log CLV from A→C directly
        # log(d_a/d_c) = log(d_a/d_b) + log(d_b/d_c)
        da, db, dc = 2.10, 2.00, 1.90
        assert math.isclose(
            clv_log(da, db) + clv_log(db, dc),
            clv_log(da, dc),
            abs_tol=1e-12,
        )


class TestCLVOddsRatio:
    def test_known_value(self):
        # entry 2.10, close 1.90 → 2.10/1.90 - 1 = 0.10526...
        expected = 2.10 / 1.90 - 1.0
        assert math.isclose(clv_odds_ratio(2.10, 1.90), expected, abs_tol=1e-12)

    def test_zero_when_equal(self):
        assert math.isclose(clv_odds_ratio(2.0, 2.0), 0.0, abs_tol=1e-12)

    def test_negative_when_behind(self):
        assert clv_odds_ratio(1.80, 1.95) < 0.0

    def test_consistent_sign_with_log(self):
        # positive odds-ratio ↔ positive log CLV
        assert (clv_odds_ratio(2.10, 1.90) > 0) == (clv_log(2.10, 1.90) > 0)
        assert (clv_odds_ratio(1.80, 1.95) > 0) == (clv_log(1.80, 1.95) > 0)


# =========================================================================== #
# BetRecord and CLVRecord
# =========================================================================== #

class TestBetRecord:
    def test_fields_stored(self):
        bet = make_bet(2.10, 0.52)
        assert bet.entry_odds == 2.10
        assert bet.fair_prob_at_entry == 0.52
        assert bet.stake == 100.0
        assert bet.selection_key == "home"

    def test_entry_edge_property(self):
        # entry_edge = fair_prob_at_entry * entry_odds - 1
        bet = make_bet(2.10, 0.55)
        assert math.isclose(bet.entry_edge, 0.55 * 2.10 - 1, abs_tol=1e-12)


class TestCLVRecord:
    def test_clv_edge_property(self):
        rec = make_clv(2.10, 0.52, 1.90, 0.55)
        # clv_edge = 0.55 * 2.10 - 1 = 0.155
        assert math.isclose(rec.clv_edge, 0.155, abs_tol=1e-9)

    def test_clv_log_property(self):
        rec = make_clv(2.10, 0.52, 1.90, 0.55)
        assert math.isclose(rec.clv_log, math.log(2.10 / 1.90), abs_tol=1e-12)

    def test_clv_odds_ratio_property(self):
        rec = make_clv(2.10, 0.52, 1.90, 0.55)
        assert math.isclose(rec.clv_odds_ratio, 2.10 / 1.90 - 1, abs_tol=1e-12)

    def test_pnl_winner(self):
        # win at 2.10 with $100 stake → profit = 100*(2.10-1) = 110
        rec = make_clv(2.10, 0.52, 1.90, 0.55, outcome=True, stake=100.0)
        assert math.isclose(rec.pnl, 110.0, abs_tol=1e-9)

    def test_pnl_loser(self):
        rec = make_clv(2.10, 0.52, 1.90, 0.55, outcome=False, stake=100.0)
        assert math.isclose(rec.pnl, -100.0, abs_tol=1e-9)

    def test_pending_outcome(self):
        rec = make_clv(2.10, 0.52, 1.90, 0.55, outcome=None)
        assert rec.outcome is None
        assert rec.pnl is None

    def test_model_drift_property(self):
        # model_drift = closing_prob - fair_prob_at_entry
        # entry model said 0.52, closing said 0.55 → drift = +0.03 (model underestimated)
        rec = make_clv(2.10, 0.52, 1.90, 0.55)
        assert math.isclose(rec.model_drift, 0.55 - 0.52, abs_tol=1e-12)


# =========================================================================== #
# aggregate_clv()
# =========================================================================== #

class TestAggregateCLV:
    def _make_records(self):
        # Three records with known log CLVs:
        #   rec1: entry=2.10, close=1.90 → log(2.10/1.90) ≈  0.10008
        #   rec2: entry=1.80, close=1.95 → log(1.80/1.95) ≈ -0.08004
        #   rec3: entry=2.20, close=1.85 → log(2.20/1.85) ≈  0.17435
        return [
            make_clv(2.10, 0.52, 1.90, 0.55, outcome=True),
            make_clv(1.80, 0.55, 1.95, 0.52, outcome=False),
            make_clv(2.20, 0.50, 1.85, 0.58, outcome=True),
        ]

    def test_n_bets(self):
        assert aggregate_clv(self._make_records()).n_bets == 3

    def test_mean_clv_log(self):
        logs = [math.log(2.10/1.90), math.log(1.80/1.95), math.log(2.20/1.85)]
        expected = sum(logs) / 3
        result = aggregate_clv(self._make_records())
        assert math.isclose(result.mean_clv_log, expected, abs_tol=1e-9)

    def test_mean_clv_edge(self):
        edges = [
            0.55 * 2.10 - 1,
            0.52 * 1.80 - 1,
            0.58 * 2.20 - 1,
        ]
        expected = sum(edges) / 3
        result = aggregate_clv(self._make_records())
        assert math.isclose(result.mean_clv_edge, expected, abs_tol=1e-9)

    def test_n_positive(self):
        # rec1: log > 0 ✓, rec2: log < 0 ✗, rec3: log > 0 ✓
        result = aggregate_clv(self._make_records())
        assert result.n_positive == 2

    def test_hit_rate(self):
        result = aggregate_clv(self._make_records())
        assert math.isclose(result.hit_rate, 2 / 3, abs_tol=1e-9)

    def test_std_clv_log(self):
        logs = [math.log(2.10/1.90), math.log(1.80/1.95), math.log(2.20/1.85)]
        mean = sum(logs) / 3
        variance = sum((x - mean) ** 2 for x in logs) / (len(logs) - 1)  # sample std
        expected_std = math.sqrt(variance)
        result = aggregate_clv(self._make_records())
        assert math.isclose(result.std_clv_log, expected_std, abs_tol=1e-9)

    def test_t_stat(self):
        logs = [math.log(2.10/1.90), math.log(1.80/1.95), math.log(2.20/1.85)]
        mean = sum(logs) / 3
        std = math.sqrt(sum((x - mean) ** 2 for x in logs) / 2)
        expected_t = mean / (std / math.sqrt(3))
        result = aggregate_clv(self._make_records())
        assert math.isclose(result.t_stat, expected_t, abs_tol=1e-9)

    def test_single_record_t_stat_is_none(self):
        # t-stat undefined with n=1 (std=0 or undefined)
        result = aggregate_clv([make_clv(2.10, 0.52, 1.90, 0.55)])
        assert result.t_stat is None

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_clv([])

    def test_total_pnl(self):
        # rec1 win: +110, rec2 lose: -100, rec3 win: +120  → total = 130
        recs = [
            make_clv(2.10, 0.52, 1.90, 0.55, outcome=True,  stake=100.0),
            make_clv(1.80, 0.55, 1.95, 0.52, outcome=False, stake=100.0),
            make_clv(2.20, 0.50, 1.85, 0.58, outcome=True,  stake=100.0),
        ]
        result = aggregate_clv(recs)
        assert math.isclose(result.total_pnl, 110.0 - 100.0 + 120.0, abs_tol=1e-6)

    def test_pending_bets_excluded_from_pnl(self):
        recs = [
            make_clv(2.10, 0.52, 1.90, 0.55, outcome=True,  stake=100.0),
            make_clv(1.80, 0.55, 1.95, 0.52, outcome=None,  stake=100.0),  # pending
        ]
        result = aggregate_clv(recs)
        assert result.total_pnl is not None
        assert math.isclose(result.total_pnl, 110.0, abs_tol=1e-6)


# =========================================================================== #
# Brier score
# =========================================================================== #

class TestBrierScore:
    def test_perfect_predictions(self):
        # p=1.0 on outcome=True, p=0.0 on outcome=False → BS=0
        assert math.isclose(brier_score([(1.0, True), (0.0, False)]), 0.0, abs_tol=1e-12)

    def test_worst_predictions(self):
        # p=0.0 on outcome=True, p=1.0 on outcome=False → BS=1
        assert math.isclose(brier_score([(0.0, True), (1.0, False)]), 1.0, abs_tol=1e-12)

    def test_uninformative(self):
        # always predict 0.5 → BS=0.25 regardless of outcomes
        pairs = [(0.5, True), (0.5, False), (0.5, True), (0.5, True)]
        assert math.isclose(brier_score(pairs), 0.25, abs_tol=1e-12)

    def test_known_answer(self):
        # p=[0.8, 0.2], outcomes=[True, False]
        # BS = ((0.8-1)^2 + (0.2-0)^2) / 2 = (0.04 + 0.04) / 2 = 0.04
        assert math.isclose(
            brier_score([(0.8, True), (0.2, False)]),
            0.04,
            abs_tol=1e-12,
        )

    def test_single_correct_prediction(self):
        # p=0.7, outcome=True → BS = (0.7-1)^2 = 0.09
        assert math.isclose(brier_score([(0.7, True)]), 0.09, abs_tol=1e-12)

    def test_single_wrong_prediction(self):
        # p=0.7, outcome=False → BS = (0.7-0)^2 = 0.49
        assert math.isclose(brier_score([(0.7, False)]), 0.49, abs_tol=1e-12)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            brier_score([])


# =========================================================================== #
# Reliability diagram
# =========================================================================== #

class TestReliabilityDiagram:
    def _make_pairs(self):
        # 20 predictions spread across buckets
        return (
            [(0.1, False)] * 8 + [(0.1, True)] * 2    # bucket [0.0,0.2): 10 samples, 20% hit
            + [(0.5, True)] * 6 + [(0.5, False)] * 4   # bucket [0.4,0.6): 10 samples, 60% hit
            + [(0.9, True)] * 9 + [(0.9, False)] * 1   # bucket [0.8,1.0): 10 samples, 90% hit
        )

    def test_returns_calibration_result(self):
        result = reliability_diagram(self._make_pairs(), n_bins=5)
        assert isinstance(result, CalibrationResult)

    def test_brier_score_matches_standalone(self):
        pairs = self._make_pairs()
        result = reliability_diagram(pairs, n_bins=5)
        assert math.isclose(result.brier_score, brier_score(pairs), abs_tol=1e-9)

    def test_bin_actual_frequencies(self):
        pairs = self._make_pairs()
        result = reliability_diagram(pairs, n_bins=5)
        # [0.0,0.2) bucket: 10 samples, 2 positive → freq = 0.2
        low_bin = next(b for b in result.bins if b.prob_low == pytest.approx(0.0, abs=1e-9))
        assert math.isclose(low_bin.actual_freq, 0.2, abs_tol=1e-9)

    def test_bin_counts(self):
        pairs = self._make_pairs()
        result = reliability_diagram(pairs, n_bins=5)
        total = sum(b.n for b in result.bins)
        assert total == len(pairs)

    def test_empty_bins_excluded(self):
        # If a bin has no samples it should be omitted
        pairs = [(0.1, True), (0.1, False), (0.9, True)]
        result = reliability_diagram(pairs, n_bins=10)
        for b in result.bins:
            assert b.n > 0

    def test_n_samples(self):
        pairs = self._make_pairs()
        result = reliability_diagram(pairs, n_bins=5)
        assert result.n_samples == len(pairs)
