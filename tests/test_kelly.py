"""Known-answer tests for Kelly sizing.

Every numeric result is derived independently before being checked against
the implementation.

Kelly formula used throughout:
    f* = edge / (decimal_odds - 1)
       = (fair_prob * decimal_odds - 1) / (decimal_odds - 1)

With config applied:
    f_applied = min(f* * kelly_fraction * confidence, max_fraction)
    stake      = bankroll * f_applied   (floored at 0 — never bet negative EV)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from fve.sizing.kelly import (
    BetSize,
    KellyConfig,
    kelly_fraction,
    kelly_stake,
    size_ev_signal,
)
from fve.signals.ev import EVSignal
from fve.types import Market, MarketType, Selection, Venue, VenueKind

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
TS = datetime(2026, 6, 7, 12, 0, 0, tzinfo=timezone.utc)
MARKET = Market(
    key="test:event:ml",
    sport="test",
    event_key="test:event",
    type=MarketType.MONEYLINE,
    selections=(Selection("home", "Home"), Selection("away", "Away")),
)
VENUE = Venue("kalshi", "Kalshi", VenueKind.PREDICTION_MARKET)

DEFAULT_CFG = KellyConfig()  # half-Kelly, 5% cap, confidence=1.0


def make_signal(fair_prob: float, decimal_odds: float, confidence: float = 1.0) -> EVSignal:
    from fve.signals.ev import edge as edge_fn
    return EVSignal(
        market=MARKET,
        selection_key="home",
        selection_name="Home",
        venue=VENUE,
        fair_prob=fair_prob,
        decimal_odds=decimal_odds,
        edge=edge_fn(fair_prob, decimal_odds),
    )


# =========================================================================== #
# kelly_fraction() — pure math, no config
# =========================================================================== #

class TestKellyFraction:
    def test_known_answer_even_money(self):
        # p=0.60, d=2.0 → edge=0.20, b=1.0 → f*=0.20
        assert math.isclose(kelly_fraction(0.60, 2.0), 0.20, abs_tol=1e-12)

    def test_known_answer_favorite(self):
        # p=0.70, d=1.50 → edge=0.70*1.5-1=0.05, b=0.50 → f*=0.10
        assert math.isclose(kelly_fraction(0.70, 1.50), 0.10, abs_tol=1e-12)

    def test_known_answer_longshot(self):
        # p=0.20, d=6.0 → edge=0.20*6-1=0.20, b=5.0 → f*=0.04
        assert math.isclose(kelly_fraction(0.20, 6.0), 0.04, abs_tol=1e-12)

    def test_zero_edge_is_zero_fraction(self):
        # quoted at fair price → f*=0
        for p in (0.3, 0.5, 0.7):
            assert math.isclose(kelly_fraction(p, 1.0 / p), 0.0, abs_tol=1e-12)

    def test_negative_edge_returns_zero(self):
        # never bet negative EV — floor at 0
        assert kelly_fraction(0.45, 2.0) == 0.0   # edge = -0.10
        assert kelly_fraction(0.48, 1.90) == 0.0  # edge = -0.088

    def test_prediction_market_form(self):
        # Kalshi YES contract at price c=0.40, fair_prob=0.55
        # d = 1/0.40 = 2.5, edge = 0.55*2.5-1 = 0.375, b = 1.5
        # f* = 0.375/1.5 = 0.25
        # Equivalently: (p - c)/(1 - c) = (0.55-0.40)/(1-0.40) = 0.15/0.60 = 0.25
        c = 0.40
        p = 0.55
        d = 1.0 / c
        assert math.isclose(kelly_fraction(p, d), 0.25, abs_tol=1e-9)

    def test_formula_equivalence(self):
        # f* = edge/(d-1) == (p*d-1)/(d-1) == (p - 1/d) / (1 - 1/d)
        p, d = 0.63, 2.1
        f = kelly_fraction(p, d)
        edge = p * d - 1
        b = d - 1
        assert math.isclose(f, edge / b, abs_tol=1e-12)
        assert math.isclose(f, (p - 1.0/d) / (1.0 - 1.0/d), abs_tol=1e-9)


# =========================================================================== #
# KellyConfig defaults and validation
# =========================================================================== #

class TestKellyConfig:
    def test_defaults(self):
        cfg = KellyConfig()
        assert cfg.kelly_fraction == 0.5
        assert cfg.max_fraction == 0.05
        assert cfg.confidence == 1.0

    def test_custom_values(self):
        cfg = KellyConfig(kelly_fraction=0.25, max_fraction=0.02, confidence=0.8)
        assert cfg.kelly_fraction == 0.25
        assert cfg.max_fraction == 0.02
        assert cfg.confidence == 0.8

    def test_invalid_kelly_fraction_raises(self):
        with pytest.raises(ValueError):
            KellyConfig(kelly_fraction=0.0)
        with pytest.raises(ValueError):
            KellyConfig(kelly_fraction=1.1)

    def test_invalid_max_fraction_raises(self):
        with pytest.raises(ValueError):
            KellyConfig(max_fraction=0.0)
        with pytest.raises(ValueError):
            KellyConfig(max_fraction=1.1)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            KellyConfig(confidence=0.0)
        with pytest.raises(ValueError):
            KellyConfig(confidence=1.1)


# =========================================================================== #
# kelly_stake() — dollar amount with full config applied
# =========================================================================== #

class TestKellyStake:
    def test_half_kelly_no_cap(self):
        # p=0.52, d=2.0 → f*=0.04, half-Kelly=0.02 → cap=5% not binding → stake=200
        cfg = KellyConfig(kelly_fraction=0.5, max_fraction=0.20, confidence=1.0)
        stake = kelly_stake(bankroll=10_000, fair_prob=0.52, decimal_odds=2.0, config=cfg)
        assert math.isclose(stake, 200.0, abs_tol=1e-6)

    def test_half_kelly_with_default_cap(self):
        # p=0.60, d=2.0 → f*=0.20, half-Kelly=0.10 → cap=5% IS binding → stake=500
        stake = kelly_stake(bankroll=10_000, fair_prob=0.60, decimal_odds=2.0, config=DEFAULT_CFG)
        assert math.isclose(stake, 500.0, abs_tol=1e-6)

    def test_cap_binding(self):
        # p=0.80, d=3.0 → f*=(0.80*3-1)/(3-1)=1.4/2=0.70 → half=0.35 → capped at 5%
        stake = kelly_stake(bankroll=10_000, fair_prob=0.80, decimal_odds=3.0, config=DEFAULT_CFG)
        assert math.isclose(stake, 500.0, abs_tol=1e-6)   # 5% of 10k

    def test_negative_edge_is_zero_stake(self):
        stake = kelly_stake(bankroll=10_000, fair_prob=0.45, decimal_odds=2.0, config=DEFAULT_CFG)
        assert stake == 0.0

    def test_zero_edge_is_zero_stake(self):
        stake = kelly_stake(bankroll=10_000, fair_prob=0.50, decimal_odds=2.0, config=DEFAULT_CFG)
        assert stake == 0.0

    def test_confidence_scales_stake(self):
        # confidence=0.5 → stake is exactly half of confidence=1.0 (cap not binding)
        cfg_full = KellyConfig(kelly_fraction=0.5, max_fraction=0.20, confidence=1.0)
        cfg_half = KellyConfig(kelly_fraction=0.5, max_fraction=0.20, confidence=0.5)
        s_full = kelly_stake(10_000, 0.60, 2.0, cfg_full)
        s_half = kelly_stake(10_000, 0.60, 2.0, cfg_half)
        assert math.isclose(s_half, s_full * 0.5, abs_tol=1e-6)

    def test_quarter_kelly_config(self):
        # f*=0.20 with quarter-Kelly → 0.05 → stake=500 on 10k bankroll
        cfg = KellyConfig(kelly_fraction=0.25, max_fraction=0.10, confidence=1.0)
        stake = kelly_stake(10_000, 0.60, 2.0, cfg)
        assert math.isclose(stake, 500.0, abs_tol=1e-6)

    def test_bankroll_scales_proportionally(self):
        s1 = kelly_stake(10_000, 0.60, 2.0, DEFAULT_CFG)
        s2 = kelly_stake(20_000, 0.60, 2.0, DEFAULT_CFG)
        assert math.isclose(s2, 2 * s1, abs_tol=1e-6)


# =========================================================================== #
# size_ev_signal() — full pipeline from EVSignal to BetSize
# =========================================================================== #

class TestSizeEvSignal:
    def test_basic_sizing(self):
        # p=0.60, d=2.0 → f*=0.20, half-Kelly=0.10 → 5% cap binds → stake=500
        sig = make_signal(fair_prob=0.60, decimal_odds=2.0)
        result = size_ev_signal(sig, bankroll=10_000, config=DEFAULT_CFG)
        assert isinstance(result, BetSize)
        assert math.isclose(result.stake, 500.0, abs_tol=1e-6)
        assert math.isclose(result.fraction, 0.05, abs_tol=1e-9)

    def test_zero_stake_on_no_edge(self):
        sig = make_signal(fair_prob=0.50, decimal_odds=1.90)  # edge = -0.05
        result = size_ev_signal(sig, bankroll=10_000, config=DEFAULT_CFG)
        assert result.stake == 0.0
        assert result.fraction == 0.0

    def test_betsize_fields(self):
        # Use raised cap so half-Kelly (0.10) is not capped, isolating field values
        cfg = KellyConfig(kelly_fraction=0.5, max_fraction=0.20, confidence=1.0)
        sig = make_signal(fair_prob=0.60, decimal_odds=2.0)
        result = size_ev_signal(sig, bankroll=10_000, config=cfg)
        assert math.isclose(result.full_kelly, 0.20, abs_tol=1e-9)
        assert math.isclose(result.fraction, 0.10, abs_tol=1e-9)   # half-Kelly, not capped
        assert math.isclose(result.stake, result.fraction * 10_000, abs_tol=1e-6)

    def test_confidence_flows_through(self):
        sig = make_signal(fair_prob=0.60, decimal_odds=2.0)
        cfg_low = KellyConfig(kelly_fraction=0.5, max_fraction=0.20, confidence=0.4)
        result = size_ev_signal(sig, bankroll=10_000, config=cfg_low)
        # full_kelly=0.20, half=0.10, confidence=0.4 → 0.04 → $400
        assert math.isclose(result.stake, 400.0, abs_tol=1e-6)

    def test_cap_reflected_in_fraction(self):
        sig = make_signal(fair_prob=0.80, decimal_odds=3.0)  # full kelly=0.70
        result = size_ev_signal(sig, bankroll=10_000, config=DEFAULT_CFG)
        assert math.isclose(result.fraction, 0.05, abs_tol=1e-9)   # capped
        assert math.isclose(result.stake, 500.0, abs_tol=1e-6)

    def test_expected_profit(self):
        # expected_profit = stake * edge
        sig = make_signal(fair_prob=0.60, decimal_odds=2.0)
        result = size_ev_signal(sig, bankroll=10_000, config=DEFAULT_CFG)
        assert math.isclose(result.expected_profit, result.stake * sig.edge, abs_tol=1e-6)
