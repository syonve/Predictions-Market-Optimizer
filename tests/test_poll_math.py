"""Known-answer tests for poll aggregation math (models/poll.py).

All tests use explicit numerical expected values — no self-referential "compute
and check" patterns. The tolerance for float comparisons is 1e-9 unless noted.
"""

from __future__ import annotations

import math
from datetime import date

from fve.models.poll import (
    AggregationConfig,
    InsufficientDataError,
    Methodology,
    Poll,
    Population,
    aggregate_polls,
    poll_confidence,
    poll_weight,
    size_weight,
    time_weight,
)

TOL = 1e-9
TODAY = date(2026, 1, 15)


# ============================================================================ #
# time_weight
# ============================================================================ #
def test_time_weight_same_day():
    """Poll from today has weight exactly 1.0."""
    assert abs(time_weight(TODAY, TODAY, half_life_days=21.0) - 1.0) < TOL


def test_time_weight_one_half_life():
    """Poll from exactly half_life_days ago has weight ≈ 0.5."""
    field_end = date(2025, 12, 25)  # 21 days before 2026-01-15
    w = time_weight(field_end, TODAY, half_life_days=21.0)
    assert abs(w - 0.5) < 1e-9


def test_time_weight_two_half_lives():
    """Poll from 2 × half_life_days ago has weight ≈ 0.25."""
    field_end = date(2025, 12, 4)  # 42 days before 2026-01-15
    w = time_weight(field_end, TODAY, half_life_days=21.0)
    assert abs(w - 0.25) < 1e-9


def test_time_weight_custom_half_life():
    """Weight with half_life=7 days: 7 days old → 0.5."""
    field_end = date(2026, 1, 8)  # 7 days before TODAY
    w = time_weight(field_end, TODAY, half_life_days=7.0)
    assert abs(w - 0.5) < 1e-9


def test_time_weight_future_poll_clamped_to_1():
    """Future polls (field_end > reference_date) are clamped to weight 1.0."""
    future = date(2026, 2, 1)
    w = time_weight(future, TODAY, half_life_days=21.0)
    assert abs(w - 1.0) < TOL


def test_time_weight_older_polls_lighter():
    """Older polls have strictly lower weight than newer polls."""
    recent = date(2026, 1, 10)   # 5 days old
    older = date(2025, 12, 20)   # 26 days old
    w_recent = time_weight(recent, TODAY)
    w_older = time_weight(older, TODAY)
    assert w_recent > w_older


# ============================================================================ #
# size_weight
# ============================================================================ #
def test_size_weight_at_reference():
    """Poll at reference_n returns weight 1.0."""
    assert abs(size_weight(1000, 1000) - 1.0) < TOL


def test_size_weight_4x_sample():
    """4× sample size returns weight 2.0 (√4 = 2)."""
    assert abs(size_weight(4000, 1000) - 2.0) < TOL


def test_size_weight_quarter_sample():
    """¼ sample size returns weight 0.5 (√0.25 = 0.5)."""
    assert abs(size_weight(250, 1000) - 0.5) < TOL


def test_size_weight_zero_clamped():
    """n=0 is clamped to 1 (avoids sqrt(0) = 0 zero-weighting)."""
    w = size_weight(0, 1000)
    assert w > 0.0
    assert abs(w - size_weight(1, 1000)) < TOL


def test_size_weight_scales_as_sqrt():
    """Verify sqrt scaling: weight(4n) / weight(n) == 2.0."""
    w1 = size_weight(500, 1000)
    w4 = size_weight(2000, 1000)
    assert abs(w4 / w1 - 2.0) < TOL


# ============================================================================ #
# poll_weight composite
# ============================================================================ #
def _make_poll(
    prob: float = 0.50,
    n: int = 1000,
    field_end: date = TODAY,
    pollster: str = "Generic",
    methodology: Methodology = Methodology.LIVE_PHONE,
    population: Population = Population.LIKELY_VOTERS,
    quality_weight: float = 1.0,
) -> Poll:
    return Poll(
        market_key="test",
        selection_key="yes",
        prob=prob,
        n=n,
        field_end=field_end,
        pollster=pollster,
        methodology=methodology,
        population=population,
        quality_weight=quality_weight,
    )


def test_poll_weight_same_day_lv_reference_n():
    """Same-day LV poll at reference_n gets composite weight 1.0."""
    p = _make_poll(n=1000, field_end=TODAY, population=Population.LIKELY_VOTERS)
    config = AggregationConfig(reference_n=1000)
    w = poll_weight(p, TODAY, config)
    assert abs(w - 1.0) < TOL


def test_poll_weight_rv_discount():
    """Registered voters poll is discounted to 0.75× an LV poll."""
    lv = _make_poll(population=Population.LIKELY_VOTERS)
    rv = _make_poll(population=Population.REGISTERED_VOTERS)
    config = AggregationConfig()
    w_lv = poll_weight(lv, TODAY, config)
    w_rv = poll_weight(rv, TODAY, config)
    assert abs(w_rv / w_lv - 0.75) < TOL


def test_poll_weight_adults_discount():
    """Adults poll is discounted to 0.5× an LV poll."""
    lv = _make_poll(population=Population.LIKELY_VOTERS)
    a = _make_poll(population=Population.ADULTS)
    config = AggregationConfig()
    w_lv = poll_weight(lv, TODAY, config)
    w_a = poll_weight(a, TODAY, config)
    assert abs(w_a / w_lv - 0.5) < TOL


def test_poll_weight_quality_multiplier():
    """Quality weight of 0.5 halves the composite weight."""
    full = _make_poll(quality_weight=1.0)
    half = _make_poll(quality_weight=0.5)
    config = AggregationConfig()
    w_full = poll_weight(full, TODAY, config)
    w_half = poll_weight(half, TODAY, config)
    assert abs(w_half / w_full - 0.5) < TOL


# ============================================================================ #
# aggregate_polls
# ============================================================================ #
def test_aggregate_two_equal_polls_returns_mean():
    """Two identical polls with equal weights return the raw probability."""
    polls = [
        _make_poll(prob=0.48, n=1000, field_end=TODAY),
        _make_poll(prob=0.52, n=1000, field_end=TODAY),
    ]
    prob, _ = aggregate_polls(polls, "yes", TODAY, AggregationConfig())
    assert abs(prob - 0.50) < TOL


def test_aggregate_fresher_poll_dominates():
    """A same-day poll should dominate a 42-day-old poll at default half-life=21."""
    stale_date = date(2025, 12, 4)  # 42 days old → time_weight ≈ 0.25
    polls = [
        _make_poll(prob=0.60, n=1000, field_end=TODAY),       # fresh: weight ≈ 1.0
        _make_poll(prob=0.40, n=1000, field_end=stale_date),  # stale: weight ≈ 0.25
    ]
    prob, _ = aggregate_polls(polls, "yes", TODAY, AggregationConfig())
    # Weighted avg: (1.0*0.60 + 0.25*0.40) / 1.25 = (0.60 + 0.10) / 1.25 = 0.56
    expected = (1.0 * 0.60 + 0.25 * 0.40) / (1.0 + 0.25)
    assert abs(prob - expected) < 1e-6


def test_aggregate_larger_sample_dominates():
    """A 4000-sample poll has twice the weight of a 1000-sample poll."""
    polls = [
        _make_poll(prob=0.60, n=4000, field_end=TODAY),  # size_weight = 2.0
        _make_poll(prob=0.40, n=1000, field_end=TODAY),  # size_weight = 1.0
    ]
    prob, _ = aggregate_polls(polls, "yes", TODAY, AggregationConfig())
    # Weighted avg: (2.0*0.60 + 1.0*0.40) / 3.0 = 1.60 / 3.0 ≈ 0.5333...
    expected = (2.0 * 0.60 + 1.0 * 0.40) / 3.0
    assert abs(prob - expected) < 1e-6


def test_aggregate_house_effect_applied():
    """Positive house effect subtracts from raw probability."""
    polls = [
        _make_poll(prob=0.55, n=1000, field_end=TODAY, pollster="RasmussenBias"),
    ]
    config = AggregationConfig(house_effects={"RasmussenBias": 0.05})
    prob, _ = aggregate_polls(polls, "yes", TODAY, config)
    assert abs(prob - 0.50) < TOL


def test_aggregate_house_effect_negative():
    """Negative house effect (polls under-estimate YES) adds to probability."""
    polls = [
        _make_poll(prob=0.45, n=1000, field_end=TODAY, pollster="Underdog"),
    ]
    config = AggregationConfig(house_effects={"Underdog": -0.05})
    prob, _ = aggregate_polls(polls, "yes", TODAY, config)
    assert abs(prob - 0.50) < TOL


def test_aggregate_no_house_effect_for_unknown_pollster():
    """Pollsters not in house_effects dict receive zero correction."""
    polls = [_make_poll(prob=0.55, pollster="NewPollster")]
    config = AggregationConfig(house_effects={"OtherPollster": 0.10})
    prob, _ = aggregate_polls(polls, "yes", TODAY, config)
    assert abs(prob - 0.55) < TOL


def test_aggregate_effective_n_equal_polls():
    """Equal-weight polls: effective_n == n_polls (Kish formula for uniform weights)."""
    polls = [_make_poll(n=1000, field_end=TODAY) for _ in range(4)]
    _, eff_n = aggregate_polls(polls, "yes", TODAY, AggregationConfig())
    # All weights equal → n_eff = (4w)² / (4w²) = 16w² / 4w² = 4
    assert abs(eff_n - 4.0) < 1e-6


def test_aggregate_effective_n_single_poll():
    """Single poll: effective_n == 1.0."""
    polls = [_make_poll(n=1000, field_end=TODAY)]
    _, eff_n = aggregate_polls(polls, "yes", TODAY, AggregationConfig())
    assert abs(eff_n - 1.0) < TOL


def test_aggregate_insufficient_data_raises():
    """InsufficientDataError raised when fewer polls than min_polls."""
    polls = [_make_poll()]
    config = AggregationConfig(min_polls=2)
    try:
        aggregate_polls(polls, "yes", TODAY, config)
        assert False, "expected InsufficientDataError"
    except InsufficientDataError:
        pass


def test_aggregate_wrong_selection_key_raises():
    """InsufficientDataError raised when no polls match the requested selection."""
    polls = [_make_poll(prob=0.55)]  # selection_key = "yes"
    try:
        aggregate_polls(polls, "no", TODAY, AggregationConfig())
        assert False, "expected InsufficientDataError"
    except InsufficientDataError:
        pass


def test_aggregate_probability_clamped_to_unit_interval():
    """Corrected probability is clamped to [0, 1] even with large house effects."""
    # House effect of 0.10 on a 0.05 poll → corrected = -0.05 → clamped to 0.0
    polls = [_make_poll(prob=0.05, pollster="Extreme")]
    config = AggregationConfig(house_effects={"Extreme": 0.10})
    prob, _ = aggregate_polls(polls, "yes", TODAY, config)
    assert 0.0 <= prob <= 1.0


# ============================================================================ #
# poll_confidence
# ============================================================================ #
def test_poll_confidence_at_reference():
    """effective_n == confidence_reference_n → confidence == 1.0."""
    assert abs(poll_confidence(5000.0, 5000.0) - 1.0) < TOL


def test_poll_confidence_zero_eff_n():
    """effective_n == 0 → confidence == 0.0."""
    assert abs(poll_confidence(0.0, 5000.0) - 0.0) < TOL


def test_poll_confidence_quarter_reference():
    """effective_n == reference/4 → confidence == 0.5 (sqrt scaling)."""
    assert abs(poll_confidence(1250.0, 5000.0) - 0.5) < TOL


def test_poll_confidence_capped_at_1():
    """effective_n > reference → confidence capped at 1.0."""
    assert abs(poll_confidence(99999.0, 5000.0) - 1.0) < TOL


def test_poll_confidence_negative_eff_n_treated_as_zero():
    """Negative effective_n is treated as 0 (defensive)."""
    assert abs(poll_confidence(-1.0, 5000.0) - 0.0) < TOL


# ============================================================================ #
# Poll validation
# ============================================================================ #
def test_poll_invalid_prob_raises():
    try:
        Poll(
            market_key="m", selection_key="yes", prob=1.5, n=500,
            field_end=TODAY, pollster="P",
            methodology=Methodology.LIVE_PHONE,
            population=Population.LIKELY_VOTERS,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_poll_zero_n_raises():
    try:
        Poll(
            market_key="m", selection_key="yes", prob=0.50, n=0,
            field_end=TODAY, pollster="P",
            methodology=Methodology.LIVE_PHONE,
            population=Population.LIKELY_VOTERS,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_poll_zero_quality_weight_raises():
    try:
        Poll(
            market_key="m", selection_key="yes", prob=0.50, n=500,
            field_end=TODAY, pollster="P",
            methodology=Methodology.LIVE_PHONE,
            population=Population.LIKELY_VOTERS,
            quality_weight=0.0,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    # Run all test_ functions directly (no pytest required)
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:
                print(f"  FAIL  {name}: {exc}", file=sys.stderr)
                failed += 1
    print(f"\n{'All tests passed.' if failed == 0 else f'{failed} test(s) failed.'}")
    sys.exit(failed)
