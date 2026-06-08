"""Known-answer tests for the devig methods.

Each method is checked against an independently-derived reference, not against
its own implementation:

  proportional  closed form p_i = q_i / B computed by hand.
  shin          a forward generative model (algebraic inverse of the solver's
                formula) builds odds from KNOWN (p, z); the solver must recover
                both p and z.
  power         odds built from KNOWN (p, k) via q_i = p_i**(1/k); the solver
                must recover p and k.

Plus invariants: sum-to-one, symmetry, and the favorite-longshot direction
that distinguishes Shin from proportional.
"""

from __future__ import annotations

import math

import pytest

from fve.pricing import devig
from fve.pricing.devig import (
    booksum,
    implied_probs,
    overround,
    power,
    proportional,
    shin,
)
from fve.pricing.devig import shin_z

ABS = 1e-9


def _odds(probs):
    return [1.0 / p for p in probs]


# --------------------------------------------------------------------------- #
# proportional: closed form
# --------------------------------------------------------------------------- #
def test_proportional_closed_form():
    # odds chosen so the answer is exact: q = [0.625, 0.41667], B = 1.041667
    odds = [1.6, 2.4]
    p = devig(odds, method="proportional")
    assert p == pytest.approx((0.6, 0.4), abs=ABS)
    assert math.isclose(sum(p), 1.0, abs_tol=ABS)


def test_proportional_three_way():
    odds = [2.0, 4.0, 4.0]  # q = [0.5, 0.25, 0.25], B = 1.0 -> already fair
    p = proportional(implied_probs(odds))
    assert p == pytest.approx((0.5, 0.25, 0.25), abs=ABS)


# --------------------------------------------------------------------------- #
# shin: round-trip against the forward generative model
# --------------------------------------------------------------------------- #
def _shin_forward(p, z):
    """Build self-consistent decimal odds from true probs p and insider z.

    Inverting the solver formula gives  q_i**2 / B = (1-z) p_i**2 + z p_i.
    Self-consistency (B = sum q_i) forces  B = C**2  with
    C = sum_i sqrt((1-z) p_i**2 + z p_i), hence q_i = C * sqrt(...).
    """
    roots = [math.sqrt((1.0 - z) * pi * pi + z * pi) for pi in p]
    c = sum(roots)
    q = [c * r for r in roots]
    return [1.0 / qi for qi in q]


@pytest.mark.parametrize(
    "p, z",
    [
        ((0.7, 0.3), 0.05),
        ((0.55, 0.45), 0.02),
        ((0.8, 0.2), 0.10),
        ((0.5, 0.3, 0.2), 0.04),
    ],
)
def test_shin_recovers_known_probs_and_z(p, z):
    odds = _shin_forward(p, z)
    assert booksum(odds) > 1.0  # the construction must produce a real vig
    recovered = shin(implied_probs(odds))
    assert recovered == pytest.approx(p, abs=1e-7)
    assert shin_z(implied_probs(odds)) == pytest.approx(z, abs=1e-6)


def test_shin_requires_overround():
    with pytest.raises(ValueError):
        shin([0.5, 0.4])  # book sum 0.9 < 1


# --------------------------------------------------------------------------- #
# power: round-trip against q_i = p_i ** (1/k)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "p, k",
    [
        ((0.6, 0.4), 1.10),
        ((0.75, 0.25), 1.20),
        ((0.5, 0.3, 0.2), 1.08),
    ],
)
def test_power_recovers_known_probs(p, k):
    q = [pi ** (1.0 / k) for pi in p]
    odds = [1.0 / qi for qi in q]
    assert booksum(odds) > 1.0
    recovered = power(implied_probs(odds))
    assert recovered == pytest.approx(p, abs=1e-7)


# --------------------------------------------------------------------------- #
# shared invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["proportional", "shin", "power"])
def test_sums_to_one(method):
    # every book carries a positive vig (book sum > 1) so all methods apply
    for odds in ([1.91, 1.91], [1.4, 3.1], [2.5, 3.2, 3.4], [1.05, 19.0]):
        p = devig(odds, method=method)
        assert math.isclose(sum(p), 1.0, abs_tol=1e-9)
        assert all(0.0 < x < 1.0 for x in p)


@pytest.mark.parametrize("method", ["proportional", "shin", "power"])
def test_symmetric_book_is_even(method):
    # equal odds -> equal fair probabilities for every method
    p = devig([1.91, 1.91], method=method)
    assert p == pytest.approx((0.5, 0.5), abs=1e-9)
    p3 = devig([2.85, 2.85, 2.85], method=method)
    assert p3 == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=1e-9)


def test_shin_corrects_favorite_longshot_vs_proportional():
    # Shin pushes the favorite UP and the longshot DOWN relative to proportional.
    odds = [1.37973, 3.07901]  # an asymmetric book with ~5% vig
    prop = proportional(implied_probs(odds))
    sh = shin(implied_probs(odds))
    assert sh[0] > prop[0]  # favorite gets more probability under Shin
    assert sh[1] < prop[1]  # longshot gets less


# --------------------------------------------------------------------------- #
# dispatch / helpers
# --------------------------------------------------------------------------- #
def test_unknown_method_raises():
    with pytest.raises(ValueError):
        devig([1.9, 1.9], method="bogus")  # type: ignore[arg-type]


def test_overround_and_booksum():
    odds = [1.91, 1.91]
    assert booksum(odds) == pytest.approx(2 / 1.91, abs=ABS)
    assert overround(odds) == pytest.approx(2 / 1.91 - 1.0, abs=ABS)
