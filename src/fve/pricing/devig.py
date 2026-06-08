"""Devigging: convert raw implied probabilities into a fair distribution.

A bookmaker's quoted odds embed a margin (the "vig"), so raw implied
probabilities q_i = 1/odds_i sum to the booksum B > 1. Devigging removes the
margin to recover a probability distribution p_i that sums to 1.

Three methods, swappable via `devig(..., method=...)`. They differ in HOW the
margin is attributed across outcomes, which changes the favorite-longshot
profile of the result — and therefore which edges surface downstream:

  proportional  Margin removed in proportion to implied probability
                (p_i = q_i / B). Transparent baseline; does not correct
                favorite-longshot bias.

  shin          Attributes margin to a fraction z of informed ("insider")
                volume and backs it out. Loads more margin onto favorites,
                correcting favorite-longshot bias in the empirically right
                direction. Good default for 2-way sharp-anchored markets.

  power         Fits an exponent k so that sum(q_i**k) = 1. Most flexible,
                frequently best-calibrated empirically (esp. many-outcome).

All functions take the raw implied probabilities (q_i = 1/odds_i, summing to
B > 1) and return a tuple of fair probabilities summing to 1. Use `devig()` to
go straight from decimal odds.

The "best" method is an empirical question: pick it by Brier score /
reliability on the markets actually traded (see tracking/calibration). The
default here is a starting point, not a verdict.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import math

DevigMethod = Literal["proportional", "shin", "power"]

_TOL = 1e-12
_MAX_ITER = 200


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def implied_probs(odds: Sequence[float]) -> tuple[float, ...]:
    """Raw implied probabilities q_i = 1/odds_i (still include the vig)."""
    return tuple(1.0 / o for o in odds)


def booksum(odds: Sequence[float]) -> float:
    """Sum of raw implied probabilities (a.k.a. the book sum / overround+1)."""
    return sum(1.0 / o for o in odds)


def overround(odds: Sequence[float]) -> float:
    """The bookmaker margin: booksum - 1 (negative implies a single-venue arb)."""
    return booksum(odds) - 1.0


def _bisect(f: Callable[[float], float], lo: float, hi: float) -> float:
    """Bisection root-find for a monotone f with a sign change on [lo, hi]."""
    f_lo = f(lo)
    for _ in range(_MAX_ITER):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < _TOL or 0.5 * (hi - lo) < _TOL:
            return mid
        if (f_mid > 0.0) == (f_lo > 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _validate(probs: Sequence[float]) -> None:
    if len(probs) < 2:
        raise ValueError("devig needs at least 2 outcomes")
    if any(p <= 0.0 for p in probs):
        raise ValueError("implied probabilities must be strictly positive")


def _normalize(probs: list[float]) -> tuple[float, ...]:
    s = sum(probs)
    return tuple(p / s for p in probs)


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
def proportional(probs: Sequence[float]) -> tuple[float, ...]:
    """p_i = q_i / sum(q). Margin removed in proportion to implied probability."""
    _validate(probs)
    return _normalize(list(probs))


def power(probs: Sequence[float]) -> tuple[float, ...]:
    """Find exponent k with sum(q_i**k) = 1; return q_i**k.

    sum(q_i**k) is strictly decreasing in k (each q_i in (0,1)), so the root
    is unique. Works for B > 1 (k > 1) and the degenerate B < 1 (k < 1).
    """
    _validate(probs)
    s = sum(probs)
    if abs(s - 1.0) < _TOL:
        return tuple(probs)

    def f(k: float) -> float:
        return sum(p**k for p in probs) - 1.0

    lo, hi = 1e-9, 1.0  # f(lo) ~= n-1 > 0
    while f(hi) > 0.0:  # push hi until sum(q**hi) < 1
        hi *= 2.0
    k = _bisect(f, lo, hi)
    return _normalize([p**k for p in probs])


def shin(probs: Sequence[float]) -> tuple[float, ...]:
    """Shin's method: back out the informed-money fraction z, then fair probs.

    For book sum B = sum(q_j), the fair probability of outcome i is

        p_i(z) = [ sqrt(z**2 + 4(1-z) * q_i**2 / B) - z ] / (2(1-z)),

    with z in [0, 1) chosen so that sum_i p_i = 1. sum_i p_i(z) decreases from
    sqrt(B) > 1 at z = 0, so a root exists for normal books. If no crossing is
    found (degenerate / extreme book), falls back to proportional.
    """
    _validate(probs)
    b = sum(probs)
    if abs(b - 1.0) < _TOL:
        return tuple(probs)
    if b < 1.0:
        raise ValueError("Shin's method requires a book sum > 1 (positive vig)")

    def p_of_z(z: float) -> list[float]:
        a = 2.0 * (1.0 - z)
        return [(math.sqrt(z * z + 2.0 * a * (q * q / b)) - z) / a for q in probs]

    def g(z: float) -> float:
        return sum(p_of_z(z)) - 1.0

    hi = 1.0 - 1e-12
    if g(hi) > 0.0:  # no sign change -> degenerate book, fall back
        return proportional(probs)
    z = _bisect(g, 0.0, hi)
    return _normalize(p_of_z(z))


def shin_z(probs: Sequence[float]) -> float:
    """The fitted informed-money fraction z for Shin's method (diagnostic)."""
    _validate(probs)
    b = sum(probs)
    if b <= 1.0:
        return 0.0

    def g(z: float) -> float:
        a = 2.0 * (1.0 - z)
        return sum((math.sqrt(z * z + 2.0 * a * (q * q / b)) - z) / a for q in probs) - 1.0

    hi = 1.0 - 1e-12
    if g(hi) > 0.0:
        return 0.0
    return _bisect(g, 0.0, hi)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
_METHODS: dict[str, Callable[[Sequence[float]], tuple[float, ...]]] = {
    "proportional": proportional,
    "shin": shin,
    "power": power,
}


def devig(odds: Sequence[float], method: DevigMethod = "shin") -> tuple[float, ...]:
    """Devig decimal odds into fair probabilities summing to 1.

    `method` defaults to Shin's for 2-way sharp-anchored markets; the
    calibration layer may select a different method per market class.
    """
    try:
        fn = _METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown devig method {method!r}; choose from {sorted(_METHODS)}"
        ) from None
    return fn(implied_probs(odds))
