"""Poll data model and weighted aggregation math.

All functions here are pure (no IO, no side effects). The PollingModelProvider
in polling.py handles data ingestion and wires these functions together.

Weighting scheme
----------------
Each poll is assigned a composite weight:

    w = time_weight × size_weight × population_weight × quality_weight

where:

  time_weight    = exp(−λ × age_days),  λ = ln(2) / half_life_days
                   Exponential decay: polls lose half their weight every
                   ``half_life_days`` days. Default 21 days. Day-of-field-end
                   polls get weight = 1.0; a 21-day-old poll gets weight ≈ 0.5.

  size_weight    = sqrt(n / reference_n)
                   Square-root scaling: doubling the sample size increases
                   weight by √2, not 2. Consistent with polling-error theory
                   where MOE scales as 1/√n. Default reference_n = 1000.

  population_weight  LIKELY_VOTERS=1.0, REGISTERED_VOTERS=0.75, ADULTS=0.5
                   LV screens are the most predictive of actual outcomes;
                   adults polls measure opinion, not voter behavior.

  quality_weight     Pollster-level quality score in (0, 1]. Defaults to 1.0.
                   Callers may inject FiveThirtyEight grades or their own
                   scoring. Not a house-effect correction — see below.

House-effect correction
-----------------------
A house effect is a systematic partisan lean in a pollster's results
(e.g. Rasmussen consistently shows Republicans +3 vs. the consensus).
It is applied as an additive offset BEFORE weighting:

    corrected_prob = raw_prob − house_effects.get(pollster, 0.0)

Convention: positive house_effects entry = pollster over-estimates the YES
selection; negative = pollster under-estimates it. Start with an empty dict
and calibrate from historical data.

Effective N
-----------
The Kish (1965) effective sample size formula approximates the precision of
a weighted aggregate as if it came from an unweighted sample of size n_eff:

    n_eff = (Σ w_i)² / Σ w_i²

This is converted to a [0,1] confidence score by:

    confidence = min(1, sqrt(n_eff / CONFIDENCE_REFERENCE_N))

where CONFIDENCE_REFERENCE_N = 5000 (default). Interpretation: an aggregate
equivalent to ~5000 unweighted polls earns confidence = 1.0.

Flagged decisions
-----------------
1. **Direct topline as probability.** ``Poll.prob`` is treated as the
   probability that the selection_key outcome occurs. This is exact when the
   poll directly asks the binary question ("Will X win?") and valid as a
   proxy when the question is "Who do you prefer?" and the binary market is
   "Will X win?". It is NOT valid for multi-candidate races where you need
   an electoral simulation step — the topline vote share is not the win
   probability. Multi-candidate markets should feed a simulation layer and
   pass the simulated win prob as the Poll.prob. Flag this if you see races
   where 3+ candidates have non-trivial shares.

2. **Half-life default = 21 days.** Appropriate for state-level Senate/gov
   races 1–3 months out. Use shorter half-lives (7–14 days) for national
   races with heavy polling volume close to election day; longer (28–42 days)
   for early-cycle or low-polling markets.

3. **House effects are caller-supplied, default zero.** The system does not
   auto-calibrate house effects from data. Callers should maintain a
   ``house_effects`` dict per-election-cycle, not globally, because pollster
   bias shifts year to year.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CONFIDENCE_REFERENCE_N: float = 5000.0
"""Effective-N at which confidence reaches 1.0. Adjust to taste."""


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Methodology(str, Enum):
    """Data-collection methodology of a poll."""

    LIVE_PHONE = "live_phone"      # Human-conducted telephone
    ONLINE_PANEL = "online_panel"  # Opt-in online panel
    IVR = "ivr"                    # Interactive voice response (robopoll)
    TEXT = "text"                  # SMS-based
    MIXED = "mixed"                # Multiple methodologies blended


class Population(str, Enum):
    """Target population of a poll."""

    LIKELY_VOTERS = "lv"       # Most predictive of election outcomes
    REGISTERED_VOTERS = "rv"   # Broader; less predictive than LV
    ADULTS = "adults"          # Least predictive; measures opinion only


# Population quality weights: LV polls are most predictive.
_POPULATION_WEIGHT: dict[Population, float] = {
    Population.LIKELY_VOTERS: 1.0,
    Population.REGISTERED_VOTERS: 0.75,
    Population.ADULTS: 0.5,
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Poll:
    """A single survey result for one selection in one market.

    Each row in a polling database maps to one Poll per tracked selection.
    For a binary Kalshi market ("Will X win?"), you typically have one Poll
    per survey with ``selection_key="yes"`` and ``prob=topline_share``.

    Parameters
    ----------
    market_key:
        The ``Market.key`` this poll covers.
    selection_key:
        Which selection the probability refers to (e.g. ``"yes"``).
    prob:
        Probability in [0, 1] for ``selection_key``. For a head-to-head poll,
        this is typically the two-way vote share or the stated probability from
        a direct-question poll.
    n:
        Sample size (number of respondents).
    field_end:
        Last day of field work. Used for recency weighting.
    pollster:
        Pollster name (used for house-effect lookup and quality weighting).
    methodology:
        How the poll was conducted.
    population:
        Target population screened for.
    sponsor:
        Optional poll sponsor (used for identifying partisan polls).
    quality_weight:
        Pre-assigned pollster quality score in (0, 1]. Defaults to 1.0.
        Callers may map FiveThirtyEight grades (A+=1.0, A=0.9, B=0.75,
        C=0.5, unknown=0.6) or their own scheme.
    """

    market_key: str
    selection_key: str
    prob: float
    n: int
    field_end: date
    pollster: str
    methodology: Methodology
    population: Population
    sponsor: str | None = None
    quality_weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"Poll.prob must be in [0,1], got {self.prob}")
        if self.n < 1:
            raise ValueError(f"Poll.n must be >= 1, got {self.n}")
        if not 0.0 < self.quality_weight <= 1.0:
            raise ValueError(
                f"Poll.quality_weight must be in (0,1], got {self.quality_weight}"
            )


@dataclass
class AggregationConfig:
    """Configuration for poll aggregation.

    Parameters
    ----------
    half_life_days:
        Recency decay half-life in days. Polls ``half_life_days`` old receive
        half the time weight of a same-day poll.
    reference_n:
        Normalisation reference for sample-size weighting. A poll with
        ``n = reference_n`` gets size_weight = 1.0.
    min_polls:
        Minimum number of polls required to return an estimate. Raise
        ``InsufficientDataError`` if fewer polls are available.
    house_effects:
        ``{pollster_name: bias}`` mapping. Bias is subtracted from the raw
        poll probability before weighting. Positive = pollster over-estimates
        the YES selection.
    confidence_reference_n:
        Effective-N at which confidence reaches 1.0.
    """

    half_life_days: float = 21.0
    reference_n: int = 1000
    min_polls: int = 1
    house_effects: dict[str, float] = field(default_factory=dict)
    confidence_reference_n: float = CONFIDENCE_REFERENCE_N

    def __post_init__(self) -> None:
        if self.half_life_days <= 0:
            raise ValueError(f"half_life_days must be > 0, got {self.half_life_days}")
        if self.reference_n < 1:
            raise ValueError(f"reference_n must be >= 1, got {self.reference_n}")
        if self.min_polls < 1:
            raise ValueError(f"min_polls must be >= 1, got {self.min_polls}")


class InsufficientDataError(ValueError):
    """Raised when fewer polls are available than AggregationConfig.min_polls."""


# --------------------------------------------------------------------------- #
# Pure weighting functions
# --------------------------------------------------------------------------- #
def time_weight(
    field_end: date,
    reference_date: date,
    half_life_days: float = 21.0,
) -> float:
    """Exponential recency weight for a poll.

    Parameters
    ----------
    field_end:
        Last field date of the poll.
    reference_date:
        The date to measure age against (typically today).
    half_life_days:
        Number of days after which weight halves.

    Returns
    -------
    float
        Weight in (0, 1]. Returns 1.0 for same-day polls; 0.5 for polls
        exactly half_life_days old; never returns 0 (future polls are clamped
        to age=0).
    """
    age = max(0, (reference_date - field_end).days)
    return math.exp(-math.log(2.0) * age / half_life_days)


def size_weight(n: int, reference_n: int = 1000) -> float:
    """Square-root sample-size weight.

    A poll with n=reference_n returns 1.0. Doubling sample size returns √2,
    consistent with polling MOE scaling as 1/√n.

    Parameters
    ----------
    n:
        Sample size (clamped to 1 if zero or negative).
    reference_n:
        The normalisation reference (weight = 1.0 at this n).
    """
    return math.sqrt(max(1, n) / reference_n)


def poll_weight(
    poll: Poll,
    reference_date: date,
    config: AggregationConfig,
) -> float:
    """Composite weight for a single poll.

    Combines time, sample-size, population, and quality dimensions
    multiplicatively. All factors are non-negative, so the composite weight
    is also non-negative.
    """
    tw = time_weight(poll.field_end, reference_date, config.half_life_days)
    sw = size_weight(poll.n, config.reference_n)
    pw = _POPULATION_WEIGHT.get(poll.population, 1.0)
    return tw * sw * pw * poll.quality_weight


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate_polls(
    polls: list[Poll],
    selection_key: str,
    reference_date: date,
    config: AggregationConfig,
) -> tuple[float, float]:
    """Weighted-average poll probability with house-effect correction.

    Parameters
    ----------
    polls:
        All polls for the market. Polls not matching ``selection_key`` are
        silently ignored.
    selection_key:
        Which selection to aggregate (e.g. ``"yes"``).
    reference_date:
        Date used for recency weighting (typically today).
    config:
        Aggregation settings including house effects and min_polls.

    Returns
    -------
    (prob, effective_n) : tuple[float, float]
        ``prob`` is the weighted-average probability, clamped to [0, 1].
        ``effective_n`` is the Kish effective sample size (number of polls,
        not respondents) measuring the precision of the aggregate.

    Raises
    ------
    InsufficientDataError
        If the number of relevant polls is below ``config.min_polls``.
    """
    relevant = [p for p in polls if p.selection_key == selection_key]

    if len(relevant) < config.min_polls:
        raise InsufficientDataError(
            f"Need >= {config.min_polls} polls for selection {selection_key!r}, "
            f"found {len(relevant)}"
        )

    weights = [poll_weight(p, reference_date, config) for p in relevant]
    total_w = sum(weights)

    # House-effect correction: subtract known pollster bias before weighting
    corrected_probs = [
        p.prob - config.house_effects.get(p.pollster, 0.0)
        for p in relevant
    ]

    weighted_prob = sum(w * cp for w, cp in zip(weights, corrected_probs)) / total_w

    # Kish effective N: measures how many unweighted polls the aggregate
    # is equivalent to in terms of precision.
    sum_w2 = sum(w * w for w in weights)
    effective_n = (total_w * total_w) / sum_w2 if sum_w2 > 0 else 0.0

    return max(0.0, min(1.0, weighted_prob)), effective_n


def poll_confidence(
    effective_n: float,
    confidence_reference_n: float = CONFIDENCE_REFERENCE_N,
) -> float:
    """Map Kish effective-N to a [0, 1] confidence score.

    Uses square-root scaling so confidence grows quickly at first and
    asymptotes toward 1.0:

        confidence = min(1.0, sqrt(effective_n / confidence_reference_n))

    At effective_n == confidence_reference_n, confidence == 1.0.
    At effective_n == 0, confidence == 0.0.

    Parameters
    ----------
    effective_n:
        Kish effective number of polls from ``aggregate_polls``.
    confidence_reference_n:
        Effective-N that maps to confidence = 1.0.
    """
    if confidence_reference_n <= 0:
        raise ValueError(
            f"confidence_reference_n must be > 0, got {confidence_reference_n}"
        )
    return min(1.0, math.sqrt(max(0.0, effective_n) / confidence_reference_n))
