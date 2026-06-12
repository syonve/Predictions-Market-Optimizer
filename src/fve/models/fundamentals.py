"""Election-fundamentals model: incumbency, fundraising, primary results.

Plain-English summary
---------------------
Polls measure what voters *say*; fundamentals measure the structural facts of
a race that historically predict outcomes even before (or without) good
polling: whether a candidate is the sitting officeholder, who is winning the
money race, and how convincingly they won their own party's primary. This
module turns those three facts into a probability estimate that the consensus
engine can blend alongside market prices and the polling model.

How the math works
------------------
All adjustments happen in *log-odds* space. Log-odds (the "logit") is the
natural scale for stacking independent-ish signals: adding a fixed bump moves
a 50/50 race a lot and an already-90/10 race only a little, which is the
behavior you want — a fundraising advantage matters less when the outcome is
nearly certain anyway.

    logit(p) = logit(prior) + incumbency + fundraising + primary
    p        = sigmoid(of the sum)

Each signal contributes a bounded, configurable bump:

  incumbency   ±incumbency_coef. Sitting officeholders out-perform; the
               candidate being an incumbent adds the coef, facing one
               subtracts it, an open seat adds nothing.

  fundraising  fundraising_coef[race_level] × (2·money_share − 1), where
               money_share = own_total / (own_total + opponent_total).
               Even money → 0; total money dominance → the full coefficient.
               The coefficient DEPENDS ON THE RACE LEVEL: money is weighted
               most in LOCAL races, then STATE, then FEDERAL — in
               low-information down-ballot races, fundraising buys the name
               recognition that decides the race, while federal races are
               saturated with free information and money hits diminishing
               returns.

  primary      primary_coef × primary_margin, where primary_margin is the
               candidate's primary vote share minus their nearest rival's
               (in [-1, 1]). A dominant primary win signals candidate quality
               and party unity; a squeaker signals a divided base. Margin
               over the runner-up (not raw share) keeps crowded multi-way
               primaries from punishing a clear winner with a sub-50% share.

Flagged decisions (all configurable via FundamentalsConfig)
-----------------------------------------------------------
1. **Default coefficient magnitudes are priors, not estimates.** They are
   set conservatively from the public literature (modern incumbency advantage
   ~2–3 points of vote share; fundraising largely endogenous at the federal
   level) but have NOT been fit to data. Before trusting this model's
   deviations as edge, fit/validate the coefficients on historical races —
   the calibration layer (Brier, reliability) is the gate, same as every
   other model.
       incumbency_coef     0.40  (50% → ~59.9% for an otherwise even race)
       fundraising_coefs   FEDERAL 0.25 · STATE 0.50 · LOCAL 0.80
       primary_coef        0.30  (a +30-point primary win → 50% → ~52.2%)

2. **Fundraising uses money share, not log-ratio.** Share is bounded, robust
   to one side reporting zero, and symmetric. The known limitation: it
   ignores absolute scale ($80k vs $20k reads the same as $8M vs $2M).

3. **Fundraising is also endogenous** — donors give to likely winners, so
   part of the money signal is the market's opinion recycled. This is the
   main reason the FEDERAL coefficient is small. If you later add a
   poll-conditional version (fundraising | polls), revisit these weights.

4. **Confidence is signal-count based**: max_confidence × (signals present)/3,
   capped at max_confidence = 0.5 by default — deliberately below a
   well-polled polling aggregate (which can reach 1.0), because fundamentals
   are coarse, slow-moving priors. As with the polling model, confidence is
   not yet plumbed into consensus weights.

5. **Binary markets only (v0).** Multi-way races need a different treatment
   (per-candidate fundamentals and a normalization step); estimate() returns
   None for markets with more than two selections rather than guessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from fve.models.base import ModelEstimate
from fve.types import Market, Venue, VenueKind

# Synthetic venue for fundamentals-derived snapshots. Distinct key from the
# polling model so consensus venue_probs diagnostics keep them separate.
FUNDAMENTALS_VENUE = Venue(
    key="fundamentals_model",
    name="Fundamentals Model",
    kind=VenueKind.MODEL,
)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class RaceLevel(str, Enum):
    """Jurisdiction level of the race — controls the fundraising weight."""

    FEDERAL = "federal"   # President, US Senate, US House
    STATE = "state"       # Governor, state legislature, statewide offices
    LOCAL = "local"       # Mayor, county, school board, ballot measures


class Incumbency(str, Enum):
    """The described candidate's relationship to the seat."""

    INCUMBENT = "incumbent"     # candidate currently holds the seat
    CHALLENGER = "challenger"   # candidate is running against the incumbent
    OPEN_SEAT = "open_seat"     # no incumbent in the race (or unknown)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RaceFundamentals:
    """Structural facts about one race, from the perspective of one selection.

    Parameters
    ----------
    market_key:
        The ``Market.key`` this race maps to (must match the venue ticker).
    selection_key:
        Which selection the fundamentals describe — for a Kalshi binary
        "Will X win?", use ``"yes"`` with X's fundamentals.
    race_level:
        FEDERAL / STATE / LOCAL. Selects the fundraising coefficient.
    incumbency:
        Whether the described candidate holds the seat, challenges the
        holder, or the seat is open. Defaults to OPEN_SEAT (no signal).
    fundraising_own / fundraising_opponent:
        Total dollars raised by the described candidate and their opponent.
        Same units, same as-of date. None (or both zero) = no signal.
    primary_margin:
        Candidate's primary vote share minus their nearest rival's share,
        in [-1, 1]. None = no primary signal (e.g. unopposed, no primary yet).
    prior_prob:
        Starting probability before any fundamentals, in (0, 1). Defaults to
        0.5 (a tossup). Supply a partisan-lean base rate here if you have one
        (e.g. the seat's historical two-party share).
    """

    market_key: str
    selection_key: str
    race_level: RaceLevel
    incumbency: Incumbency = Incumbency.OPEN_SEAT
    fundraising_own: float | None = None
    fundraising_opponent: float | None = None
    primary_margin: float | None = None
    prior_prob: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.prior_prob < 1.0:
            raise ValueError(f"prior_prob must be in (0,1), got {self.prior_prob}")
        if self.primary_margin is not None and not -1.0 <= self.primary_margin <= 1.0:
            raise ValueError(
                f"primary_margin must be in [-1,1], got {self.primary_margin}"
            )
        for label, v in (
            ("fundraising_own", self.fundraising_own),
            ("fundraising_opponent", self.fundraising_opponent),
        ):
            if v is not None and v < 0.0:
                raise ValueError(f"{label} must be >= 0, got {v}")


@dataclass
class FundamentalsConfig:
    """Coefficients for the fundamentals model (all in log-odds units).

    See the module docstring for the reasoning behind the defaults — they are
    conservative priors, NOT fitted estimates, and should be revisited once
    the calibration layer has historical races to score against.
    """

    incumbency_coef: float = 0.40
    fundraising_coefs: dict[RaceLevel, float] = field(
        default_factory=lambda: {
            RaceLevel.FEDERAL: 0.25,
            RaceLevel.STATE: 0.50,
            RaceLevel.LOCAL: 0.80,
        }
    )
    primary_coef: float = 0.30
    max_confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.incumbency_coef < 0:
            raise ValueError(f"incumbency_coef must be >= 0, got {self.incumbency_coef}")
        if self.primary_coef < 0:
            raise ValueError(f"primary_coef must be >= 0, got {self.primary_coef}")
        if any(c < 0 for c in self.fundraising_coefs.values()):
            raise ValueError("fundraising_coefs must all be >= 0")
        if not 0.0 <= self.max_confidence <= 1.0:
            raise ValueError(f"max_confidence must be in [0,1], got {self.max_confidence}")


# --------------------------------------------------------------------------- #
# Pure math
# --------------------------------------------------------------------------- #
def logit(p: float) -> float:
    """Log-odds of a probability in the open interval (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"logit requires p in (0,1), got {p}")
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Inverse of logit: maps log-odds back to a probability in (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def incumbency_bump(incumbency: Incumbency, coef: float) -> float:
    """Log-odds bump for incumbency status: +coef / -coef / 0."""
    if incumbency is Incumbency.INCUMBENT:
        return coef
    if incumbency is Incumbency.CHALLENGER:
        return -coef
    return 0.0


def fundraising_bump(
    own: float | None,
    opponent: float | None,
    coef: float,
) -> float:
    """Log-odds bump from the money race: coef × (2·share − 1).

    share = own / (own + opponent). Even money → 0; all the money → +coef;
    none of the money → -coef. Returns 0.0 when either total is missing or
    nothing has been raised on both sides (no signal).
    """
    if own is None or opponent is None:
        return 0.0
    if own < 0.0 or opponent < 0.0:
        raise ValueError(f"fundraising totals must be >= 0, got {own}, {opponent}")
    total = own + opponent
    if total <= 0.0:
        return 0.0
    share = own / total
    return coef * (2.0 * share - 1.0)


def primary_bump(margin: float | None, coef: float) -> float:
    """Log-odds bump from primary performance: coef × margin.

    margin is the candidate's primary share minus their nearest rival's,
    in [-1, 1]. None = no signal.
    """
    if margin is None:
        return 0.0
    if not -1.0 <= margin <= 1.0:
        raise ValueError(f"primary_margin must be in [-1,1], got {margin}")
    return coef * margin


def fundamentals_probability(
    race: RaceFundamentals,
    config: FundamentalsConfig,
) -> tuple[float, int]:
    """Combine all available fundamentals into a probability.

    Returns
    -------
    (prob, n_signals) : tuple[float, int]
        ``prob`` is the model probability for ``race.selection_key``.
        ``n_signals`` counts how many of the three signals were actually
        present (incumbency != OPEN_SEAT, fundraising data, primary data) —
        used for the confidence score.
    """
    x = logit(race.prior_prob)
    n_signals = 0

    if race.incumbency is not Incumbency.OPEN_SEAT:
        x += incumbency_bump(race.incumbency, config.incumbency_coef)
        n_signals += 1

    fund_coef = config.fundraising_coefs.get(race.race_level, 0.0)
    has_money = (
        race.fundraising_own is not None
        and race.fundraising_opponent is not None
        and (race.fundraising_own + race.fundraising_opponent) > 0.0
    )
    if has_money:
        x += fundraising_bump(race.fundraising_own, race.fundraising_opponent, fund_coef)
        n_signals += 1

    if race.primary_margin is not None:
        x += primary_bump(race.primary_margin, config.primary_coef)
        n_signals += 1

    return sigmoid(x), n_signals


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #
class FundamentalsModelProvider:
    """ModelProvider backed by a list of RaceFundamentals.

    Satisfies the same ``ModelProvider`` protocol as PollingModelProvider, so
    the consensus engine consumes both identically. Binary (two-selection)
    markets only in v0; multi-way markets return None.
    """

    def __init__(
        self,
        races: list[RaceFundamentals],
        config: FundamentalsConfig | None = None,
    ) -> None:
        self._races = list(races)
        self._config = config or FundamentalsConfig()

    # --- ModelProvider interface ---

    @property
    def model_name(self) -> str:
        return "election_fundamentals"

    def estimate(self, market: Market) -> ModelEstimate | None:
        """Produce a fundamentals-based estimate for ``market``.

        Returns None when no fundamentals are loaded for the market key,
        the market is not binary, or the described selection is unknown.
        """
        race = next((r for r in self._races if r.market_key == market.key), None)
        if race is None:
            return None
        if len(market.selections) != 2:
            return None

        keys = [s.key for s in market.selections]
        if race.selection_key not in keys:
            return None
        other_key = next(k for k in keys if k != race.selection_key)

        prob, n_signals = fundamentals_probability(race, self._config)
        if n_signals == 0:
            # Prior-only "estimate" carries no information; stay silent
            # rather than injecting a bare 50/50 into consensus.
            return None

        confidence = self._config.max_confidence * (n_signals / 3.0)

        return ModelEstimate(
            market=market,
            selection_probs={race.selection_key: prob, other_key: 1.0 - prob},
            confidence=confidence,
            n_samples=n_signals,
            model_name=self.model_name,
            computed_at=datetime.now(tz=timezone.utc),
            venue=FUNDAMENTALS_VENUE,
        )
