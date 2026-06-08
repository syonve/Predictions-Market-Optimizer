"""Expected-value signal.

An EV signal flags a bet where the quoted decimal odds at a venue imply a
return greater than what the consensus fair probability justifies.

Core formula
------------
    edge = fair_prob * decimal_odds - 1

Interpretation:
  edge > 0  →  positive EV; betting this selection is profitable in expectation
  edge = 0  →  quoted at fair price; no edge, no penalty
  edge < 0  →  negative EV; the book is taking more margin than the fair prob
               warrants on this side

The formula is equivalent to:
    edge = fair_prob / implied_prob(decimal_odds) - 1

i.e., how much *better* the quoted price is than the fair price, expressed as
a fraction of stake. At a 10% edge on a $100 bet you expect +$10 in profit.

scan_ev() is the operational entry point: it maps the edge formula over every
(snapshot, selection) pair and returns only those meeting the threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fve.pricing.consensus import ConsensusResult
from fve.types import Market, MarketSnapshot, Venue


# --------------------------------------------------------------------------- #
# Signal type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EVSignal:
    """A positive-EV betting opportunity on one selection at one venue.

    All fields are at the snapshot timestamp; do not cache across refreshes.
    """

    market: Market
    selection_key: str      # matches Market.selections[i].key
    selection_name: str     # human-readable, from Market.selections[i].name
    venue: Venue
    fair_prob: float        # from ConsensusResult (devigged, sharp-anchored)
    decimal_odds: float     # quoted by venue at snapshot time
    edge: float             # fair_prob * decimal_odds - 1  (> 0 = +EV)

    @property
    def edge_pct(self) -> float:
        """Edge expressed as a percentage of stake (e.g. 0.08 → 8.0)."""
        return self.edge * 100.0

    @property
    def fair_decimal(self) -> float:
        """The fair decimal odds implied by the consensus probability."""
        return 1.0 / self.fair_prob


# --------------------------------------------------------------------------- #
# Pure math
# --------------------------------------------------------------------------- #
def edge(fair_prob: float, decimal_odds: float) -> float:
    """EV per unit staked.

    Parameters
    ----------
    fair_prob:
        Consensus fair probability for the selection, in (0, 1).
    decimal_odds:
        Decimal odds quoted by the venue (payout including stake, > 1).

    Returns
    -------
    float
        Positive → +EV; zero → fair price; negative → vig is eating the edge.
    """
    return fair_prob * decimal_odds - 1.0


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #
def scan_ev(
    result: ConsensusResult,
    snapshots: Sequence[MarketSnapshot],
    min_edge: float = 0.0,
) -> list[EVSignal]:
    """Find all bets with edge >= min_edge across the supplied snapshots.

    Parameters
    ----------
    result:
        Consensus fair probabilities for the market. Must cover the same
        selections as the snapshots.
    snapshots:
        One snapshot per venue (or more — duplicates are each evaluated).
        All must be for the same market.
    min_edge:
        Only return signals with edge >= this threshold. Default 0.0 (any
        positive EV). Set e.g. 0.03 to require at least 3% edge before
        surfacing a signal.

    Returns
    -------
    list[EVSignal]
        Sorted by edge descending (best opportunities first).
    """
    # Build a lookup from selection_key → selection object (for names)
    market = snapshots[0].market
    sel_by_key = {s.key: s for s in market.selections}

    signals: list[EVSignal] = []

    for snap in snapshots:
        for sel_key, price in snap.prices.items():
            fp = result.prob(sel_key)
            e = edge(fp, price.decimal_odds)
            if e >= min_edge:
                signals.append(
                    EVSignal(
                        market=market,
                        selection_key=sel_key,
                        selection_name=sel_by_key[sel_key].name,
                        venue=snap.venue,
                        fair_prob=fp,
                        decimal_odds=price.decimal_odds,
                        edge=e,
                    )
                )

    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals
