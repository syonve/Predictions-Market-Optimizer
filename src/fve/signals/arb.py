"""Arbitrage signal.

A cross-venue arbitrage exists when the best available decimal odds for each
outcome — taken from *different* venues — collectively imply a book sum below
1.0. This means you can bet every outcome simultaneously and lock in a
guaranteed profit regardless of result.

Math
----
Let d_1, ..., d_n be the best decimal odds available for each of n mutually-
exclusive, exhaustive outcomes. Define the arb sum:

    S = sum(1 / d_i)   (sum of implied probabilities at best odds)

No arb:   S >= 1.0  (the market has a positive margin even at best prices)
Arb:      S <  1.0  (you can cover all outcomes and keep the difference)

When S < 1, for every $1 of guaranteed return allocate:
    stake_i = 1 / d_i   →   return_i = stake_i * d_i = 1   for all i
    total stake          = S
    guaranteed profit    = 1 - S
    ROI (per $ staked)   = (1 - S) / S

Stake fractions (stakes as fractions of total bankroll):
    f_i = stake_i / total = (1 / d_i) / S = 1 / (d_i * S)

The relationship to EV signals: a large positive EV on one side is the soft
version of arb. True arb (S < 1) is the extreme where every side is +EV
simultaneously.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fve.types import Market, MarketSnapshot, Venue


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArbLeg:
    """One side of an arbitrage: the specific venue and odds to bet.

    `stake_fraction` is the share of total bankroll to place on this leg so
    that every leg returns the same amount regardless of outcome.
    """

    selection_key: str
    selection_name: str
    venue: Venue
    decimal_odds: float
    stake_fraction: float   # fraction of total stake; all legs sum to 1.0


@dataclass(frozen=True)
class ArbSignal:
    """A confirmed cross-venue arbitrage opportunity.

    `roi` is the guaranteed return per unit of total stake. On a $1,000
    bankroll allocation, the locked-in profit is `roi * 1_000` dollars before
    friction (commission, withdrawal limits, etc.).

    Always check execution feasibility: line moves between discovery and bet
    placement eat the margin quickly at small ROIs.
    """

    market: Market
    legs: tuple[ArbLeg, ...]
    arb_sum: float      # sum(1/best_odds); < 1.0 confirms the arb
    roi: float          # (1 - arb_sum) / arb_sum; guaranteed profit per $ staked


# --------------------------------------------------------------------------- #
# Pure math
# --------------------------------------------------------------------------- #
def arb_roi(inv_odds_sum: float) -> float:
    """Guaranteed ROI given arb_sum = sum(1 / best_decimal_odds_i).

    Returns 0.0 when inv_odds_sum == 1 (break-even) and is positive for
    inv_odds_sum < 1. Callers should check inv_odds_sum < 1 before calling.
    """
    return (1.0 - inv_odds_sum) / inv_odds_sum


def stake_fractions(best_odds: Sequence[float]) -> tuple[float, ...]:
    """Stake fractions that guarantee equal return across all outcomes.

    Each element is the fraction of total bankroll to wager on the
    corresponding outcome. Fractions sum to 1.0.

    Derivation: to guarantee return R from total stake T, set
        s_i * d_i = R  →  s_i = R / d_i
        T = R * sum(1/d_i) = R * S
        f_i = s_i / T = 1 / (d_i * S)
    """
    s = sum(1.0 / d for d in best_odds)
    return tuple(1.0 / (d * s) for d in best_odds)


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #
def scan_arb(snapshots: Sequence[MarketSnapshot]) -> ArbSignal | None:
    """Detect a cross-venue arbitrage across the supplied snapshots.

    For each outcome, finds the venue offering the highest decimal odds, then
    checks whether those best odds collectively produce arb_sum < 1.

    Parameters
    ----------
    snapshots:
        All available snapshots for a single market. May span any number of
        venues and venue kinds — arb detection is venue-agnostic.

    Returns
    -------
    ArbSignal
        If arb_sum < 1: the confirmed arbitrage with legs, arb_sum, and ROI.
    None
        If no arbitrage exists at current prices.
    """
    if not snapshots:
        return None

    market = snapshots[0].market
    sel_by_key = {s.key: s for s in market.selections}

    # --- find best (highest) decimal odds per outcome across all venues ---
    best: dict[str, tuple[float, Venue]] = {}  # sel_key → (odds, venue)
    for snap in snapshots:
        for sel_key, price in snap.prices.items():
            prev_odds, _ = best.get(sel_key, (0.0, snap.venue))
            if price.decimal_odds > prev_odds:
                best[sel_key] = (price.decimal_odds, snap.venue)

    # --- arb check ---
    ordered_keys = [s.key for s in market.selections]
    best_odds_list = [best[k][0] for k in ordered_keys]
    arb_sum = sum(1.0 / d for d in best_odds_list)

    if arb_sum >= 1.0:
        return None

    # --- build ArbSignal ---
    roi = arb_roi(arb_sum)
    fracs = stake_fractions(best_odds_list)

    legs = tuple(
        ArbLeg(
            selection_key=k,
            selection_name=sel_by_key[k].name,
            venue=best[k][1],
            decimal_odds=best[k][0],
            stake_fraction=fracs[i],
        )
        for i, k in enumerate(ordered_keys)
    )

    return ArbSignal(market=market, legs=legs, arb_sum=arb_sum, roi=roi)
