"""Consensus fair-value engine.

Takes a list of `MarketSnapshot`s for the same market from different venues
and produces a single `ConsensusResult`: a weighted probability distribution
that is the fair value used by EV and arb signals.

Weighting model
---------------
Each snapshot gets an *effective weight*:

    w_eff = base_weight(venue_kind) * liquidity_scale(snapshot)

Base weights (configurable via the `weights` argument):

    SHARP            1.0   — Pinnacle et al., anchor consensus
    EXCHANGE         1.0   — Betfair; peer-to-peer, low vig
    PREDICTION_MARKET 0.5  — Kalshi, Polymarket; weighted by liquidity
    SOFT             0.1   — recreational books; echo sharps, small voice
    MODEL            0.3   — independent model estimate (polling aggregator,
                             power ratings, etc.); layered in where sharp
                             consensus is weak or absent

Liquidity scaling (EXCHANGE and PREDICTION_MARKET only):

    If any selection in the snapshot has a non-None `Price.size`, we compute
    the average size across all selections and multiply the base weight by
    sqrt(avg_size). sqrt dampens outliers while still rewarding deep books.
    Snapshots without size data are left at their base weight.

    Design note: absolute size values are not comparable across venues
    (Betfair uses GBP, Kalshi uses contracts, Polymarket uses USDC). Weights
    are normalized before the weighted average, so only *relative* sizes
    within the same consensus call matter.

No-sharp policy
---------------
If no SHARP or EXCHANGE snapshot is present, we raise `NoSharpAnchorError`
rather than silently returning a soft-books consensus. The caller can inspect
`result.n_sharp` to distinguish sharp-anchored from degraded results.

For political prediction markets (Kalshi/Polymarket elections, Fed rate
decisions), there is typically no Pinnacle or Betfair equivalent. In those
cases, callers should pass the Kalshi snapshot as a PREDICTION_MARKET anchor
and the polling model snapshot as a MODEL supplement. The consensus will still
raise ``NoSharpAnchorError`` — callers must catch it and decide whether to
proceed with PM-only or PM+model consensus. A future improvement would make
the sharp requirement configurable per market class.

The devig method is applied uniformly across all snapshots. If per-venue
method selection is needed, call `devig()` manually per snapshot and assemble
the weighted average yourself.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from fve.pricing.devig import DevigMethod, booksum, devig, implied_probs
from fve.types import MarketSnapshot, VenueKind

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
_DEFAULT_WEIGHTS: dict[VenueKind, float] = {
    VenueKind.SHARP: 1.0,
    VenueKind.EXCHANGE: 1.0,
    VenueKind.PREDICTION_MARKET: 0.5,
    VenueKind.SOFT: 0.1,
    VenueKind.MODEL: 0.3,
    # MODEL weight: below PREDICTION_MARKET because it lacks the continuous
    # price-discovery of a live order book. When n_sharp > 0, this weight
    # means a polling model contributes ~23% as much as a Pinnacle snapshot
    # (0.3 vs 1.0 after normalization with typical snapshot counts). Increase
    # if the model has strong calibration history on this market class.
}


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConsensusResult:
    """Weighted fair-probability distribution for a market.

    `fair_probs` and `selection_keys` are parallel tuples in the market's
    canonical selection order (the order the `Market.selections` tuple defines).

    `venue_probs` holds each venue's independently devigged distribution for
    diagnostics, deviation detection, and calibration analysis.
    """

    selection_keys: tuple[str, ...]
    fair_probs: tuple[float, ...]
    n_sharp: int          # SHARP + EXCHANGE snapshots consumed
    n_soft: int           # SOFT snapshots consumed
    n_pm: int             # PREDICTION_MARKET snapshots consumed
    n_model: int          # MODEL snapshots consumed
    venue_probs: dict[str, tuple[float, ...]]  # venue_key → devigged probs

    def prob(self, selection_key: str) -> float:
        """Fair probability for a selection by key."""
        idx = self.selection_keys.index(selection_key)
        return self.fair_probs[idx]

    def fair_decimal(self, selection_key: str) -> float:
        """Fair decimal odds (1 / fair_prob) for a selection."""
        return 1.0 / self.prob(selection_key)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class NoSharpAnchorError(ValueError):
    """Raised when no SHARP or EXCHANGE snapshot is available.

    Fair value from soft books alone is unreliable; the caller should either
    supply a sharp snapshot or explicitly decide to proceed without one.
    """


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _snapshot_weight(
    snapshot: MarketSnapshot,
    base_weights: Mapping[VenueKind, float],
) -> float:
    """Effective weight for one snapshot.

    For EXCHANGE and PREDICTION_MARKET snapshots that carry `Price.size` data,
    the base weight is scaled by sqrt(average size across all selections).
    All other snapshots use the bare base weight.
    """
    base = base_weights.get(snapshot.venue.kind, 0.0)
    if base == 0.0:
        return 0.0

    if snapshot.venue.kind in (VenueKind.EXCHANGE, VenueKind.PREDICTION_MARKET):
        sizes = [p.size for p in snapshot.prices.values() if p.size is not None]
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            return base * math.sqrt(avg_size)

    return base


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def consensus(
    snapshots: Sequence[MarketSnapshot],
    method: DevigMethod = "shin",
    weights: Mapping[VenueKind, float] | None = None,
) -> ConsensusResult:
    """Compute a weighted fair-probability distribution for a market.

    Parameters
    ----------
    snapshots:
        One or more `MarketSnapshot`s for the *same* market. The function does
        not validate that all snapshots share a market; pass consistent data.
    method:
        Devig method applied to every snapshot. Defaults to Shin's, which
        corrects the favorite-longshot bias in the right empirical direction.
        Override per market class once calibration data is available.
    weights:
        Mapping from `VenueKind` to base weight. Defaults to `_DEFAULT_WEIGHTS`.
        Supply a custom mapping to override any or all venue-kind weights.

    Returns
    -------
    ConsensusResult

    Raises
    ------
    ValueError
        If `snapshots` is empty or all effective weights are zero.
    NoSharpAnchorError
        If no SHARP or EXCHANGE snapshot is present.
    """
    if not snapshots:
        raise ValueError("consensus requires at least one snapshot")

    eff_weights = weights if weights is not None else _DEFAULT_WEIGHTS

    # Tally venue kinds
    n_sharp = sum(1 for s in snapshots if s.venue.is_sharp)
    n_soft = sum(1 for s in snapshots if s.venue.kind == VenueKind.SOFT)
    n_pm = sum(1 for s in snapshots if s.venue.kind == VenueKind.PREDICTION_MARKET)
    n_model = sum(1 for s in snapshots if s.venue.kind == VenueKind.MODEL)

    if n_sharp == 0:
        raise NoSharpAnchorError(
            f"no SHARP or EXCHANGE snapshots available "
            f"(soft={n_soft}, pm={n_pm}). "
            "Provide at least one sharp/exchange snapshot for a trustworthy fair value, "
            "or explicitly check n_sharp on the result if you intend to proceed without one."
        )

    # Build the weighted sum over devigged per-venue distributions
    market = snapshots[0].market
    selection_keys = tuple(s.key for s in market.selections)
    n_sel = len(selection_keys)

    weighted_sum = [0.0] * n_sel
    total_weight = 0.0
    venue_probs: dict[str, tuple[float, ...]] = {}

    for snap in snapshots:
        odds = snap.decimal_odds()  # tuple, in selection order

        # MODEL snapshots carry pre-devigged fair probabilities (booksum ≈ 1.0).
        # Shin's and power methods require booksum > 1 (positive vig); applying
        # them to a fair-prob snapshot is a no-op at best and an error at worst.
        # When booksum is within 1e-6 of 1.0, use implied probs directly.
        bs = booksum(odds)
        if abs(bs - 1.0) < 1e-6:
            fair_p = implied_probs(odds)
        else:
            fair_p = devig(odds, method=method)
        venue_probs[snap.venue.key] = fair_p

        w = _snapshot_weight(snap, eff_weights)
        for i, p in enumerate(fair_p):
            weighted_sum[i] += w * p
        total_weight += w

    if total_weight == 0.0:
        raise ValueError(
            "all effective snapshot weights are zero — check the weight mapping "
            "and ensure at least one snapshot has a non-zero weight."
        )

    fair_probs = tuple(ws / total_weight for ws in weighted_sum)

    return ConsensusResult(
        selection_keys=selection_keys,
        fair_probs=fair_probs,
        n_sharp=n_sharp,
        n_soft=n_soft,
        n_pm=n_pm,
        n_model=n_model,
        venue_probs=venue_probs,
    )
