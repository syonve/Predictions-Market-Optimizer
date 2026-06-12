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

Liquidity scaling (EXCHANGE and PREDICTION_MARKET):

    If any selection in the snapshot has a non-None `Price.size`, the base
    weight is multiplied by a NORMALIZED, BOUNDED depth scale:

        scale = clamp(sqrt(avg_size / reference_size), min_scale, max_scale)

    Defaults (LiquidityConfig): reference_size=1000, min_scale=0.1,
    max_scale=2.0. A book at the reference depth keeps its base weight; a
    book 4x deeper hits the 2.0 cap — which deliberately promotes a deep
    prediction market (0.5 * 2.0 = 1.0) to sharp-anchor weight, per the
    design principle that deep PM order books count as sharp. A near-empty
    book fades toward the 0.1 floor. Snapshots without size data keep their
    bare base weight.

    History: the original scheme used raw sqrt(avg_size) with no bound. The
    first live Kalshi run showed why that fails: a ~10k-contract book scaled
    its weight to ~50, drowning every other voice 1000:1 in units that mean
    nothing across venues (contracts vs GBP vs USDC). reference_size is
    venue-unit-specific by construction — set it per venue class in config
    when a second order-book venue is added.

Confidence scaling (MODEL):

    For MODEL snapshots, `Price.size` carries the model's confidence score
    in [0, 1] (set by ``ModelEstimate.as_snapshot``), and the base weight is
    multiplied by it directly — linearly, NOT sqrt: confidence is already a
    calibrated quality score, and sqrt would compress it upward and
    over-trust thin models. A model backed by rich data keeps its full base
    weight; a model resting on two stale polls is discounted toward zero.
    MODEL snapshots without size data keep their bare base weight
    (hand-built snapshots remain supported).

    This implements "models are complements, not equal votes": with default
    weights, a fully-confident polling model contributes 0.3 and the
    fundamentals model is capped at 0.15 (its max confidence is 0.5), so
    even both models together (0.45) cannot outvote a Kalshi anchor (0.5).

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


@dataclass(frozen=True)
class LiquidityConfig:
    """Bounds for order-book depth scaling (EXCHANGE / PREDICTION_MARKET).

    reference_size:
        Top-of-book size (in the venue's own units) at which a book keeps
        exactly its base weight. Default 1000 — calibrated to Kalshi
        contracts; revisit per venue class when other order books are added.
    min_scale / max_scale:
        Clamp on the depth multiplier. The 2.0 default cap means the deepest
        book can at most double its base weight (0.5 → 1.0 for a prediction
        market — i.e., deep PM books earn sharp-anchor weight, never more).
    """

    reference_size: float = 1000.0
    min_scale: float = 0.1
    max_scale: float = 2.0

    def __post_init__(self) -> None:
        if self.reference_size <= 0:
            raise ValueError(f"reference_size must be > 0, got {self.reference_size}")
        if not 0.0 < self.min_scale <= self.max_scale:
            raise ValueError(
                f"need 0 < min_scale <= max_scale, got {self.min_scale}, {self.max_scale}"
            )


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
    liquidity: LiquidityConfig = LiquidityConfig(),
) -> float:
    """Effective weight for one snapshot.

    EXCHANGE / PREDICTION_MARKET snapshots with `Price.size` data scale by a
    normalized, clamped depth multiplier (see LiquidityConfig) — deeper
    books, louder voice, bounded. MODEL snapshots with `Price.size` data
    scale linearly by it: for models, size carries the confidence score in
    [0, 1] (clamped here defensively). Snapshots without size data use the
    bare base weight.
    """
    base = base_weights.get(snapshot.venue.kind, 0.0)
    if base == 0.0:
        return 0.0

    sizes = [p.size for p in snapshot.prices.values() if p.size is not None]

    if snapshot.venue.kind in (VenueKind.EXCHANGE, VenueKind.PREDICTION_MARKET):
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            scale = math.sqrt(max(0.0, avg_size) / liquidity.reference_size)
            scale = max(liquidity.min_scale, min(liquidity.max_scale, scale))
            return base * scale
    elif snapshot.venue.kind is VenueKind.MODEL:
        if sizes:
            confidence = max(0.0, min(1.0, sum(sizes) / len(sizes)))
            return base * confidence

    return base


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def consensus(
    snapshots: Sequence[MarketSnapshot],
    method: DevigMethod = "shin",
    weights: Mapping[VenueKind, float] | None = None,
    require_sharp: bool = True,
    liquidity: LiquidityConfig = LiquidityConfig(),
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
    require_sharp:
        If True (default), raises NoSharpAnchorError when no SHARP or EXCHANGE
        snapshot is present. Set to False for political/prediction markets where
        Kalshi or Polymarket IS the primary price source and no sharp books exist.
    liquidity:
        Bounds for order-book depth scaling. See LiquidityConfig.

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

    if require_sharp and n_sharp == 0:
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

        w = _snapshot_weight(snap, eff_weights, liquidity)
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
