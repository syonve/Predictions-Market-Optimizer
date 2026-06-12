"""ModelProvider protocol and ModelEstimate output type.

A ModelProvider is an independent probability estimator — polling aggregator,
power-rating model, base-rate model, etc. — that can stand in as a synthetic
"venue" when sharp consensus is weak or absent.

Relationship to OddsProvider
-----------------------------
``OddsProvider`` wraps a market-price source (Kalshi, Polymarket, Pinnacle).
``ModelProvider`` wraps a fundamentals-based estimator. Both produce
``MarketSnapshot``-compatible output so the same consensus engine consumes
them, but they are kept as separate protocols because their update frequencies,
error modes, and confidence signals differ fundamentally.

Flagged decisions
-----------------
1. **as_snapshot() uses VenueKind.MODEL with consensus weight 0.3.**
   The weight is defined in ``pricing.consensus._DEFAULT_WEIGHTS``. It is
   positioned between PREDICTION_MARKET (0.5) and SOFT (0.1): a polling
   model is informative on thin markets but should not override deep
   order-book consensus. Adjust the weight in config if you trust the model
   more (e.g., on very illiquid obscure elections).

2. **ModelEstimate carries a confidence score [0, 1] — and it IS plumbed
   into consensus weights.** ``as_snapshot()`` stores the confidence in each
   ``Price.size`` field; ``pricing.consensus._snapshot_weight`` multiplies
   the MODEL base weight by it. A fully-confident model gets its full base
   weight (0.3); a thin model is discounted proportionally. Convention: for
   MODEL venues, ``Price.size`` means confidence in [0, 1], not liquidity.

3. **ModelProvider is synchronous.** Matching OddsProvider v0. Async can
   be added later as an AsyncModelProvider Protocol if polling APIs or
   inference calls need it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from fve.types import Market, MarketSnapshot, Price, Venue, VenueKind


# Synthetic venue used for all model-derived snapshots.
MODEL_VENUE = Venue(
    key="polling_model",
    name="Polling Model",
    kind=VenueKind.MODEL,
)


@dataclass(frozen=True)
class ModelEstimate:
    """Probability estimate produced by a ModelProvider for one market.

    Parameters
    ----------
    market:
        The market this estimate covers.
    selection_probs:
        Mapping ``{selection_key: probability}`` for every selection in the
        market. Probabilities are normalized to sum to 1.0.
    confidence:
        Data quality score in [0, 1]. 1.0 means the model is well-supported
        (many polls / large samples / recent data); 0.0 means the estimate
        rests on essentially no data and should be treated with extreme
        caution. Computed by the provider; definition varies by model type.
    n_samples:
        Number of data points (polls, observations) that produced this
        estimate. Informational.
    model_name:
        Short identifier for the producing model, e.g. ``"polling_aggregate"``.
    computed_at:
        UTC timestamp of computation.
    venue:
        The synthetic Venue to use when producing a MarketSnapshot. Defaults
        to MODEL_VENUE.
    """

    market: Market
    selection_probs: dict[str, float]
    confidence: float
    n_samples: int
    model_name: str
    computed_at: datetime
    venue: Venue = MODEL_VENUE  # type: ignore[assignment]

    def prob(self, selection_key: str) -> float:
        """Probability for a selection; raises KeyError if not present."""
        return self.selection_probs[selection_key]

    def as_snapshot(self) -> MarketSnapshot:
        """Convert this estimate to a MarketSnapshot for use in consensus.

        Each selection's probability is converted to decimal odds (1/p).
        The model's confidence is stored in each ``Price.size`` so the
        consensus weight scales with data quality (see flagged decision 2).
        The snapshot is *not* marked as closing.
        """
        prices: dict[str, Price] = {}
        for sel in self.market.selections:
            p = self.selection_probs.get(sel.key)
            if p is None or p <= 0.0:
                raise ValueError(
                    f"ModelEstimate missing valid probability for selection "
                    f"{sel.key!r} in market {self.market.key!r}"
                )
            # Clamp to avoid decimal_odds <= 1.0 at floating-point boundaries
            p_safe = max(1e-6, min(1.0 - 1e-6, p))
            prices[sel.key] = Price(decimal_odds=1.0 / p_safe, size=self.confidence)

        return MarketSnapshot(
            market=self.market,
            venue=self.venue,
            timestamp=self.computed_at,
            prices=prices,
            is_closing=False,
        )


class ModelProvider(Protocol):
    """Interface for independent probability estimators.

    Implementations must be stateless with respect to market data — all
    mutable state (poll lists, config) is injected at construction time.
    """

    @property
    def model_name(self) -> str:
        """Short stable identifier for this model."""
        ...

    def estimate(self, market: Market) -> ModelEstimate | None:
        """Produce a probability estimate for ``market``.

        Returns ``None`` if the model has no data for the given market (e.g.
        no polls have been ingested for that ticker). Never raises on missing
        data — the caller decides how to handle the absence of a model signal.
        """
        ...
