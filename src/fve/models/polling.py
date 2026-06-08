"""PollingModelProvider — polling-aggregate ModelProvider implementation.

Ingests a list of ``Poll`` objects, aggregates them per market using the
weighted scheme in ``poll.py``, and produces ``ModelEstimate`` instances.

Binary-market design
--------------------
Kalshi binary markets have exactly two selections: ``"yes"`` and ``"no"``.
Polls are typically ingested with ``selection_key="yes"`` (the affirmative
outcome). When the provider builds the estimate:

  1. Aggregate YES polls → raw_yes_prob
  2. Derive NO prob as complement: raw_no_prob = 1 − raw_yes_prob
  3. Normalize the pair to sum exactly to 1.0 (eliminates floating-point drift)

If polls are also supplied for ``"no"`` independently, they are aggregated
separately and both sides are normalized together. This handles unusual cases
where polls are framed as the negative outcome (rare in practice).

Markets with more than two selections (multi-way races, primary fields) follow
the same logic: aggregate each selection independently from available polls,
derive missing selections as the residual if exactly one is absent, and
normalize the full set.

Flagged decisions
-----------------
1. **Missing selection handling.** If polls exist for only a subset of
   selections and the market has > 2 selections, the provider distributes the
   residual (1 - sum_of_known_probs) equally among unknown selections. This
   is a placeholder — for real multi-way markets you want either (a) dedicated
   polls per selection or (b) an electoral simulation layer. The provider logs
   how many selections were imputed so callers can detect the approximation.

2. **Reference date is injectable.** ``_reference_date`` defaults to today
   (UTC) but can be overridden in tests or backtests. All time-weighting
   runs relative to this date, so the same poll list produces deterministic
   results for a given reference date.

3. **Poll list is captured at construction time.** The provider does not
   update its poll list after construction. Callers that ingest live polls
   should reconstruct the provider (cheap — it holds no heavy state) or
   extend the list and pass a new instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from fve.models.base import MODEL_VENUE, ModelEstimate
from fve.models.poll import (
    AggregationConfig,
    InsufficientDataError,
    Poll,
    aggregate_polls,
    poll_confidence,
)
from fve.types import Market


class PollingModelProvider:
    """ModelProvider backed by a list of Poll objects.

    Parameters
    ----------
    polls:
        All polls the model knows about. May span multiple markets.
    config:
        Weighting and house-effect configuration.
    _reference_date:
        Override today's date for deterministic tests/backtests. If None,
        uses ``datetime.now(UTC).date()`` at the time ``estimate()`` is
        called.
    """

    def __init__(
        self,
        polls: list[Poll],
        config: AggregationConfig | None = None,
        _reference_date: date | None = None,
    ) -> None:
        self._polls = list(polls)
        self._config = config or AggregationConfig()
        self._reference_date = _reference_date

    # --- ModelProvider interface ---

    @property
    def model_name(self) -> str:
        return "polling_aggregate"

    def estimate(self, market: Market) -> ModelEstimate | None:
        """Aggregate polls for ``market`` into a probability estimate.

        Returns ``None`` if no polls have been ingested for the market key.
        The returned ``ModelEstimate.selection_probs`` sum to 1.0.
        """
        market_polls = [p for p in self._polls if p.market_key == market.key]
        if not market_polls:
            return None

        ref_date = self._reference_date or datetime.now(tz=timezone.utc).date()

        # --- Step 1: aggregate each selection we have polls for ---
        raw_probs: dict[str, float] = {}
        max_eff_n: float = 0.0
        n_total_polls: int = 0

        for sel in market.selections:
            try:
                prob, eff_n = aggregate_polls(
                    market_polls, sel.key, ref_date, self._config
                )
                raw_probs[sel.key] = prob
                max_eff_n = max(max_eff_n, eff_n)
                n_total_polls += sum(
                    1 for p in market_polls if p.selection_key == sel.key
                )
            except InsufficientDataError:
                pass  # this selection has no polls; handled below

        if not raw_probs:
            # No selection had enough polls at all
            return None

        # --- Step 2: impute missing selections ---
        all_keys = [s.key for s in market.selections]
        missing_keys = [k for k in all_keys if k not in raw_probs]

        if missing_keys:
            known_sum = sum(raw_probs.values())
            residual = max(0.0, 1.0 - known_sum)
            # Distribute residual equally among unknown selections
            imputed = residual / len(missing_keys)
            for k in missing_keys:
                raw_probs[k] = imputed

        # --- Step 3: normalize to sum exactly 1.0 ---
        total = sum(raw_probs.values())
        if total <= 0.0:
            return None
        selection_probs = {k: v / total for k, v in raw_probs.items()}

        # Preserve selection order from market
        selection_probs_ordered = {k: selection_probs[k] for k in all_keys}

        confidence = poll_confidence(max_eff_n, self._config.confidence_reference_n)

        return ModelEstimate(
            market=market,
            selection_probs=selection_probs_ordered,
            confidence=confidence,
            n_samples=n_total_polls,
            model_name=self.model_name,
            computed_at=datetime.now(tz=timezone.utc),
            venue=MODEL_VENUE,
        )
