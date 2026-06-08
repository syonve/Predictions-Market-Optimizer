"""Model calibration: Brier score and reliability diagram.

Calibration answers the question: "when my model says 70%, does the event
actually happen 70% of the time?" A miscalibrated model produces biased edge
estimates — overconfident models find fake edge, underconfident ones miss real
edge.

Brier Score
-----------
    BS = (1/N) × Σ (p_i − o_i)²

where p_i is the model probability and o_i ∈ {0, 1} is the outcome.
Range: 0 (perfect) to 1 (worst possible). Uninformative baseline (always
predict the base rate) achieves BS = p̄(1 − p̄) ≈ 0.25 for near-50/50 markets.

Reliability Diagram
-------------------
Bins predictions by probability, computes the actual hit rate in each bin,
and returns the data for plotting. A perfectly calibrated model lies on the
diagonal (predicted prob == actual freq). Consistent over-prediction plots
below the diagonal; under-prediction plots above.

Bins are equal-width by default (e.g. 10 bins of width 0.1). Empty bins
(no predictions) are excluded from the output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Brier score
# --------------------------------------------------------------------------- #
def brier_score(predictions: list[tuple[float, bool]]) -> float:
    """Mean squared error between model probabilities and binary outcomes.

    Parameters
    ----------
    predictions:
        Sequence of (probability, outcome) pairs. Probability must be in
        [0, 1]; outcome is True (event happened) or False.

    Returns
    -------
    float
        Brier score in [0, 1]. Lower is better. 0 = perfect, 0.25 =
        uninformative on near-even markets, 1 = maximally wrong.

    Raises
    ------
    ValueError
        If predictions is empty.
    """
    if not predictions:
        raise ValueError("brier_score requires at least one prediction")
    n = len(predictions)
    return sum((p - float(o)) ** 2 for p, o in predictions) / n


# --------------------------------------------------------------------------- #
# Reliability diagram
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReliabilityBin:
    """One bin in the reliability diagram.

    ``mean_prob`` is the average predicted probability of samples in this bin
    (not necessarily the bin midpoint — predictions are rarely uniform).
    ``actual_freq`` is the fraction of those samples where the event occurred.
    For a perfectly calibrated model: mean_prob ≈ actual_freq.
    """

    prob_low: float       # lower edge of bin (inclusive)
    prob_high: float      # upper edge of bin (exclusive, except last bin)
    mean_prob: float      # average predicted probability within the bin
    actual_freq: float    # fraction of positive outcomes in the bin
    n: int                # number of predictions in the bin


@dataclass(frozen=True)
class CalibrationResult:
    """Output of reliability_diagram(): Brier score + per-bin calibration data.

    ``bins`` is sorted by prob_low. Empty bins (no predictions) are excluded.

    Use ``bins`` to plot the reliability diagram: x = mean_prob, y = actual_freq,
    with size proportional to n. The diagonal (y=x) is perfect calibration.
    """

    brier_score: float
    n_samples: int
    bins: tuple[ReliabilityBin, ...]


def reliability_diagram(
    predictions: list[tuple[float, bool]],
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute calibration data for a reliability diagram.

    Parameters
    ----------
    predictions:
        Sequence of (probability, outcome) pairs.
    n_bins:
        Number of equal-width probability bins in [0, 1]. Default 10.

    Returns
    -------
    CalibrationResult
        Brier score plus per-bin statistics. Empty bins are excluded.

    Raises
    ------
    ValueError
        If predictions is empty or n_bins < 1.
    """
    if not predictions:
        raise ValueError("reliability_diagram requires at least one prediction")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    bs = brier_score(predictions)
    width = 1.0 / n_bins

    # Accumulate per-bin stats
    bin_probs: list[list[float]] = [[] for _ in range(n_bins)]
    bin_outcomes: list[list[bool]] = [[] for _ in range(n_bins)]

    for p, o in predictions:
        idx = min(int(p / width), n_bins - 1)  # clamp p=1.0 into last bin
        bin_probs[idx].append(p)
        bin_outcomes[idx].append(o)

    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        if not bin_probs[i]:
            continue
        n = len(bin_probs[i])
        bins.append(
            ReliabilityBin(
                prob_low=i * width,
                prob_high=(i + 1) * width,
                mean_prob=sum(bin_probs[i]) / n,
                actual_freq=sum(float(o) for o in bin_outcomes[i]) / n,
                n=n,
            )
        )

    return CalibrationResult(
        brier_score=bs,
        n_samples=len(predictions),
        bins=tuple(bins),
    )
