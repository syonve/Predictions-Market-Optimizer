"""Closing Line Value (CLV) tracking.

CLV measures whether you obtained better odds at entry than the market settled
on at close. It is the primary success metric for this system because:

  1. Outcome variance dominates P&L over short samples (a coin-flip of 1,000
     bets produces ~32 std dev swings in profit). CLV is far less noisy.
  2. Consistently beating the closing line implies your model is ahead of the
     market when you act — which is the definition of edge.
  3. It separates signal from luck: a losing run with positive CLV is noise;
     a winning run with negative CLV is luck running out.

Three CLV metrics are tracked, all from different angles on the same question:

  clv_edge       = closing_prob × entry_odds − 1
      The edge you had at entry, measured using the closing probability as
      ground truth. Same formula as ev.edge — the closing line is treated as
      the best available estimate of fair value after all information is in.
      Positive = you got better than fair value at entry.

  clv_log        = log(entry_odds / closing_odds)
      Log ratio of odds. Preferred for aggregation: it is additive, symmetric
      around zero, and unit-free across different odds ranges. Use this for
      the t-statistic and portfolio-level CLV.

  clv_odds_ratio = entry_odds / closing_odds − 1
      Simple percentage: how much better (worse) your entry price was than the
      close. Intuitive but not ideal for aggregation (asymmetric around zero).

All three are exposed as properties on CLVRecord to avoid recomputation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from fve.types import Market, Venue


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #
def clv_edge(entry_odds: float, closing_prob: float) -> float:
    """Edge at entry using the devigged closing probability as ground truth.

    Identical to ``signals.ev.edge(closing_prob, entry_odds)``.
    Positive = entry was better than closing fair value.
    """
    return closing_prob * entry_odds - 1.0


def clv_log(entry_odds: float, closing_odds: float) -> float:
    """Log-ratio CLV: log(entry_odds / closing_odds).

    Positive = you got longer odds (better price) than the close.
    Additive across bets — the preferred metric for portfolio aggregation.
    """
    return math.log(entry_odds / closing_odds)


def clv_odds_ratio(entry_odds: float, closing_odds: float) -> float:
    """Simple odds-ratio CLV: entry_odds / closing_odds − 1.

    Positive = entry was cheaper (higher odds) than the close.
    Intuitive for single-bet inspection; use clv_log for aggregation.
    """
    return entry_odds / closing_odds - 1.0


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BetRecord:
    """A placed bet, recorded at the moment of entry.

    This is the immutable record of what was wagered. It is enriched into a
    CLVRecord once closing odds are available.
    """

    id: str                      # unique identifier (e.g. UUID)
    market: Market
    selection_key: str
    selection_name: str
    venue: Venue
    entry_odds: float            # decimal odds at time of bet
    fair_prob_at_entry: float    # consensus fair probability when bet was placed
    stake: float                 # dollar amount wagered
    placed_at: datetime

    @property
    def entry_edge(self) -> float:
        """Edge at time of placement: fair_prob_at_entry × entry_odds − 1."""
        return self.fair_prob_at_entry * self.entry_odds - 1.0


@dataclass(frozen=True)
class CLVRecord:
    """A BetRecord enriched with closing-line data and (optionally) outcome.

    All three CLV metrics are available as properties computed on demand.
    ``outcome`` and ``pnl`` are None for pending (unresolved) bets.
    """

    bet: BetRecord
    closing_odds: float          # decimal odds at market close (devigged source)
    closing_prob: float          # devigged closing probability (the CLV ground truth)
    settled_at: datetime | None  # when the market resolved; None if pending
    outcome: bool | None         # True=won, False=lost, None=pending
    pnl: float | None            # realized profit/loss; None if pending

    # --- CLV metrics (properties to avoid redundant storage) ---

    @property
    def clv_edge(self) -> float:
        """closing_prob × entry_odds − 1."""
        return clv_edge(self.bet.entry_odds, self.closing_prob)

    @property
    def clv_log(self) -> float:
        """log(entry_odds / closing_odds)."""
        return clv_log(self.bet.entry_odds, self.closing_odds)

    @property
    def clv_odds_ratio(self) -> float:
        """entry_odds / closing_odds − 1."""
        return clv_odds_ratio(self.bet.entry_odds, self.closing_odds)

    @property
    def model_drift(self) -> float:
        """closing_prob − fair_prob_at_entry.

        Positive = market moved in your favour after entry (you were early).
        Negative = market moved against you (you were early in the wrong
        direction, or the model underestimated the closing probability).
        """
        return self.closing_prob - self.bet.fair_prob_at_entry


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CLVSummary:
    """Portfolio-level CLV statistics across a list of CLVRecords.

    ``t_stat`` is the primary signal-vs-noise metric: it tests whether
    mean_clv_log is distinguishable from zero given the observed variance.
    Rule of thumb: |t_stat| > 2 suggests the CLV signal is real; < 1.5
    means the sample is too small to be confident.

    ``total_pnl`` is computed only over settled bets (outcome is not None).
    If all bets are pending, total_pnl is None.
    """

    n_bets: int
    mean_clv_log: float          # primary aggregation metric
    std_clv_log: float           # sample standard deviation of clv_log
    t_stat: float | None         # mean / (std / sqrt(n)); None if n == 1
    mean_clv_edge: float
    n_positive: int              # bets where clv_log > 0 (beat the close)
    hit_rate: float              # n_positive / n_bets
    total_pnl: float | None      # sum of pnl for settled bets; None if none settled


def aggregate_clv(records: list[CLVRecord]) -> CLVSummary:
    """Compute portfolio CLV statistics over a list of CLVRecords.

    Parameters
    ----------
    records:
        One or more CLVRecords. All must have closing odds populated.

    Raises
    ------
    ValueError
        If records is empty.
    """
    if not records:
        raise ValueError("aggregate_clv requires at least one CLVRecord")

    n = len(records)
    logs = [r.clv_log for r in records]
    edges = [r.clv_edge for r in records]

    mean_log = sum(logs) / n
    mean_edge = sum(edges) / n
    n_positive = sum(1 for x in logs if x > 0.0)

    # Sample standard deviation (ddof=1); undefined for n=1
    if n > 1:
        variance = sum((x - mean_log) ** 2 for x in logs) / (n - 1)
        std_log = math.sqrt(variance)
        t_stat: float | None = mean_log / (std_log / math.sqrt(n)) if std_log > 0 else None
    else:
        std_log = 0.0
        t_stat = None

    # P&L: only settled bets
    settled_pnls = [r.pnl for r in records if r.pnl is not None]
    total_pnl: float | None = sum(settled_pnls) if settled_pnls else None

    return CLVSummary(
        n_bets=n,
        mean_clv_log=mean_log,
        std_clv_log=std_log,
        t_stat=t_stat,
        mean_clv_edge=mean_edge,
        n_positive=n_positive,
        hit_rate=n_positive / n,
        total_pnl=total_pnl,
    )
