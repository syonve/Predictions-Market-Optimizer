"""Kelly criterion sizing.

Given a fair probability and a quoted price, Kelly tells you what fraction of
your bankroll to wager to maximise long-run geometric growth.

Core formula
------------
For a binary bet at decimal odds d with fair probability p:

    b    = d - 1          (net profit per unit staked on a win)
    edge = p * d - 1      (EV per unit staked, same as signals.ev.edge)
    f*   = edge / b       (full-Kelly fraction)

Equivalently, for a prediction-market contract priced at c = 1/d:

    f*   = (p - c) / (1 - c)

Both are identical; the decimal-odds form is used here since that's the
canonical representation across the rest of the codebase.

Practical adjustments
---------------------
Full Kelly is theoretically optimal but produces ruinous drawdowns when the
edge estimate is even slightly wrong — which it always is. Three adjustments
are applied via KellyConfig:

1. **kelly_fraction** (default 0.5, half-Kelly): multiplier on f*.
   Half-Kelly cuts variance by ~75% at the cost of ~25% of long-run growth.

2. **confidence** (default 1.0): second multiplier reflecting model certainty.
   Set to e.g. 0.6 early in a model's life, or for news-driven signals where
   uncertainty is higher than for a well-calibrated polling model.

3. **max_fraction** (default 0.05): hard cap on any single position regardless
   of what Kelly computes. Protects against fat-edge estimates and illiquid
   markets where Kelly produces unrealistically large fractions.

Applied fraction:
    f_applied = min(f* * kelly_fraction * confidence, max_fraction)
    stake      = bankroll * f_applied          (0 if edge <= 0)

Simultaneous bets
-----------------
The formula assumes bets are placed sequentially on independent outcomes and
the bankroll is updated between bets. With multiple concurrent positions the
"correct" answer requires simultaneous Kelly (a quadratic programme). In
practice, running standard Kelly with a conservative max_fraction is an
acceptable approximation for uncorrelated markets. Flag correlation explicitly
if building multi-position sizing — do not silently apply this formula to
highly correlated bets (e.g. two legs of the same election).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fve.signals.ev import EVSignal


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class KellyConfig:
    """Parameters governing how the raw Kelly fraction is adjusted.

    Attributes
    ----------
    kelly_fraction:
        Multiplier on f*. 0.5 = half-Kelly (default). Must be in (0, 1].
    max_fraction:
        Hard cap: no single bet exceeds this fraction of bankroll. Must be
        in (0, 1]. Default 0.05 (5%).
    confidence:
        Model-confidence multiplier in (0, 1]. Set below 1 when the edge
        estimate carries high uncertainty — e.g. early in a model's life,
        for news-driven signals, or when the polling sample is thin.
        Default 1.0 (full confidence in the edge estimate).
    """

    kelly_fraction: float = 0.5
    max_fraction: float = 0.05
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.kelly_fraction <= 1.0:
            raise ValueError(
                f"kelly_fraction must be in (0, 1], got {self.kelly_fraction}"
            )
        if not 0.0 < self.max_fraction <= 1.0:
            raise ValueError(
                f"max_fraction must be in (0, 1], got {self.max_fraction}"
            )
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in (0, 1], got {self.confidence}"
            )


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BetSize:
    """Recommended stake for one EVSignal given a bankroll and KellyConfig.

    Attributes
    ----------
    full_kelly:
        Raw f* before any multipliers or cap. Useful for diagnostics: a very
        large full_kelly with a low applied fraction means the cap is doing
        a lot of work — treat the signal with extra scepticism.
    fraction:
        Applied fraction of bankroll (after kelly_fraction, confidence, cap).
    stake:
        Dollar amount to wager: bankroll * fraction.
    expected_profit:
        stake * edge — the expected dollar gain per bet at these odds.
        Does not account for bankroll growth between bets.
    """

    full_kelly: float
    fraction: float
    stake: float
    expected_profit: float


# --------------------------------------------------------------------------- #
# Pure math
# --------------------------------------------------------------------------- #
def kelly_fraction(fair_prob: float, decimal_odds: float) -> float:
    """Raw full-Kelly fraction f* = edge / (decimal_odds - 1).

    Returns 0.0 for zero or negative edge (never bet -EV). No config applied;
    use kelly_stake() or size_ev_signal() for production sizing.

    Parameters
    ----------
    fair_prob:
        Consensus fair probability for the selection, in (0, 1).
    decimal_odds:
        Decimal odds quoted by the venue (> 1).
    """
    b = decimal_odds - 1.0
    edge = fair_prob * decimal_odds - 1.0
    if edge <= 0.0:
        return 0.0
    return edge / b


def kelly_stake(
    bankroll: float,
    fair_prob: float,
    decimal_odds: float,
    config: KellyConfig | None = None,
) -> float:
    """Dollar stake after applying KellyConfig adjustments.

    Parameters
    ----------
    bankroll:
        Current total bankroll in dollars (or any consistent currency unit).
    fair_prob:
        Consensus fair probability for the selection.
    decimal_odds:
        Decimal odds quoted by the venue.
    config:
        Sizing parameters. Defaults to KellyConfig() (half-Kelly, 5% cap).

    Returns
    -------
    float
        Dollar stake. 0.0 if edge is zero or negative.
    """
    cfg = config if config is not None else KellyConfig()
    fk = kelly_fraction(fair_prob, decimal_odds)
    if fk == 0.0:
        return 0.0
    applied = min(fk * cfg.kelly_fraction * cfg.confidence, cfg.max_fraction)
    return bankroll * applied


# --------------------------------------------------------------------------- #
# Signal-level entry point
# --------------------------------------------------------------------------- #
def size_ev_signal(
    signal: EVSignal,
    bankroll: float,
    config: KellyConfig | None = None,
) -> BetSize:
    """Produce a BetSize recommendation from an EVSignal and a bankroll.

    This is the primary entry point for the sizing layer. The EVSignal
    carries everything needed (fair_prob, decimal_odds, edge); this function
    applies the KellyConfig and returns the concrete stake.

    Parameters
    ----------
    signal:
        A positive-EV signal from signals.ev.scan_ev(). Signals with
        edge <= 0 return a BetSize with zero stake.
    bankroll:
        Current bankroll in dollars.
    config:
        Sizing parameters. Defaults to KellyConfig().
    """
    cfg = config if config is not None else KellyConfig()
    fk = kelly_fraction(signal.fair_prob, signal.decimal_odds)

    if fk == 0.0:
        return BetSize(full_kelly=0.0, fraction=0.0, stake=0.0, expected_profit=0.0)

    applied = min(fk * cfg.kelly_fraction * cfg.confidence, cfg.max_fraction)
    stake = bankroll * applied
    expected_profit = stake * signal.edge

    return BetSize(
        full_kelly=fk,
        fraction=applied,
        stake=stake,
        expected_profit=expected_profit,
    )
