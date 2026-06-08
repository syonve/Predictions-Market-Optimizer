# fve — sports prediction-market fair-value engine

One fair-value engine drives both arbitrage detection and positive-EV signal:
arbitrage is just the extreme case of a large deviation from fair value.
**Analysis and signal only — this code never places or executes bets.**

## Layout

```
src/fve/
  types.py            Normalized, provider-agnostic domain types (the boundary)
  providers/
    base.py           OddsProvider interface (Protocol) — every source satisfies it
  pricing/
    devig.py          Pluggable devig: proportional, Shin, power  [done]
    consensus.py      Sharp-weighted consensus fair value          [next]
  signals/            ev.py, arb.py                                 [later]
  sizing/             kelly.py                                      [later]
  tracking/           clv.py, calibration.py (Brier, reliability)  [later]
tests/
  test_devig.py       Known-answer tests, one independent reference per method
```

## The two load-bearing abstractions (built first, on purpose)

**`OddsProvider`** (`providers/base.py`) is the single contract every source
satisfies — aggregator (sharp + closing odds), The Odds API (soft books), a
direct Kalshi/Polymarket client. Pricing depends only on this interface and on
`fve.types`; no provider schema leaks past the boundary. (Sync for v0; the
async question for streaming order books is flagged in the file, deferred until
the first live provider.)

**`devig`** (`pricing/devig.py`) converts raw implied probabilities into a fair
distribution. The method is pluggable and selected by config / the calibration
harness — never silently hardcoded, because the method changes which edges
surface:

- `proportional` — margin removed in proportion to implied prob. Baseline; does
  not correct favorite-longshot bias.
- `shin` *(default)* — attributes margin to an informed-money fraction `z`.
  Loads more margin onto favorites, correcting favorite-longshot bias in the
  empirically right direction. Good for 2-way sharp-anchored markets.
- `power` — fits exponent `k` with `sum(qᵢ**k) = 1`. Most flexible, often
  best-calibrated, especially many-outcome.

Which method is "best" is an empirical question, decided by Brier score /
reliability on the markets actually traded (the `tracking/calibration` layer),
not by fiat. The default is a starting point, not a verdict.

```python
from fve.pricing import devig
devig([1.91, 1.91])                      # -> (0.5, 0.5)
devig([1.4, 3.1], method="shin")         # favorite nudged up vs. proportional
```

## Tests

Each method is checked against an **independent** reference, not its own code:
proportional against the hand closed form; Shin and power against forward
generative models (build odds from known `(p, z)` / `(p, k)`, require the
solver to recover them). Plus sum-to-one, symmetry, and the favorite-longshot
direction that separates Shin from proportional.

```bash
uv run --with pytest pytest      # or: PYTHONPATH=src pytest -q
```

## Status

Done: project skeleton, normalized types, `OddsProvider` interface, devig core
(3 methods) with known-answer tests (all green).

Next: `pricing/consensus.py` — sharp-weighted consensus across devigged venue
snapshots. Open decisions to settle when we get there: the sharp-vs-soft
weighting scheme and how order-book depth scales a prediction market's weight.
