# CLAUDE.md — Sports Prediction-Market Fair-Value Engine

## Project
A Python tool that ingests betting odds and prediction-market prices across
multiple venues, converts them to clean implied probabilities, computes a
consensus "fair value" per market, and flags (1) cross-venue arbitrage /
pricing inconsistencies and (2) positive-EV bets where an independent model
deviates from the market. ONE fair-value engine drives both — arbitrage is the
extreme case of a large deviation. Analysis and signal only: never auto-place bets.

## About the user
Portfolio manager, fluent in Python and quant concepts. Skip basics. Prioritize
mathematical correctness and clean, testable structure over hand-holding.

## Core design principles
- For liquid markets, fair value = devigged consensus, weighted toward SHARP
  venues (Pinnacle, Betfair exchange, and prediction-market order books where deep).
  Independent models (Elo, power ratings, props regressions) layer in ONLY where
  sharp consensus is weak or absent: obscure leagues, player props, alt lines, live.
- Devigging is PLUGGABLE. Implement proportional, Shin's method, and the power
  method, swappable via config. The method changes which edges surface — never
  silently hardcode one.
- CLV (closing line value) is the primary success metric, tracked from the first
  signal. Outcome variance swamps short-run P&L; an aggregator that supplies
  closing odds is preferred for this reason.
- Calibration before trust: backtesting must produce Brier scores and reliability
  diagrams on the fair-value model itself before any deviation is treated as edge.

## Data sources (fragmented — abstract behind interfaces)
- Sharp anchor is NOT freely/directly available: Pinnacle has no public API;
  Betfair Exchange API is free but auth-heavy. Plan to use an aggregator
  (e.g. SportsGameOdds / OddsPapi) for sharp + closing odds.
- Soft books: The Odds API (free tier, simple REST, ~40 soft books, no sharps).
- Prediction markets: Kalshi (REST + WebSocket, CFTC-regulated, easy) and
  Polymarket (EIP-712 signed orders on Polygon, wallet + gas — higher friction).
- Every source sits behind a clean `OddsProvider` interface. An aggregator client
  and a direct Kalshi client must both satisfy the same interface. Do NOT couple
  pricing logic to any provider's schema — normalize at the boundary.

## Conventions
- Python 3.11+, fully type-hinted. uv for dependency management.
- All math as PURE functions (devig, EV, Kelly); keep IO at the edges.
- pytest for every math module, with KNOWN-ANSWER tests for each devig method.
- Layout: providers/, pricing/ (devig, consensus), signals/ (ev, arb),
  sizing/ (kelly), tracking/ (clv, calibration).

## How to work
- Show the project structure and the `OddsProvider` interface BEFORE building providers.
- Write math modules test-first; devig correctness matters most.
- Ask before assuming any specific provider's schema, rate limits, or auth model.
- Flag any modeling choice with real consequences (devig method, sharp-venue
  weighting, Kelly fraction) rather than picking silently.
- Never write code that places or executes bets.
