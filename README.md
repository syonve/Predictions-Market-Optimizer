# fve — prediction-market fair-value engine

A Python tool that looks at betting odds and prediction-market prices from many
places at once, works out what the *true* probability of each outcome probably
is, and flags prices that look wrong — either because two venues disagree with
each other (arbitrage) or because the market disagrees with an independent
model (a positive-EV bet). One fair-value engine drives both: arbitrage is just
the extreme case of a large deviation from fair value.

**Analysis and signal only — this code never places or executes bets.**

## The idea, in plain English

**Prices are probabilities.** A Kalshi contract trading at 54¢ pays $1 if the
event happens, so the market is saying "54% likely." Sportsbook odds encode the
same thing: decimal odds of 2.00 mean a $1 bet returns $2, which is fair only
if the chance is 50%. Implied probability is simply `1 / decimal odds`.

**But quoted prices are deliberately a little wrong.** A bookmaker quoting both
sides of a coin flip won't offer 50% / 50% — they'll offer something like
52.4% / 52.4%, which sums to more than 100%. That extra ~4.8% is their margin
(the "vig" or "juice"). Before prices can be compared or trusted, the margin
has to be stripped out so the probabilities sum to exactly 100%. That step is
called **devigging**, and there's more than one defensible way to do it (see
[Devig methods](#devig-methods) below) — the engine makes the method a config
choice rather than hardcoding one, because the choice changes which bets look
attractive.

**Not all opinions are equal.** Once every venue's prices are devigged into
clean probabilities, the engine blends them into one **consensus fair value**.
The blend is weighted: "sharp" venues — bookmakers like Pinnacle that welcome
professional bettors, and deep exchange order books — get the most weight,
because their prices are sharpened by smart money. Recreational ("soft")
sportsbooks get less. Independent models get a say mainly where sharp prices
don't exist — political markets, obscure leagues, player props.

**Two independent models, and counting.** Models are synthetic "venues": they
produce a probability like a market does, but their prices can't be bet on —
they only inform fair value (the signal scanners explicitly skip them).

- *Polling model* — a weighted average of polls, where each poll counts more
  if it's recent, has a large sample, screens for likely voters, and comes
  from a quality pollster, with known partisan "house effects" subtracted.
- *Fundamentals model* — the structural facts of an election that predict
  outcomes even without polling: **incumbency** (sitting officeholders
  out-perform), **fundraising** (who's winning the money race — weighted
  most heavily in local races, then state, then federal, because in
  low-information down-ballot races money buys the name recognition that
  decides the outcome, while federal races are saturated with free
  information), and **primary results** (a dominant primary win signals
  candidate quality and party unity; a squeaker signals a divided base).
  Each signal nudges the probability up or down by a configurable amount,
  in log-odds space so nudges matter most in close races. The default
  coefficient sizes are conservative priors, not fitted estimates — the
  calibration layer is the gate before trusting them.

**Models are complements, not equal votes.** Each model estimate carries a
confidence score reflecting how much data backs it (poll count/recency/quality
for polling; how many of the three signals are present for fundamentals), and
the consensus blend scales each model's weight by it. The result matches how
the layers should behave: when polling is rich and fresh, it speaks at close
to full volume; when polling is thin — local races, early cycle — its voice
shrinks and the slow-moving structural baseline carries relatively more. The
market itself stays the anchor: even both models at maximum confidence
(0.30 + 0.15) sum to just under Kalshi's 0.50, deliberately. Political
markets do get swept up in narratives, but an unproven model can't be
trusted to outvote real money either — models earn more weight by beating
the closing line and the calibration scores, not by assumption.

**Edge is the whole game.** If the engine's fair probability for an outcome is
56.2% but some venue's price implies only 54%, betting it there has positive
expected value. The **edge** is the expected profit per dollar staked:

```
edge = fair probability × decimal odds − 1
```

A +4.2% edge means that for every $100 bet, you'd expect to be up $4.20 on
average over many repetitions — even though any single bet can lose.

**Bet sizing is its own discipline.** Even a genuinely good bet ruins you if
you bet too much of your bankroll on it. The engine sizes suggested stakes with
the **Kelly criterion** (the growth-optimal fraction of bankroll), then halves
it and caps it at 5% of bankroll, because full Kelly is famously too aggressive
when your probabilities are only estimates.

**Trust must be earned by measurement, not assumed.** Two tracking layers keep
the engine honest:

- **Calibration** — when the engine says "60%", does the event actually happen
  about 60% of the time? Measured with Brier scores and reliability diagrams.
  Until the fair-value model is shown to be calibrated, a "deviation" is just a
  disagreement, not an edge.
- **CLV (closing line value)** — did the price move *toward* our number by the
  time the market closed? Closing prices are the sharpest prices that exist,
  so consistently beating them is the strongest early evidence of real edge.
  Win/loss results are too noisy in the short run to tell you anything.

## The pipeline

```
venue prices (Kalshi, sportsbooks, aggregator)
      │  normalize at the boundary (providers/)
      ▼
implied probabilities, margin still in
      │  strip the vig (pricing/devig.py — proportional / shin / power)
      ▼
clean per-venue probabilities
      │  blend, weighted toward sharp venues (pricing/consensus.py)
      ▼
consensus fair value          ◄── independent models layer in here (models/)
      │
      ├─► signals/arb.py   venues disagree enough to lock in profit?
      ├─► signals/ev.py    any quoted price beats fair value by ≥ min edge?
      │        └─► sizing/kelly.py   how much to stake (half-Kelly, capped)
      ▼
tracking/  clv.py (did closing prices confirm us?) · calibration.py (Brier)
```

## Try it — the dry run

The demo scanner has a no-network mode that pushes one hand-built market
through the entire pipeline:

```bash
uv run python scripts/demo_ev_scan.py --dry-run
```

```
DRY-RUN MODE — no network calls

  Market:           KXTEST-DRY-RUN
  Kalshi YES:       54.0% (mid)                      ← the (simulated) market
  Polls YES:        59.9%  (3 polls, confidence 0.34) ← thin polling, quiet voice
  Fundamentals YES: 66.3%  (confidence 0.50)          ← incumbent + money + primary
  Fair YES:         57.3%  (confidence-weighted consensus)

  Signal: YES  fair=57.3%  quoted=1.852x  edge=+6.0%  stake=$354
```

Reading the signal line: the engine believes YES is 57.3% likely; the venue's
price (1.852 decimal odds, i.e. implied 54%) is too cheap relative to that;
the expected profit is +6.0¢ per $1; and half-Kelly sizing on a $10,000
bankroll suggests a $354 stake. The live mode (no `--dry-run` flag) runs the
same pipeline against real Kalshi markets.

## Layout

```
src/fve/
  types.py            Normalized, provider-agnostic domain types (the boundary)
  providers/
    base.py           OddsProvider interface (Protocol) — every source satisfies it
    kalshi.py         Direct Kalshi REST client
  pricing/
    devig.py          Pluggable devig: proportional, Shin, power
    consensus.py      Sharp-weighted consensus fair value
  models/
    base.py           ModelProvider interface + MODEL venue kind
    poll.py           Pure poll-aggregation math (recency, sample size, house effects)
    polling.py        PollingModelProvider — polls → probability estimates
    fundamentals.py   Election fundamentals — incumbency, fundraising (weighted
                      by race level: local > state > federal), primary margins
  signals/
    ev.py             Positive-EV scan: fair value vs. quoted odds
    arb.py            Cross-venue arbitrage scan
  sizing/
    kelly.py          Half-Kelly stake sizing with bankroll cap
  tracking/
    clv.py            Closing line value
    calibration.py    Brier score, reliability diagrams
scripts/
  demo_ev_scan.py                 End-to-end Kalshi scanner (+ --dry-run mode)
  backtest_kalshi_calibration.py  Is Kalshi politics pricing calibrated? (Brier/reliability)
  backtest_recut.py               Same, split by close-timing regime (artifact control)
  backtest_jumps.py               Are sudden moves information or overreaction?
  output/                         Saved run reports (committed)
  cache/                          Fetched API data (disposable, gitignored)
tests/                Known-answer tests for every math module
```

## The two load-bearing abstractions

**`OddsProvider`** (`providers/base.py`) is the single contract every data
source satisfies — aggregator (sharp + closing odds), The Odds API (soft
books), a direct Kalshi/Polymarket client. Pricing depends only on this
interface and on `fve.types`; no provider schema leaks past the boundary.

**`devig`** (`pricing/devig.py`) converts raw implied probabilities into a fair
distribution. The method is pluggable and selected by config / the calibration
harness — never silently hardcoded, because the method changes which edges
surface.

### Devig methods

- `proportional` — margin removed in proportion to implied prob. Baseline; does
  not correct favorite-longshot bias (the well-documented tendency of bettors
  to overpay for longshots, which skews quoted prices).
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
uv run --extra dev pytest
```

## Status

Done: normalized types, `OddsProvider` interface, devig core (3 methods),
sharp-weighted consensus (with `require_sharp=False` for prediction-market-anchored
markets), EV + arb scans (both skip MODEL venues — a model's price is not
bettable), half-Kelly sizing, CLV + calibration tracking, polling model,
election-fundamentals model (incumbency / fundraising by race level / primary
margins), direct Kalshi client, end-to-end demo scanner with dry-run mode.
252 tests green.

Resolved (was flagged): model weight in consensus now scales with each
model's confidence score (`Price.size` carries confidence for MODEL venues).
Even both models at full confidence (0.30 + 0.15) stay under Kalshi's 0.50 —
the market remains the anchor until calibration evidence justifies more.

First live Kalshi runs (2026-06-09, LA mayoral KXMAYORLA + DC primary
KXDCMAYORD — see scripts/output/) surfaced and fixed two bugs: the Kalshi
list endpoint needs `status=open` (`status=active` silently returns nothing),
and `scan_arb` priced legs at bid/ask midpoints, flagging a phantom arb that
cost $1.019 per $1 at executable prices (now uses executable odds + a
`min_roi` floor).

Resolved (was flagged): order-book depth scaling is now normalized and
bounded — `clamp(sqrt(avg_size / 1000), 0.1, 2.0)` (`LiquidityConfig`). A
book at the 1000-contract reference depth keeps its base weight; the deepest
books cap at 2x, which promotes a deep prediction market (0.5 → 1.0) to
exactly sharp-anchor weight and no further — deep PM books count as sharp,
per the core design. The live-run Bass book (~10k contracts) now weighs 1.0
instead of 49.9. The reference depth is in Kalshi-contract units; set it per
venue class when a second order-book venue is added.

First backtest (2026-06-12, scripts/backtest_kalshi_calibration.py +
backtest_recut.py — reports in scripts/output/): calibrated Kalshi's own
politics prices on 600 high-volume resolved binary markets, scored at 24h/7d
before close. Headline: on SCHEDULED-close markets (election-style, no
early-close conditioning; n=229 at 24h) Kalshi is well-calibrated — Brier
0.075 vs 0.199 baseline, with <10¢ and >80¢ bins within ±2 points of perfect.
The raw pooled run had shown mid-range prices "underpriced" by 9–21 points;
the re-cut proved that to be an artifact of markets that close early *because
the event occurs* (their pre-close prices are conditioned on YES). Lessons
encoded: (1) the market anchor weight is empirically justified for political
markets — no free edge from naive "the market is biased" priors; (2) weak
hint of classic longshot overpricing in the 10–20¢ band (gap −0.08, n=21) —
worth re-testing with a larger sample; (3) calibration methodology (close-
timing conditioning, correlated outcomes within events) matters as much as
the score itself.

Jump study (2026-06-12, scripts/backtest_jumps.py): tested whether sudden
moves (≥15 pts in a day, ≥2 days before close) in politics markets are
information or narrative overreaction. On scheduled-close markets (304
jumps): jumps are directionally real — outcomes landed +21.6 pts (±2.6) from
the PRE-jump price in the move's direction — but systematically OVERSHOOT:
outcomes landed −7.4 pts (±2.6) back from the POST-jump price, and a week
later prices had given back −10.1 pts (±2.5) of the move on average (n=123).
Same pattern on early-close markets (−5.4 / −8.4 pts).

Fade tradeability test (2026-06-12, scripts/backtest_jump_fade.py): the
overreaction is REAL but NOT tradeable with market orders. With an
event-level bootstrap (sibling markets of one event cluster as a single
observation — 304 jumps are only 77 independent events), the frictionless
fade survives: +7.4 pts, 95% CI [+2.2, +11.9]. But executed at actual
closing quotes net of Kalshi fees it evaporates: −1.1 pts [−6.5, +3.9]
entering the jump day, −3.8 pts [−8.5, +0.6] entering the next day. The
bid-ask spread on jump days widens by just about the size of the mispricing.
Three uses survive anyway: (1) post-jump prices overstate the move by ~7 pts
— the consensus engine should discount a freshly-jumped market's weight (or
nudge fair value partway back toward the pre-jump price) rather than treat
it as settled truth; (2) never chase momentum after a jump; (3) a
maker-side fade (posting limit orders inside the spread) might capture the
edge — untested, needs order-book data. The arc of this investigation is
the whole methodology lesson: "+20 pts of edge" (raw) → artifact re-cut →
"+7 pts real phenomenon" (bootstrap) → "0 pts at executable prices" (fees +
spread). Every gate mattered.

Next, in rough order:
1. Real data into the models — polls via an aggregator API, federal
   fundraising via the OpenFEC REST API (free key; state/local filings are
   non-standardized and come later), race context (incumbency, primary
   margins, open seats) from Ballotpedia (no API — manual or scraped).
2. Model backtests against the now-established market baseline: replay
   historical polls/fundamentals through the models on resolved races and
   require them to beat (or usefully complement) the market's Brier before
   any weight increase.
3. Enlarge the market-calibration sample (raise MAX_MARKETS, more horizons)
   to confirm or kill the 10–20¢ longshot-overpricing hint.
4. CLV tracking from the first real signal.
5. Later model upgrades: partisan lean / prior margins as the fundamentals
   prior (the `prior_prob` hook exists), cash-on-hand and burn rate,
   contested-primary flag, and a staged fundamentals-as-prior →
   polls-as-update structure once there's data to fit it on.

(Done since the original roadmap: live Kalshi scanner runs, the market-
baseline calibration backtest above.)
