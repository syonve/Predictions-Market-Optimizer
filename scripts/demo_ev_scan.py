"""End-to-end EV scanner demo.

Fetches live Kalshi markets, deviggs the order-book prices into fair
probabilities, runs EV and arb scans, and prints a ranked opportunity table.

Optionally layers in a polling model where polls have been supplied.

Usage
-----
    # From the project root:
    PYTHONPATH=src python scripts/demo_ev_scan.py

    # Specific series (default: KXELECTIONS):
    PYTHONPATH=src python scripts/demo_ev_scan.py KXFED

    # With a Kalshi API key (unlocks authenticated endpoints in future):
    KALSHI_API_KEY=<your-key> PYTHONPATH=src python scripts/demo_ev_scan.py

    # Dry-run against a single hardcoded market (no network):
    PYTHONPATH=src python scripts/demo_ev_scan.py --dry-run

Pipeline
--------
    KalshiProvider.list_markets(series)
        └─► KalshiProvider.fetch_quotes(market)           ← live order book
        └─► PollingModelProvider.estimate(market)         ← polling signal
        └─► FundamentalsModelProvider.estimate(market)    ← incumbency/money/primary
        └─► consensus([pm_snap, *model_snaps], require_sharp=False)
        └─► scan_ev(result, snapshots, min_edge=MIN_EDGE)
        └─► size_ev_signal(signal, bankroll=BANKROLL)
        └─► print ranked table

Configuration (edit the constants below or set env vars)
---------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Configuration — tweak these or override via environment variables
# ---------------------------------------------------------------------------
SERIES: str = os.environ.get("SERIES", "KXELECTIONS")
BANKROLL: float = float(os.environ.get("BANKROLL", "10000"))
MIN_EDGE: float = float(os.environ.get("MIN_EDGE", "0.02"))   # 2% minimum edge
MAX_MARKETS: int = int(os.environ.get("MAX_MARKETS", "50"))   # cap API calls
DEVIG_METHOD: str = os.environ.get("DEVIG_METHOD", "shin")
API_KEY: str | None = os.environ.get("KALSHI_API_KEY")

# ---------------------------------------------------------------------------
# Imports (after PYTHONPATH is set)
# ---------------------------------------------------------------------------
try:
    from fve.models.fundamentals import (
        FundamentalsConfig,
        FundamentalsModelProvider,
        Incumbency,
        RaceFundamentals,
        RaceLevel,
    )
    from fve.models.poll import AggregationConfig, Methodology, Poll, Population
    from fve.models.polling import PollingModelProvider
    from fve.pricing.consensus import NoSharpAnchorError, consensus
    from fve.providers.kalshi import KalshiProvider
    from fve.signals.arb import scan_arb
    from fve.signals.ev import EVSignal, scan_ev
    from fve.sizing.kelly import KellyConfig, size_ev_signal
    from fve.types import Market, MarketSnapshot
except ModuleNotFoundError as exc:
    print(f"\nImport error: {exc}")
    print("Run with: PYTHONPATH=src python scripts/demo_ev_scan.py\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Polling model configuration
# ---------------------------------------------------------------------------
# Add real Poll objects here as you ingest data from DDHQ / FiveThirtyEight.
# Each Poll must have a market_key matching the Kalshi ticker exactly.
# Example (commented out — fill in real data):
#
#   POLLS = [
#       Poll(
#           market_key="KXELECT-2026-SENAZ-DEM-YES",
#           selection_key="yes",
#           prob=0.41,
#           n=800,
#           field_end=date(2026, 5, 28),
#           pollster="Emerson",
#           methodology=Methodology.LIVE_PHONE,
#           population=Population.LIKELY_VOTERS,
#       ),
#   ]
POLLS: list[Poll] = []

POLLING_CONFIG = AggregationConfig(
    half_life_days=21.0,
    reference_n=1000,
    house_effects={},   # e.g. {"Rasmussen": 0.03}
)

# ---------------------------------------------------------------------------
# Fundamentals model configuration
# ---------------------------------------------------------------------------
# Add real RaceFundamentals here as you ingest FEC / state filings.
# Each entry must have a market_key matching the Kalshi ticker exactly.
# Example (commented out — fill in real data):
#
#   FUNDAMENTALS = [
#       RaceFundamentals(
#           market_key="KXELECT-2026-SENAZ-DEM-YES",
#           selection_key="yes",
#           race_level=RaceLevel.FEDERAL,
#           incumbency=Incumbency.CHALLENGER,
#           fundraising_own=4_200_000,
#           fundraising_opponent=6_800_000,
#           primary_margin=0.22,        # won primary by 22 points
#       ),
#   ]
# 2026 LA mayoral general (KXMAYORLA): Bass (incumbent) vs Raman (challenger).
# Incumbency only — the June 2 primary is still being counted as of
# 2026-06-09 (Kalshi's first-round-winner market is itself unresolved), and
# fundraising totals are unverified. Add primary_margin and fundraising once
# certified results / LA Ethics filings are ingested.
FUNDAMENTALS: list[RaceFundamentals] = [
    RaceFundamentals(
        market_key="KXMAYORLA-26-KBAS",
        selection_key="yes",
        race_level=RaceLevel.LOCAL,
        incumbency=Incumbency.INCUMBENT,
    ),
    RaceFundamentals(
        market_key="KXMAYORLA-26-NRAM",
        selection_key="yes",
        race_level=RaceLevel.LOCAL,
        incumbency=Incumbency.CHALLENGER,
    ),
]

FUNDAMENTALS_CONFIG = FundamentalsConfig(
    # Log-odds coefficients — conservative priors, NOT fitted. Fundraising is
    # weighted most for LOCAL races, least for FEDERAL (see module docstring).
    incumbency_coef=0.40,
    fundraising_coefs={
        RaceLevel.FEDERAL: 0.25,
        RaceLevel.STATE: 0.50,
        RaceLevel.LOCAL: 0.80,
    },
    primary_coef=0.30,
)

KELLY_CONFIG = KellyConfig(
    kelly_fraction=0.5,
    max_fraction=0.05,
    confidence=1.0,
)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
BOLD  = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED   = "\033[31m"
DIM   = "\033[2m"
RESET = "\033[0m"

SEP   = "─" * 90
DSEP  = "═" * 90


def _pct(v: float) -> str:
    return f"{v * 100:+.1f}%"


def _prob(v: float) -> str:
    return f"{v * 100:.1f}%"


def _odds(v: float) -> str:
    return f"{v:.3f}"


def _edge_color(edge: float) -> str:
    if edge >= 0.05:
        return GREEN
    if edge >= 0.02:
        return YELLOW
    return RESET


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
class ScanResult(NamedTuple):
    market: Market
    snapshot: MarketSnapshot
    signal: EVSignal
    stake: float
    expected_profit: float
    has_model: bool


def run_scan(
    series: str,
    bankroll: float,
    min_edge: float,
    max_markets: int,
) -> tuple[list[ScanResult], dict]:
    """Run the full pipeline and return ranked signals + summary stats."""

    kalshi = KalshiProvider(api_key=API_KEY)
    models = [
        PollingModelProvider(
            polls=POLLS,
            config=POLLING_CONFIG,
            _reference_date=date.today(),
        ),
        FundamentalsModelProvider(
            races=FUNDAMENTALS,
            config=FUNDAMENTALS_CONFIG,
        ),
    ]

    stats = {
        "series": series,
        "markets_fetched": 0,
        "markets_with_quotes": 0,
        "markets_with_model": 0,
        "consensus_errors": 0,
        "signals_found": 0,
        "arb_found": 0,
        "polls_loaded": len(POLLS),
        "fundamentals_loaded": len(FUNDAMENTALS),
    }

    # 1. List markets
    print(f"\n{DIM}Fetching market list for {series}...{RESET}", end="", flush=True)
    try:
        markets = list(kalshi.list_markets(series))[:max_markets]
    except Exception as exc:
        print(f"\n{RED}Failed to list markets: {exc}{RESET}")
        return [], stats

    stats["markets_fetched"] = len(markets)
    print(f" {len(markets)} markets found.")

    results: list[ScanResult] = []

    # 2. Per-market pipeline
    print(f"{DIM}Scanning {len(markets)} markets...{RESET}")
    for i, market in enumerate(markets):
        print(f"  {DIM}[{i+1}/{len(markets)}] {market.key}{RESET}", end="\r", flush=True)

        # Fetch live quotes
        try:
            snap = kalshi.fetch_quotes(market)
        except Exception:
            continue
        if snap is None:
            continue
        stats["markets_with_quotes"] += 1

        # Model estimates (polling, fundamentals — each optional)
        snapshots: list[MarketSnapshot] = [snap]
        has_model = False
        for model in models:
            model_est = model.estimate(market)
            if model_est is not None:
                snapshots.append(model_est.as_snapshot())
                has_model = True
        if has_model:
            stats["markets_with_model"] += 1

        # Consensus — require_sharp=False: Kalshi PM is our anchor
        try:
            result = consensus(
                snapshots,
                method=DEVIG_METHOD,   # type: ignore[arg-type]
                require_sharp=False,
            )
        except Exception:
            stats["consensus_errors"] += 1
            continue

        # Arb scan (single-venue won't fire, but ready for multi-venue)
        arb = scan_arb(snapshots)
        if arb is not None:
            stats["arb_found"] += 1

        # EV scan
        signals = scan_ev(result, snapshots, min_edge=min_edge)
        for sig in signals:
            sized = size_ev_signal(sig, bankroll=bankroll, config=KELLY_CONFIG)
            results.append(ScanResult(
                market=market,
                snapshot=snap,
                signal=sig,
                stake=sized.stake,
                expected_profit=sized.expected_profit,
                has_model=has_model,
            ))

    # Clear the progress line
    print(" " * 70, end="\r")

    stats["signals_found"] = len(results)
    results.sort(key=lambda r: r.signal.edge, reverse=True)
    return results, stats


# ---------------------------------------------------------------------------
# Dry-run mode (no network)
# ---------------------------------------------------------------------------
def dry_run() -> None:
    """Validate the pipeline with a mocked single-market run (no network)."""
    from fve.types import Market, MarketSnapshot, MarketType, Price, Selection, Venue, VenueKind
    from fve.pricing.consensus import consensus
    from fve.signals.ev import scan_ev
    from fve.sizing.kelly import size_ev_signal

    print(f"\n{BOLD}DRY-RUN MODE{RESET} — no network calls\n")

    market = Market(
        key="KXTEST-DRY-RUN",
        sport="KXTEST",
        event_key="DRY-RUN",
        type=MarketType.BINARY,
        selections=(Selection("yes", "Yes"), Selection("no", "No")),
    )
    kalshi_venue = Venue("kalshi", "Kalshi", VenueKind.PREDICTION_MARKET)
    ts = datetime.now(tz=timezone.utc)

    # Simulate Kalshi showing YES at 52¢ bid / 56¢ ask → fair ≈ 54%
    # Decimal odds: yes_mid = 1/0.54 ≈ 1.852, no_mid = 1/0.46 ≈ 2.174
    snap = MarketSnapshot(
        market=market, venue=kalshi_venue, timestamp=ts,
        prices={
            "yes": Price(decimal_odds=1/0.54, bid=1/0.56, ask=1/0.52),
            "no":  Price(decimal_odds=1/0.46, bid=1/0.48, ask=1/0.44),
        },
    )

    # Simulate three recent polls averaging ~60% YES. Built through the real
    # provider so the confidence score (≈0.35 for 3 polls) flows into the
    # consensus weight — sparse polling gets a quieter voice.
    today = date.today()
    polling = PollingModelProvider(
        polls=[
            Poll(market.key, "yes", 0.59, n=800, field_end=today - timedelta(days=2),
                 pollster="DryRun A", methodology=Methodology.LIVE_PHONE,
                 population=Population.LIKELY_VOTERS),
            Poll(market.key, "yes", 0.60, n=800, field_end=today - timedelta(days=5),
                 pollster="DryRun B", methodology=Methodology.ONLINE_PANEL,
                 population=Population.LIKELY_VOTERS),
            Poll(market.key, "yes", 0.61, n=800, field_end=today - timedelta(days=9),
                 pollster="DryRun C", methodology=Methodology.LIVE_PHONE,
                 population=Population.LIKELY_VOTERS),
        ],
        config=POLLING_CONFIG,
        _reference_date=today,
    )
    poll_est = polling.estimate(market)
    assert poll_est is not None
    model_snap = poll_est.as_snapshot()

    # Simulate fundamentals: YES candidate is a state-level incumbent with a
    # 70/30 money advantage who won their primary by 25 points.
    fundamentals = FundamentalsModelProvider(
        races=[
            RaceFundamentals(
                market_key=market.key,
                selection_key="yes",
                race_level=RaceLevel.STATE,
                incumbency=Incumbency.INCUMBENT,
                fundraising_own=700_000,
                fundraising_opponent=300_000,
                primary_margin=0.25,
            )
        ],
        config=FUNDAMENTALS_CONFIG,
    )
    fund_est = fundamentals.estimate(market)
    assert fund_est is not None
    fund_snap = fund_est.as_snapshot()

    snapshots = [snap, model_snap, fund_snap]
    result = consensus(snapshots, method="shin", require_sharp=False)
    signals = scan_ev(result, snapshots, min_edge=0.0)

    print(f"  Market:           {market.key}")
    print(f"  Kalshi YES:       {_prob(1/snap.prices['yes'].decimal_odds)} (mid)")
    print(f"  Polls YES:        {_prob(poll_est.prob('yes'))}  "
          f"(3 polls, confidence {poll_est.confidence:.2f})")
    print(f"  Fundamentals YES: {_prob(fund_est.prob('yes'))}  "
          f"(incumbent + 70/30 money + primary blowout; confidence {fund_est.confidence:.2f})")
    print(f"  Fair YES:         {_prob(result.prob('yes'))}  "
          f"(confidence-weighted consensus, n_model={result.n_model})")
    print()

    for sig in signals:
        sized = size_ev_signal(sig, bankroll=10000)
        color = _edge_color(sig.edge)
        print(f"  Signal: {sig.selection_key.upper():4s}  "
              f"fair={_prob(sig.fair_prob)}  "
              f"quoted={_odds(sig.decimal_odds)}x  "
              f"edge={color}{_pct(sig.edge)}{RESET}  "
              f"stake=${sized.stake:.0f}")

    print(f"\n{GREEN}Dry-run passed — pipeline wiring is correct.{RESET}\n")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_results(results: list[ScanResult], stats: dict, bankroll: float) -> None:
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(f"\n{BOLD}{DSEP}{RESET}")
    print(f"{BOLD}  Kalshi EV Scanner  ·  {stats['series']}  ·  {now}{RESET}")
    print(f"{BOLD}{DSEP}{RESET}")

    if not results:
        print(f"\n  {YELLOW}No signals above {MIN_EDGE*100:.0f}% edge threshold.{RESET}")
        print(f"  (Lower MIN_EDGE env var to see smaller edges)\n")
    else:
        print(f"\n{BOLD}  TOP OPPORTUNITIES  (edge ≥ {MIN_EDGE*100:.0f}%){RESET}")
        print(f"  {SEP}")
        header = (
            f"  {'MARKET':<40} {'SIDE':<5} {'FAIR':>6} {'QUOTED':>7} "
            f"{'EDGE':>7} {'STAKE':>8}  {'MODEL'}"
        )
        print(f"{BOLD}{header}{RESET}")
        print(f"  {SEP}")

        for r in results:
            sig = r.signal
            color = _edge_color(sig.edge)
            model_tag = "📊" if r.has_model else "  "
            # Truncate long market keys
            mkey = sig.market.key if len(sig.market.key) <= 40 else sig.market.key[:37] + "..."
            print(
                f"  {mkey:<40} "
                f"{sig.selection_key.upper():<5} "
                f"{_prob(sig.fair_prob):>6} "
                f"{_odds(sig.decimal_odds):>7} "
                f"{color}{_pct(sig.edge):>7}{RESET} "
                f"${r.stake:>7.0f}  "
                f"{model_tag}"
            )

        print(f"  {SEP}")
        total_stake = sum(r.stake for r in results)
        total_exp   = sum(r.expected_profit for r in results)
        print(f"  {'TOTAL':<40} {'':5} {'':6} {'':7} {'':7} ${total_stake:>7.0f}")
        print(f"  Expected profit: ${total_exp:.2f}  |  Bankroll: ${bankroll:,.0f}\n")

    # Summary
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"  {SEP}")
    print(f"  Markets fetched:       {stats['markets_fetched']}")
    print(f"  Markets with quotes:   {stats['markets_with_quotes']}")
    print(f"  Signals found:         {stats['signals_found']}")
    print(f"  Arb opportunities:     {stats['arb_found']}")

    model_status = (
        f"{GREEN}{stats['polls_loaded']} polls loaded{RESET}"
        if stats["polls_loaded"] > 0
        else f"{DIM}0 polls — model inactive{RESET}"
    )
    print(f"  Polling model:         {model_status}")
    if stats["polls_loaded"] == 0:
        print(f"  {DIM}  → Add Poll objects to POLLS list in this script to activate{RESET}")

    fund_status = (
        f"{GREEN}{stats['fundamentals_loaded']} races loaded{RESET}"
        if stats["fundamentals_loaded"] > 0
        else f"{DIM}0 races — model inactive{RESET}"
    )
    print(f"  Fundamentals model:    {fund_status}")
    if stats["fundamentals_loaded"] == 0:
        print(f"  {DIM}  → Add RaceFundamentals to FUNDAMENTALS list in this script to activate{RESET}")
    print(f"  Devig method:          {DEVIG_METHOD}")
    print(f"  {SEP}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = sys.argv[1:]

    if "--dry-run" in args:
        dry_run()
        return

    series = args[0] if args else SERIES

    results, stats = run_scan(
        series=series,
        bankroll=BANKROLL,
        min_edge=MIN_EDGE,
        max_markets=MAX_MARKETS,
    )
    print_results(results, stats, BANKROLL)


if __name__ == "__main__":
    main()
