"""Jump-fade tradeability test: does the overreaction edge survive reality?

backtest_jumps.py found that ≥15-point daily moves in Kalshi politics markets
overshoot: outcomes land ~7 points back from the post-jump price, and prices
give back ~10 points within a week. Before that becomes a model input, two
honesty checks (this script):

1. EVENT-LEVEL BOOTSTRAP. Jumps cluster: when a debate moves a race, every
   candidate market in the event jumps together, so 304 jumps are far fewer
   independent observations. We resample EVENTS (not jumps) with replacement
   and report 95% percentile confidence intervals. If zero is inside the CI,
   the edge is not established.

2. EXECUTABLE PRICES, NET OF FEES. The fade trade is: jump UP → buy NO at
   the NO ask (≈ 1 − yes_bid); jump DOWN → buy YES at the yes ask. Entry at
   the jump day's closing quotes (standard daily-signal convention; assumes
   you detect the jump just before close) and, stricter, at the NEXT day's
   closing quotes. Hold to resolution (no exit fee). Kalshi's taker fee is
   modeled as FEE_RATE × price × (1 − price) per contract (FEE_RATE default
   0.07 — VERIFY against kalshi.com/fees before trading; configurable).

Edge units: probability points per contract (cents per $1 payout).

Usage:  uv run python scripts/backtest_jump_fade.py
        FEE_RATE=0.10 JUMP_SIZE=0.10 uv run python scripts/backtest_jump_fade.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://external-api.kalshi.com/trade-api/v2"
CACHE = Path(__file__).resolve().parent / "cache"
OUT_DIR = Path(__file__).resolve().parent / "output"

MIN_VOLUME = float(os.environ.get("MIN_VOLUME", "1000"))
JUMP_SIZE = float(os.environ.get("JUMP_SIZE", "0.15"))
MIN_DAYS_OUT = float(os.environ.get("MIN_DAYS_OUT", "2"))
FEE_RATE = float(os.environ.get("FEE_RATE", "0.07"))
MAX_LIFE_DAYS = 400
RATE_SLEEP = 0.12
SCHEDULE_TOL_S = 7 * 86400
N_BOOT = 2000
SEED = 20260612


# --------------------------------------------------------------------------- #
# HTTP + helpers (same conventions as sibling backtest scripts)
# --------------------------------------------------------------------------- #
def get(path: str, params: dict[str, str] | None = None) -> dict:
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def volume(m: dict) -> float:
    try:
        return float(m.get("volume_fp") or m.get("volume") or 0)
    except (TypeError, ValueError):
        return 0.0


def fee(price: float) -> float:
    """Kalshi taker fee per contract at a given price (approximation)."""
    return FEE_RATE * price * (1.0 - price)


# --------------------------------------------------------------------------- #
# Universe, details, classification (reuses existing caches)
# --------------------------------------------------------------------------- #
def load_universe() -> list[dict]:
    rows = [
        json.loads(l)
        for l in (CACHE / "settled_markets.jsonl").read_text().splitlines()
        if l
    ]
    rows = list({r["ticker"]: r for r in rows}.values())
    return [
        r for r in rows
        if r.get("market_type") == "binary"
        and r.get("result") in ("yes", "no")
        and r.get("open_time") and r.get("close_time")
        and volume(r) >= MIN_VOLUME
    ]


def load_details() -> dict[str, dict]:
    cached: dict[str, dict] = {}
    f = CACHE / "market_details.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            if line:
                row = json.loads(line)
                cached[row["ticker"]] = row
    return cached


def classify(detail: dict) -> str:
    ct, ee = detail.get("close_time"), detail.get("expected_expiration_time")
    if not ct or not ee:
        return "unknown"
    return "scheduled" if abs(parse_ts(ee) - parse_ts(ct)) <= SCHEDULE_TOL_S else "early"


# --------------------------------------------------------------------------- #
# Daily quotes: [ts, trade_close, yes_bid_close, yes_ask_close]
# --------------------------------------------------------------------------- #
def fetch_daily_quotes(markets: list[dict]) -> dict[str, list[list[float | None]]]:
    cache_f = CACHE / "daily_quotes.jsonl"
    cached: dict[str, list[list[float | None]]] = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            if line:
                row = json.loads(line)
                cached[row["ticker"]] = row["series"]
    todo = [m for m in markets if m["ticker"] not in cached]
    log(f"daily quotes: {len(cached)} cached, {len(todo)} to fetch")

    def fp(block: dict | None) -> float | None:
        if not block:
            return None
        try:
            v = float(block.get("close_dollars"))
        except (TypeError, ValueError):
            return None
        return v if 0.0 < v < 1.0 else None

    with cache_f.open("a") as out:
        for i, m in enumerate(todo):
            close_ts = parse_ts(m["close_time"])
            start = max(parse_ts(m["open_time"]), close_ts - MAX_LIFE_DAYS * 86400)
            series: list[list[float | None]] = []
            try:
                data = get(
                    f"/series/{m['series_ticker']}/markets/{m['ticker']}/candlesticks",
                    {"start_ts": str(start), "end_ts": str(close_ts),
                     "period_interval": "1440"},
                )
                for c in data.get("candlesticks", []):
                    trade = fp(c.get("price"))
                    if trade is None:
                        continue
                    series.append([
                        c.get("end_period_ts", 0),
                        trade,
                        fp(c.get("yes_bid")),
                        fp(c.get("yes_ask")),
                    ])
            except Exception as exc:  # noqa: BLE001
                log(f"  skip {m['ticker']}: {exc}")
            cached[m["ticker"]] = series
            out.write(json.dumps({"ticker": m["ticker"], "series": series}) + "\n")
            time.sleep(RATE_SLEEP)
            if (i + 1) % 100 == 0:
                log(f"  …quotes {i+1}/{len(todo)}")
    return cached


# --------------------------------------------------------------------------- #
# Jumps with executable entries
# --------------------------------------------------------------------------- #
def find_jumps(m: dict, series: list[list[float | None]]) -> list[dict]:
    close_ts = parse_ts(m["close_time"])
    outcome = 1.0 if m["result"] == "yes" else 0.0
    jumps = []
    for i in range(1, len(series)):
        t0, p0 = series[i - 1][0], series[i - 1][1]
        t1, p1, bid1, ask1 = series[i]
        if t1 - t0 > 86400 + 3600:
            continue
        delta = p1 - p0
        if abs(delta) < JUMP_SIZE or not 0.03 <= p0 <= 0.97:
            continue
        if (close_ts - t1) / 86400 < MIN_DAYS_OUT:
            continue
        nxt = series[i + 1] if i + 1 < len(series) else None
        jumps.append({
            "event": m.get("event_ticker") or m["ticker"],
            "up": delta > 0,
            "pre": p0, "post": p1, "outcome": outcome,
            "bid_t": bid1, "ask_t": ask1,
            "bid_n": nxt[2] if nxt else None,
            "ask_n": nxt[3] if nxt else None,
        })
    return jumps


def fade_profit(j: dict, bid: float | None, ask: float | None) -> float | None:
    """Net profit per contract of fading the jump at the given quotes.

    Up-jump → buy NO at (1 − yes_bid): profit = yes_bid − outcome − fee.
    Down-jump → buy YES at yes_ask:    profit = outcome − yes_ask − fee.
    Returns None when the needed quote is missing or degenerate.
    """
    if j["up"]:
        if bid is None or not 0.01 <= bid <= 0.99:
            return None
        cost = 1.0 - bid
        return (1.0 - j["outcome"]) - cost - fee(cost)
    if ask is None or not 0.01 <= ask <= 0.99:
        return None
    return j["outcome"] - ask - fee(ask)


# --------------------------------------------------------------------------- #
# Event-level (cluster) bootstrap
# --------------------------------------------------------------------------- #
def cluster_bootstrap(
    values_by_event: dict[str, list[float]],
    n_boot: int = N_BOOT,
) -> tuple[float, float, float]:
    """Mean and 95% CI of the grand mean, resampling events with replacement."""
    events = list(values_by_event)
    all_vals = [v for vs in values_by_event.values() for v in vs]
    mu = sum(all_vals) / len(all_vals)
    rng = random.Random(SEED)
    means = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(len(events)):
            sample.extend(values_by_event[rng.choice(events)])
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return mu, lo, hi


def group(jumps: list[dict], key) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for j in jumps:
        v = key(j)
        if v is None:
            continue
        out.setdefault(j["event"], []).append(v)
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def render(jumps_by_cut: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    w = lines.append
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    w("=" * 78)
    w(f"  Jump-Fade Tradeability Test · {now}")
    w(f"  Jump ≥ {JUMP_SIZE:.2f}/day · vol ≥ {MIN_VOLUME:.0f} · fee model "
      f"{FEE_RATE:.2f}·p·(1−p) per contract")
    w("  All CIs: 95% EVENT-level bootstrap (clusters of correlated sibling")
    w("  markets count once). Edge is established only if the CI excludes 0.")
    w("  Units: probability points per contract (¢ per $1 payout).")
    w("=" * 78)

    metrics = [
        ("fade vs post-jump TRADE price (frictionless re-check)",
         lambda j: (j["post"] - j["outcome"]) if j["up"] else (j["outcome"] - j["post"])),
        ("fade at SAME-day closing quotes, net of fees",
         lambda j: fade_profit(j, j["bid_t"], j["ask_t"])),
        ("fade at NEXT-day closing quotes, net of fees",
         lambda j: fade_profit(j, j["bid_n"], j["ask_n"])),
    ]

    for cut, jumps in jumps_by_cut.items():
        n_events = len({j["event"] for j in jumps})
        w("")
        w(f"  ── {cut.upper()}: {len(jumps)} jumps in {n_events} events " + "─" * 22)
        if len(jumps) < 20:
            w("  Too few jumps for a read.")
            continue
        for label, key in metrics:
            by_event = group(jumps, key)
            n = sum(len(v) for v in by_event.values())
            if n < 20:
                w(f"  {label}: only {n} usable jumps — no read.")
                continue
            mu, lo, hi = cluster_bootstrap(by_event)
            verdict = "EDGE" if lo > 0 else ("loss" if hi < 0 else "not established")
            w(f"  {label}:")
            w(f"      {mu*100:+.1f} pts  [{lo*100:+.1f}, {hi*100:+.1f}]  "
              f"(n={n}, events={len(by_event)}) → {verdict}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    universe = load_universe()
    details = load_details()
    log(f"universe: {len(universe)} markets; details cached: {len(details)}")
    quotes = fetch_daily_quotes(universe)

    jumps_by_cut: dict[str, list[dict]] = {"scheduled": [], "early": []}
    for m in universe:
        d = details.get(m["ticker"])
        if d is None:
            continue
        cut = classify(d)
        if cut in jumps_by_cut:
            jumps_by_cut[cut].extend(find_jumps(m, quotes.get(m["ticker"], [])))

    report = render(jumps_by_cut)
    print(report)
    out_f = OUT_DIR / f"backtest_jump_fade_{datetime.now(tz=timezone.utc):%Y-%m-%d}.txt"
    out_f.write_text(report)
    log(f"Report saved to {out_f}")


if __name__ == "__main__":
    main()
