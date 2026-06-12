"""Jump study: are sudden price moves in Kalshi politics information or noise?

The question (user hypothesis): political prediction markets get swept up in
narratives. If that's true, prices right after a big sudden move should be
systematically WRONG in the direction of the move — outcomes should land back
toward the pre-jump price, and fading jumps would be profitable. If jumps are
real information arriving (a poll, a scandal, a debate), post-jump prices
should remain calibrated and there is nothing to fade.

No external data needed: the market's own pre-jump price stands in for the
slow-moving baseline (until historical polls/fundraising are ingested, when
this test can be upgraded to "jump vs model disagreement").

Method
------
1. Universe: settled binary Politics markets, volume ≥ MIN_VOLUME, from the
   calibration backtest cache; classified scheduled/early-close exactly as in
   backtest_recut.py (early-close markets condition prices on the outcome and
   are reported only for contrast).
2. Daily close series from /candlesticks (period_interval=1440), full life
   capped at 400 days.
3. Jump = consecutive daily closes moving ≥ JUMP_SIZE (default 0.15), with
   the pre-jump price away from the pins (0.03–0.97) and the jump occurring
   ≥ MIN_DAYS_OUT days before close (resolution-week moves are mechanical).
4. Metrics, oriented by jump direction (sign = direction of the move):
     continuation vs post price:  sign × (outcome − post_price)
         > 0 → jumps UNDERreact (price should have moved further)
         < 0 → jumps OVERreact (narrative — fade it)
         ≈ 0 → post-jump price calibrated; jumps are information
     continuation vs pre price:   sign × (outcome − pre_price)
         sanity check — strongly > 0 if jumps carry any information at all
     7-day drift:                 sign × (price_7d_later − post_price)
         same read at a fixed horizon, for jumps ≥ 14 days out
   Plus Brier of post-jump prices vs outcomes.

Usage:  uv run python scripts/backtest_jumps.py
        JUMP_SIZE=0.10 MIN_VOLUME=5000 uv run python scripts/backtest_jumps.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fve.tracking.calibration import brier_score

BASE = "https://external-api.kalshi.com/trade-api/v2"
CACHE = Path(__file__).resolve().parent / "cache"
OUT_DIR = Path(__file__).resolve().parent / "output"

MIN_VOLUME = float(os.environ.get("MIN_VOLUME", "1000"))
JUMP_SIZE = float(os.environ.get("JUMP_SIZE", "0.15"))
MIN_DAYS_OUT = float(os.environ.get("MIN_DAYS_OUT", "2"))
MAX_LIFE_DAYS = 400
RATE_SLEEP = 0.12
SCHEDULE_TOL_S = 7 * 86400


# --------------------------------------------------------------------------- #
# HTTP + small helpers (same conventions as the sibling backtest scripts)
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


# --------------------------------------------------------------------------- #
# Universe + details (extends the existing market_details.jsonl cache)
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


def fetch_details(tickers: list[str]) -> dict[str, dict]:
    cache_f = CACHE / "market_details.jsonl"
    cached: dict[str, dict] = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            if line:
                row = json.loads(line)
                cached[row["ticker"]] = row
    todo = [t for t in tickers if t not in cached]
    log(f"details: {len(cached)} cached, {len(todo)} to fetch")
    with cache_f.open("a") as out:
        for i, t in enumerate(todo):
            try:
                m = get(f"/markets/{t}").get("market", {})
            except Exception as exc:  # noqa: BLE001
                log(f"  skip {t}: {exc}")
                continue
            row = {
                "ticker": t,
                "close_time": m.get("close_time"),
                "expected_expiration_time": m.get("expected_expiration_time"),
                "can_close_early": m.get("can_close_early"),
            }
            cached[t] = row
            out.write(json.dumps(row) + "\n")
            time.sleep(RATE_SLEEP)
            if (i + 1) % 100 == 0:
                log(f"  …details {i+1}/{len(todo)}")
    return cached


def classify(detail: dict) -> str:
    ct, ee = detail.get("close_time"), detail.get("expected_expiration_time")
    if not ct or not ee:
        return "unknown"
    return "scheduled" if abs(parse_ts(ee) - parse_ts(ct)) <= SCHEDULE_TOL_S else "early"


# --------------------------------------------------------------------------- #
# Daily close series (cached)
# --------------------------------------------------------------------------- #
def fetch_daily_series(markets: list[dict]) -> dict[str, list[list[float]]]:
    cache_f = CACHE / "daily_closes.jsonl"
    cached: dict[str, list[list[float]]] = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            if line:
                row = json.loads(line)
                cached[row["ticker"]] = row["series"]
    todo = [m for m in markets if m["ticker"] not in cached]
    log(f"daily series: {len(cached)} cached, {len(todo)} to fetch")
    with cache_f.open("a") as out:
        for i, m in enumerate(todo):
            close_ts = parse_ts(m["close_time"])
            start = max(parse_ts(m["open_time"]), close_ts - MAX_LIFE_DAYS * 86400)
            series: list[list[float]] = []
            try:
                data = get(
                    f"/series/{m['series_ticker']}/markets/{m['ticker']}/candlesticks",
                    {"start_ts": str(start), "end_ts": str(close_ts),
                     "period_interval": "1440"},
                )
                for c in data.get("candlesticks", []):
                    p = (c.get("price") or {}).get("close_dollars")
                    try:
                        pf = float(p)
                    except (TypeError, ValueError):
                        continue
                    if 0.0 < pf < 1.0:
                        series.append([c.get("end_period_ts", 0), pf])
            except Exception as exc:  # noqa: BLE001
                log(f"  skip {m['ticker']}: {exc}")
            cached[m["ticker"]] = series
            out.write(json.dumps({"ticker": m["ticker"], "series": series}) + "\n")
            time.sleep(RATE_SLEEP)
            if (i + 1) % 100 == 0:
                log(f"  …series {i+1}/{len(todo)}")
    return cached


# --------------------------------------------------------------------------- #
# Jump detection + metrics
# --------------------------------------------------------------------------- #
def find_jumps(m: dict, series: list[list[float]]) -> list[dict]:
    """Consecutive-day moves ≥ JUMP_SIZE, away from pins, before the endgame."""
    close_ts = parse_ts(m["close_time"])
    outcome = 1.0 if m["result"] == "yes" else 0.0
    jumps = []
    for (t0, p0), (t1, p1) in zip(series, series[1:]):
        if t1 - t0 > 86400 + 3600:  # non-adjacent days (gap in trading) — skip
            continue
        delta = p1 - p0
        if abs(delta) < JUMP_SIZE:
            continue
        if not 0.03 <= p0 <= 0.97:
            continue
        days_out = (close_ts - t1) / 86400
        if days_out < MIN_DAYS_OUT:
            continue
        # price 7 calendar days after the jump (first close ≥ t1+7d)
        p7 = next((p for t, p in series if t >= t1 + 7 * 86400), None)
        jumps.append({
            "pre": p0, "post": p1, "delta": delta, "days_out": days_out,
            "outcome": outcome, "p7": p7,
        })
    return jumps


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / max(1, n - 1)
    return mu, math.sqrt(var / n)


def render(jumps_by_cut: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    w = lines.append
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    w("=" * 78)
    w(f"  Kalshi Politics Jump Study · {now}")
    w(f"  Jump = daily close move ≥ {JUMP_SIZE:.2f}, pre-price in [0.03, 0.97],")
    w(f"  ≥ {MIN_DAYS_OUT:.0f} days before close. Volume ≥ {MIN_VOLUME:.0f}.")
    w("  Oriented metrics: positive = move direction kept paying (underreaction),")
    w("  negative = outcomes fell back toward the old price (overreaction/fade),")
    w("  ≈ zero = post-jump price was right; jumps are information.")
    w("=" * 78)

    for cut, jumps in jumps_by_cut.items():
        w("")
        w(f"  ── {cut.upper()} markets: {len(jumps)} jumps " + "─" * 30)
        if len(jumps) < 20:
            w("  Too few jumps for a read.")
            continue
        n_up = sum(1 for j in jumps if j["delta"] > 0)
        w(f"  direction: {n_up} up / {len(jumps) - n_up} down · "
          f"median |move| {sorted(abs(j['delta']) for j in jumps)[len(jumps)//2]:.2f} · "
          f"median days-out {sorted(j['days_out'] for j in jumps)[len(jumps)//2]:.0f}")

        sgn = lambda j: 1.0 if j["delta"] > 0 else -1.0  # noqa: E731
        cont_post = [sgn(j) * (j["outcome"] - j["post"]) for j in jumps]
        cont_pre = [sgn(j) * (j["outcome"] - j["pre"]) for j in jumps]
        mu_post, se_post = mean_se(cont_post)
        mu_pre, se_pre = mean_se(cont_pre)
        bs = brier_score([(j["post"], j["outcome"] > 0.5) for j in jumps])

        w(f"  continuation vs POST-jump price: {mu_post:+.3f} ± {se_post:.3f}")
        w(f"  continuation vs PRE-jump price:  {mu_pre:+.3f} ± {se_pre:.3f}")
        w(f"  Brier of post-jump prices:       {bs:.4f}")

        drift7 = [
            sgn(j) * (j["p7"] - j["post"])
            for j in jumps
            if j["p7"] is not None and j["days_out"] >= 14
        ]
        if len(drift7) >= 20:
            mu7, se7 = mean_se(drift7)
            w(f"  7-day post-jump drift (n={len(drift7)}):     {mu7:+.3f} ± {se7:.3f}")
        else:
            w(f"  7-day drift: only {len(drift7)} jumps ≥14d out — no read.")
    w("")
    w("  Caveat: jumps in sibling markets of one event (rival candidates) are")
    w("  correlated; effective sample is smaller than the jump count.")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    universe = load_universe()
    log(f"universe: {len(universe)} settled binary markets (vol ≥ {MIN_VOLUME:.0f})")
    details = fetch_details([m["ticker"] for m in universe])
    series = fetch_daily_series(universe)

    jumps_by_cut: dict[str, list[dict]] = {"scheduled": [], "early": []}
    for m in universe:
        d = details.get(m["ticker"])
        if d is None:
            continue
        cut = classify(d)
        if cut not in jumps_by_cut:
            continue
        jumps_by_cut[cut].extend(find_jumps(m, series.get(m["ticker"], [])))

    report = render(jumps_by_cut)
    print(report)
    out_f = OUT_DIR / f"backtest_jumps_{datetime.now(tz=timezone.utc):%Y-%m-%d}.txt"
    out_f.write_text(report)
    log(f"Report saved to {out_f}")


if __name__ == "__main__":
    main()
