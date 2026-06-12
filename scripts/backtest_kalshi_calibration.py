"""Backtest: is Kalshi's politics pricing calibrated?

The foundational backtest, runnable with zero external data: take every
RESOLVED binary market in Kalshi's Politics category, look up the price it
traded at N hours before close, and ask whether those prices were honest
probabilities — when the market said 70%, did the event happen ~70% of the
time?

Why this comes before any model backtest: the engine anchors fair value on
the market. If Kalshi politics prices are well-calibrated, that anchoring is
justified and any model must clear a high bar. If they show systematic bias
(the classic favorite-longshot pattern: longshots overpriced, favorites
underpriced), that bias is itself quantified edge — and tells us which devig
corrections to trust.

Pipeline
--------
    Phase 1  /series?category=Politics            → all politics series
    Phase 2  /markets?series_ticker=X&status=settled  (per series, cached)
    Phase 3  filter: binary, result in {yes,no}, volume ≥ MIN_VOLUME,
             lifetime ≥ horizon + buffer; fetch hourly candlesticks,
             extract last-trade price at each horizon before close
    Phase 4  Brier + reliability diagram (fve.tracking.calibration)

All HTTP responses are cached under scripts/cache/ — interrupt and re-run
freely; only missing data is fetched.

Usage
-----
    uv run python scripts/backtest_kalshi_calibration.py            # full run
    MAX_MARKETS=200 uv run python scripts/backtest_kalshi_calibration.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fve.tracking.calibration import CalibrationResult, reliability_diagram

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE = "https://external-api.kalshi.com/trade-api/v2"
CACHE = Path(__file__).resolve().parent / "cache"
OUT_DIR = Path(__file__).resolve().parent / "output"

MIN_VOLUME = float(os.environ.get("MIN_VOLUME", "1000"))    # lifetime contracts
MAX_MARKETS = int(os.environ.get("MAX_MARKETS", "600"))     # candlestick budget
HORIZONS_H = (24, 168)                                       # 1 day, 7 days
RATE_SLEEP = 0.12                                            # ~8 req/s, polite

# --------------------------------------------------------------------------- #
# HTTP with retry
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
        except Exception as exc:  # noqa: BLE001 — retry any transient failure
            if attempt == 4:
                raise
            wait = 2.0 * (attempt + 1)
            print(f"    retry {attempt+1} after error: {exc} (sleep {wait}s)",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Phase 1: politics series
# --------------------------------------------------------------------------- #
def fetch_series() -> list[str]:
    f = CACHE / "politics_series.json"
    if f.exists():
        return json.loads(f.read_text())
    data = get("/series", {"category": "Politics"})
    tickers = sorted({s["ticker"] for s in data.get("series", [])})
    f.write_text(json.dumps(tickers))
    log(f"Phase 1: {len(tickers)} politics series")
    return tickers


# --------------------------------------------------------------------------- #
# Phase 2: settled markets per series (cached per series)
# --------------------------------------------------------------------------- #
def fetch_settled_markets(series: list[str]) -> list[dict]:
    done_f = CACHE / "settled_done.json"
    out_f = CACHE / "settled_markets.jsonl"
    done: set[str] = set(json.loads(done_f.read_text())) if done_f.exists() else set()

    todo = [s for s in series if s not in done]
    log(f"Phase 2: {len(done)} series cached, {len(todo)} to fetch")

    with out_f.open("a") as out:
        for i, st in enumerate(todo):
            params = {"series_ticker": st, "status": "settled",
                      "limit": "200", "mve_filter": "exclude"}
            while True:
                data = get("/markets", params)
                for m in data.get("markets", []):
                    keep = {k: m.get(k) for k in (
                        "ticker", "event_ticker", "market_type", "result",
                        "open_time", "close_time", "volume_fp", "volume",
                        "title", "yes_sub_title",
                    )}
                    keep["series_ticker"] = st
                    out.write(json.dumps(keep) + "\n")
                cursor = data.get("cursor", "")
                if not cursor:
                    break
                params["cursor"] = cursor
            done.add(st)
            time.sleep(RATE_SLEEP)
            if (i + 1) % 100 == 0:
                done_f.write_text(json.dumps(sorted(done)))
                log(f"  …{i+1}/{len(todo)} series scanned")
    done_f.write_text(json.dumps(sorted(done)))

    markets = [json.loads(line) for line in out_f.read_text().splitlines() if line]
    # The resumable JSONL can hold duplicates if a run died mid-series — dedupe.
    by_ticker = {m["ticker"]: m for m in markets}
    log(f"Phase 2: {len(by_ticker)} settled politics markets total")
    return list(by_ticker.values())


# --------------------------------------------------------------------------- #
# Phase 3: candlestick price at each horizon before close
# --------------------------------------------------------------------------- #
def _parse_ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _volume(m: dict) -> float:
    v = m.get("volume_fp") or m.get("volume") or 0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def select_sample(markets: list[dict]) -> list[dict]:
    """Binary markets with a yes/no result, enough volume, enough lifetime."""
    sample = []
    for m in markets:
        if m.get("market_type") != "binary":
            continue
        if m.get("result") not in ("yes", "no"):
            continue
        if not m.get("open_time") or not m.get("close_time"):
            continue
        if _volume(m) < MIN_VOLUME:
            continue
        lifetime_h = (_parse_ts(m["close_time"]) - _parse_ts(m["open_time"])) / 3600
        if lifetime_h < min(HORIZONS_H) + 6:
            continue
        sample.append(m)
    # Highest-volume markets first: most informative, and bounds the budget.
    sample.sort(key=_volume, reverse=True)
    return sample[:MAX_MARKETS]


def price_at_horizons(m: dict) -> dict[int, float]:
    """Last-trade price at each horizon before close, from hourly candles.

    A candle's price.close_dollars is the most recent trade price as of that
    candle. "0.0000"/missing means no trade has printed yet — walk back to an
    earlier candle (up to 48h) rather than fabricating a price.
    """
    close_ts = _parse_ts(m["close_time"])
    open_ts = _parse_ts(m["open_time"])
    start = max(open_ts, close_ts - (max(HORIZONS_H) + 56) * 3600)
    data = get(
        f"/series/{m['series_ticker']}/markets/{m['ticker']}/candlesticks",
        {"start_ts": str(start), "end_ts": str(close_ts), "period_interval": "60"},
    )
    candles = data.get("candlesticks", [])
    out: dict[int, float] = {}
    for h in HORIZONS_H:
        target = close_ts - h * 3600
        if target < open_ts + 3600:
            continue  # market wasn't alive at this horizon
        best_ts, best_price = None, None
        for c in candles:
            ts = c.get("end_period_ts", 0)
            if ts > target or ts < target - 48 * 3600:
                continue
            p = (c.get("price") or {}).get("close_dollars")
            try:
                pf = float(p)
            except (TypeError, ValueError):
                continue
            if pf <= 0.0 or pf >= 1.0:
                continue  # no trade yet / degenerate print
            if best_ts is None or ts > best_ts:
                best_ts, best_price = ts, pf
        if best_price is not None:
            out[h] = best_price
    return out


def fetch_horizon_prices(sample: list[dict]) -> list[dict]:
    cache_f = CACHE / "horizon_prices.jsonl"
    cached: dict[str, dict] = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            if line:
                row = json.loads(line)
                cached[row["ticker"]] = row

    todo = [m for m in sample if m["ticker"] not in cached]
    log(f"Phase 3: {len(cached)} markets cached, {len(todo)} candlestick fetches")

    with cache_f.open("a") as out:
        for i, m in enumerate(todo):
            try:
                prices = price_at_horizons(m)
            except Exception as exc:  # noqa: BLE001 — skip and continue
                log(f"  skip {m['ticker']}: {exc}")
                prices = {}
            row = {
                "ticker": m["ticker"],
                "series_ticker": m["series_ticker"],
                "result": m["result"],
                "volume": _volume(m),
                "prices": {str(h): p for h, p in prices.items()},
            }
            cached[m["ticker"]] = row
            out.write(json.dumps(row) + "\n")
            time.sleep(RATE_SLEEP)
            if (i + 1) % 50 == 0:
                log(f"  …{i+1}/{len(todo)} markets")

    return [cached[m["ticker"]] for m in sample if m["ticker"] in cached]


# --------------------------------------------------------------------------- #
# Phase 4: calibration report
# --------------------------------------------------------------------------- #
def render_report(rows: list[dict]) -> str:
    lines: list[str] = []
    w = lines.append
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    w("=" * 78)
    w(f"  Kalshi Politics Calibration Backtest · {now}")
    w(f"  Universe: settled binary Politics markets, volume ≥ {MIN_VOLUME:.0f},")
    w(f"  top {MAX_MARKETS} by volume. Prediction = last trade price at horizon.")
    w("=" * 78)

    for h in HORIZONS_H:
        preds = [
            (r["prices"][str(h)], r["result"] == "yes")
            for r in rows
            if str(h) in r["prices"]
        ]
        label = f"{h} hours" if h < 48 else f"{h // 24} days"
        w("")
        w(f"  ── Horizon: {label} before close "
          f"({len(preds)} markets) " + "─" * 20)
        if len(preds) < 30:
            w("  Too few samples for a meaningful reliability read.")
            continue
        res: CalibrationResult = reliability_diagram(preds, n_bins=10)
        base_rate = sum(1.0 for _, o in preds if o) / len(preds)
        ref_bs = base_rate * (1 - base_rate)  # always-predict-base-rate Brier
        w(f"  Brier score: {res.brier_score:.4f}   "
          f"(uninformative baseline at base rate {base_rate:.2f}: {ref_bs:.4f})")
        w("")
        w("  bin           mean prob   actual freq    gap      n")
        w("  " + "-" * 56)
        for b in res.bins:
            gap = b.actual_freq - b.mean_prob
            w(f"  [{b.prob_low:.1f}, {b.prob_high:.1f})   "
              f"{b.mean_prob:9.3f}   {b.actual_freq:11.3f}   {gap:+.3f}   {b.n:4d}")
        w("")
        w("  Read: gap < 0 → prices in this bin were too HIGH (event happened")
        w("  less often than the market said — overpriced). gap > 0 → too low.")

    return "\n".join(lines) + "\n"


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    series = fetch_series()
    markets = fetch_settled_markets(series)
    sample = select_sample(markets)
    log(f"Phase 3: {len(sample)} markets selected "
        f"(binary, settled, volume ≥ {MIN_VOLUME:.0f})")
    rows = fetch_horizon_prices(sample)

    report = render_report(rows)
    print(report)
    out_f = OUT_DIR / f"backtest_kalshi_politics_{datetime.now(tz=timezone.utc):%Y-%m-%d}.txt"
    out_f.write_text(report)
    log(f"Report saved to {out_f}")


if __name__ == "__main__":
    main()
