"""Re-cut the Kalshi calibration backtest by close-timing regime.

Markets that "close early if the event occurs" bias a horizon-before-close
backtest: sampling 24h before close catches a mid-range price that is about
to resolve YES by construction, inflating actual_freq in mid bins. Markets
that ran to (or near) their scheduled expiration have no such conditioning.

This pass fetches full details for every market already sampled by
backtest_kalshi_calibration.py, classifies each as:

    scheduled — closed within 7 days of expected_expiration_time
    early     — closed more than 7 days before expected_expiration_time

and recomputes the reliability diagram for each cut. The honest signal is
the SCHEDULED cut; comparing it to the EARLY cut shows how much of the
mid-bin gap is the early-close artifact.

Usage:  uv run python scripts/backtest_recut.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fve.tracking.calibration import reliability_diagram

BASE = "https://external-api.kalshi.com/trade-api/v2"
CACHE = Path(__file__).resolve().parent / "cache"
OUT_DIR = Path(__file__).resolve().parent / "output"
HORIZONS_H = (24, 168)
RATE_SLEEP = 0.12
SCHEDULE_TOL_S = 7 * 86400


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


def parse_ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def fetch_details(tickers: list[str]) -> dict[str, dict]:
    cache_f = CACHE / "market_details.jsonl"
    cached: dict[str, dict] = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            if line:
                row = json.loads(line)
                cached[row["ticker"]] = row

    todo = [t for t in tickers if t not in cached]
    print(f"{len(cached)} details cached, {len(todo)} to fetch", file=sys.stderr, flush=True)
    with cache_f.open("a") as out:
        for i, t in enumerate(todo):
            try:
                m = get(f"/markets/{t}").get("market", {})
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {t}: {exc}", file=sys.stderr, flush=True)
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
                print(f"  …{i+1}/{len(todo)}", file=sys.stderr, flush=True)
    return cached


def classify(detail: dict) -> str:
    ct, ee = detail.get("close_time"), detail.get("expected_expiration_time")
    if not ct or not ee:
        return "unknown"
    return "scheduled" if abs(parse_ts(ee) - parse_ts(ct)) <= SCHEDULE_TOL_S else "early"


def render(rows: list[dict], details: dict[str, dict]) -> str:
    lines: list[str] = []
    w = lines.append
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    w("=" * 78)
    w(f"  Kalshi Politics Calibration — close-timing re-cut · {now}")
    w("  scheduled = closed within 7d of expected expiration (no early-close")
    w("  conditioning); early = closed >7d ahead (price often reacts to the")
    w("  very event that closes the market — mid bins biased toward YES).")
    w("=" * 78)

    for h in HORIZONS_H:
        label = f"{h} hours" if h < 48 else f"{h // 24} days"
        for cut in ("scheduled", "early"):
            preds = []
            for r in rows:
                if str(h) not in r["prices"]:
                    continue
                d = details.get(r["ticker"])
                if d is None or classify(d) != cut:
                    continue
                preds.append((r["prices"][str(h)], r["result"] == "yes"))
            w("")
            w(f"  ── {label} before close · {cut.upper()} cut ({len(preds)} markets) " + "─" * 10)
            if len(preds) < 30:
                w("  Too few samples for a reliable read.")
                continue
            res = reliability_diagram(preds, n_bins=10)
            base = sum(1.0 for _, o in preds if o) / len(preds)
            w(f"  Brier {res.brier_score:.4f}  (baseline {base*(1-base):.4f} at base rate {base:.2f})")
            w("  bin           mean prob   actual freq    gap      n")
            w("  " + "-" * 56)
            for b in res.bins:
                w(f"  [{b.prob_low:.1f}, {b.prob_high:.1f})   "
                  f"{b.mean_prob:9.3f}   {b.actual_freq:11.3f}   "
                  f"{b.actual_freq - b.mean_prob:+.3f}   {b.n:4d}")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = [
        json.loads(line)
        for line in (CACHE / "horizon_prices.jsonl").read_text().splitlines()
        if line
    ]
    # dedupe (resumable JSONL may repeat)
    rows = list({r["ticker"]: r for r in rows}.values())
    details = fetch_details([r["ticker"] for r in rows])

    report = render(rows, details)
    print(report)
    out_f = OUT_DIR / f"backtest_kalshi_recut_{datetime.now(tz=timezone.utc):%Y-%m-%d}.txt"
    out_f.write_text(report)
    print(f"Report saved to {out_f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
