#!/usr/bin/env python3
"""Record the jackpot pool each draw actually carried, from the official feed.

Why this file exists
--------------------
Sales per draw are not published by anyone - the Gambling Commission answered a
2026 FOI with "does not collect data on Lotto ticket sales by day of the week",
and Allwyn reports quarterly totals. But the game procedures fix the split:

    "8.88% of sales for a Lotto Draw are allocated to the Jackpot"
    (Lotto Online Game Procedures, Edition 22, 7 June 2026, section 3.1)

so on any draw that inherits a rolled-over pool,

    sales = (pool(d) - pool(d - 1)) / 0.0888

is an identity, not an estimate. It reproduces Merseyworld's archive to within
about a tenner (see tests/test_draw_pools.py), which also settles what that
archive is: the same identity computed by a third party since 2003, not an
independent measurement of the operator's books.

What it replaces is the winner-count estimator - N ~ winners / P(tier) - which
carries about +/-15% of noise on a SINGLE draw, because the winner counts move
with how popular the drawn numbers happened to be. Fine under a 20-draw median,
useless for scoring one Must-Be-Won.

Why not read data/lotto_full_history.csv's `Jackpot` column
-----------------------------------------------------------
Because it is two different quantities under one name. The JSON ingest writes
the real pool (`drawResult.topPrize.prizeCents`); the older XML path writes the
previous draw's ANNOUNCED estimate, which is a forecast Allwyn publishes days
ahead and misses by up to ~4%. Draw 3192 is recorded there as £3,625,040 and
actually carried £3,492,117.23 - a 8.9% error in the sales that come out of it.
The estimate is the right number for "advertised pool vs redistributed" in
rolldown_history.py, so it stays where it is; this file keeps the actual pools
beside it, one provenance per column.

Usage:
    PYTHONPATH=. python scripts/backfill_draw_pools.py            # 3179 -> latest
    PYTHONPATH=. python scripts/backfill_draw_pools.py --from 3190
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from lottery.ev import TWO_ROUND_FIRST_DRAW  # noqa: E402

POOLS_FILE = Path("data/draw_pools.csv")
DRAW_URL = "https://api-dfe.national-lottery.co.uk/draw-game/results/6/{draw}"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# The feed serves the last ~180 days by draw number and 404s beyond that, so a
# backfill can only ever reach back six months. That is why the result is
# committed: the window moves, the file does not.
FEED_WINDOW_DAYS = 180


def fetch_pool(session: requests.Session, draw: int) -> dict | None:
    """The pool draw `draw` actually carried, or None if it is out of window."""
    try:
        resp = session.get(DRAW_URL.format(draw=draw), headers=UA, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload = resp.json()
        result = payload["drawResult"]
        cents = (result.get("topPrize") or {}).get("prizeCents")
        if cents is None:
            return None
        # How many times the pool had rolled INTO this draw. The feed sends
        # null after an outright win and 0 on a Must-Be-Won draw, so the count
        # cannot be read as "how hot is this draw" - what it identifies is the
        # NEXT one: a draw whose predecessor reached the cap of 5 is
        # Must-Be-Won (verified on 3189 -> 3190 and 3195 -> 3196).
        rollovers = (payload.get("prizeBreakdown") or {}).get("jackpotRolloverCount")
        return {
            "draw_number": draw,
            "draw_date": str(result["drawDate"])[:10],
            "pool_gbp": cents / 100.0,
            "rollover_count": rollovers,
        }
    except Exception as exc:                       # noqa: BLE001 - report and move on
        print(f"[pools] draw {draw}: {exc}")
        return None


def load_existing() -> pd.DataFrame:
    if POOLS_FILE.exists():
        return pd.read_csv(POOLS_FILE)
    return pd.DataFrame(columns=["draw_number", "draw_date", "pool_gbp",
                                 "rollover_count"])


def save(rows: pd.DataFrame) -> None:
    rows = rows.drop_duplicates(subset="draw_number", keep="last")
    rows = rows.sort_values("draw_number")
    rows["draw_number"] = rows["draw_number"].astype(int)
    # Nullable: the feed sends null after an outright win, and "0 rollovers"
    # is a different statement from "this draw did not roll".
    rows["rollover_count"] = rows["rollover_count"].astype("Int64")
    POOLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(POOLS_FILE, index=False)


def latest_collected() -> int:
    """Highest draw number we hold anywhere, so the backfill knows where to stop."""
    highest = TWO_ROUND_FIRST_DRAW
    for path, column in ((Path("data/prize_tiers.csv"), "draw_number"),
                         (Path("data/lotto_full_history.csv"), "DrawNumber")):
        if path.exists():
            highest = max(highest, int(pd.read_csv(path)[column].max()))
    return highest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="first", type=int, default=TWO_ROUND_FIRST_DRAW,
                    help="first draw to fetch (default: the first two-round draw)")
    ap.add_argument("--to", dest="last", type=int, default=None,
                    help="last draw to fetch (default: the latest collected)")
    ap.add_argument("--refetch", action="store_true",
                    help="re-read draws already on file instead of skipping them")
    args = ap.parse_args()

    last = args.last or latest_collected()
    existing = load_existing()
    have = set() if args.refetch else set(existing["draw_number"].astype(int))

    session = requests.Session()
    fetched, missing = [], []
    for draw in range(args.first, last + 1):
        if draw in have:
            continue
        row = fetch_pool(session, draw)
        if row is None:
            missing.append(draw)
        else:
            fetched.append(row)
        time.sleep(0.2)                            # the feed has no rate limit; be a guest

    if fetched:
        save(pd.concat([existing, pd.DataFrame(fetched)], ignore_index=True))
    print(f"[pools] {len(fetched)} new, {len(have)} already held, "
          f"{len(missing)} unavailable{' (outside the ~180-day window)' if missing else ''}")
    if missing:
        print(f"[pools] unavailable: {missing[0]}-{missing[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
