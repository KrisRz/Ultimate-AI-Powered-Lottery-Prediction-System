#!/usr/bin/env python3
"""Replay every roll-down this archive holds, priced under today's rules.

The figures this project quotes about Must-Be-Won draws - roughly nine a year,
fewer than half of them actually worth playing - have until now lived only as
prose in plan.md. A public page cannot make a quantitative claim backed by a
memo, so this rebuilds them from the data and the tests pin the counts.

What it does, per roll-down:

  1. Find it. `rolldown_draws` flags a draw whose Match 3 paid well above its
     era's base rate, which is the signature of a jackpot being redistributed.
  2. Classify it. A cap-driven roll-down is the automatic kind: the jackpot has
     rolled ROLLOVER_CAP times and must now be paid out. Anything else is an
     operator-scheduled special event, which arrives on its own schedule and
     cannot be forecast from the rollover count.
  3. Price it at its OWN measured sales. This is the part that matters. Every
     roll-down's EV is dominated by pool / lines-sold, so using an era average
     would flatter the busy draws and punish the quiet ones. Real per-draw
     sales come from data/sales_history.csv.

The result is a counterfactual and is labelled as one wherever it surfaces:
historical draws re-priced under the two-round rules that only began in June
2026. It answers "how often is a roll-down worth playing?" to an order of
magnitude, not "what would I have won?".

Usage:
  python scripts/rolldown_history.py            # summary table
  python scripts/rolldown_history.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from lottery.ev import (  # noqa: E402
    ROLLOVER_CAP,
    DrawConditions,
    FixedPrizes,
    best_unpopular_reference_line,
    line_ev,
)
from scripts.calibrate_mbw_uplift import rolldown_draws  # noqa: E402

DATA_DIR = Path("data")
TIERS_HISTORY = DATA_DIR / "prize_tiers_history.csv"
FULL_HISTORY = DATA_DIR / "lotto_full_history.csv"
SALES_HISTORY = DATA_DIR / "sales_history.csv"

# The two-round game began 2026-06-10; everything before it was single-round.
# Pricing the whole archive under today's rules is what makes this a
# counterfactual rather than a measurement.
ROUNDS_TODAY = 2

CAVEAT = (
    "Counterfactual: historical roll-downs re-priced under today's two-round "
    "rules, each at its own measured ticket sales. An order of magnitude, not "
    "a record of what was actually paid."
)


def rollover_streaks(full: pd.DataFrame, boosted: set) -> dict:
    """draw_number -> consecutive draws the jackpot had rolled before it.

    Reconstructed rather than read from a column, because the archive carries
    no rollover field before the JSON-feed era.

    A rollover sequence ends two ways, and both have to reset the count: a
    ticket matches six, or the draw rolls down and the pool is redistributed
    into the lower tiers. Counting only outright wins - the obvious reading of
    `JackpotWins` - misses the second, because a roll-down leaves Match 6 with
    no winners. That produced streaks of 29 against a cap of 5, which is not a
    thing that can happen.
    """
    rounds = (
        full.groupby("DrawNumber")["JackpotWins"].sum().sort_index()
        if "JackpotWins" in full else pd.Series(dtype=float)
    )
    streaks: dict[int, int] = {}
    running = 0
    for draw, wins in rounds.items():
        number = int(draw)
        streaks[number] = running
        resolved = (pd.notna(wins) and wins > 0) or number in boosted
        running = 0 if resolved else running + 1
    return streaks


def replay_rolldowns() -> list[dict]:
    """One row per detected roll-down, priced at its own conditions."""
    tiers = pd.read_csv(TIERS_HISTORY, parse_dates=["draw_date"])
    full = pd.read_csv(FULL_HISTORY)
    sales = pd.read_csv(SALES_HISTORY)

    boosted = rolldown_draws(tiers)
    streaks = rollover_streaks(full, boosted)
    lines_by_draw = dict(zip(sales["draw_number"].astype(int), sales["lines_sold"]))

    # The pool that was redistributed: the advertised jackpot for that draw.
    jackpot_by_draw = (
        full.groupby("DrawNumber")["Jackpot"].max().to_dict()
        if "Jackpot" in full else {}
    )
    dates = dict(zip(tiers["draw_number"].astype(int),
                     tiers["draw_date"].dt.date.astype(str)))

    line = best_unpopular_reference_line()
    rows: list[dict] = []

    for draw in sorted(boosted):
        pool = jackpot_by_draw.get(draw)
        tickets = lines_by_draw.get(draw)
        if not pool or not tickets or pool <= 0 or tickets <= 0:
            continue  # cannot price it honestly without both

        cond = DrawConditions(
            jackpot=float(pool),
            tickets_sold=int(tickets),
            roll_down=True,
            rounds=ROUNDS_TODAY,
            prizes=FixedPrizes(),
        )
        rows.append({
            "draw_number": draw,
            "date": dates.get(draw),
            "pool_gbp": float(pool),
            "tickets_sold": int(tickets),
            "cap_driven": streaks.get(draw, 0) >= ROLLOVER_CAP,
            "ev": line_ev(line, cond),
        })

    return rows


def summarise(rows: list[dict]) -> dict:
    """Headline figures, cap-driven only.

    The share that clears break-even is quoted for cap-driven draws alone,
    because those are the ones the forecast can see coming - a special event
    arrives when the operator decides, and cannot be planned for.
    """
    cap = [r for r in rows if r["cap_driven"]]
    evs = sorted(r["ev"] for r in cap)
    positive = [e for e in evs if e >= 0]

    def quantile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        return values[min(len(values) - 1, int(q * len(values)))]

    years = 0
    if rows:
        first = min(r["date"] for r in rows if r["date"])
        last = max(r["date"] for r in rows if r["date"])
        years = max(1e-9, (pd.Timestamp(last) - pd.Timestamp(first)).days / 365.25)

    return {
        "detected": len(rows),
        "cap_driven": len(cap),
        "special_event": len(rows) - len(cap),
        "window": [min((r["date"] for r in rows), default=None),
                   max((r["date"] for r in rows), default=None)],
        "per_year": round(len(rows) / years, 1) if years else None,
        "positive_ev": len(positive),
        "positive_ev_share": len(positive) / len(cap) if cap else None,
        "median_ev": quantile(evs, 0.5),
        "ev_quartiles": [quantile(evs, 0.25), quantile(evs, 0.75)],
        "median_pool_gbp": quantile(sorted(r["pool_gbp"] for r in cap), 0.5),
        "median_tickets": quantile(sorted(r["tickets_sold"] for r in cap), 0.5),
        "caveat": CAVEAT,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = parser.parse_args()

    rows = replay_rolldowns()
    stats = summarise(rows)

    if args.json:
        print(json.dumps({"summary": stats, "draws": rows}, indent=2, default=str))
        return 0

    print(f"Roll-downs detected : {stats['detected']}  "
          f"({stats['cap_driven']} cap-driven, {stats['special_event']} special event)")
    print(f"Window              : {stats['window'][0]} to {stats['window'][1]}"
          f"  ~{stats['per_year']}/year")
    print(f"Clear break-even    : {stats['positive_ev']} of {stats['cap_driven']} cap-driven"
          f"  = {(stats['positive_ev_share'] or 0) * 100:.0f}%")
    print(f"Median EV per line  : GBP{stats['median_ev']:+.3f}"
          f"   quartiles {stats['ev_quartiles'][0]:+.2f} / {stats['ev_quartiles'][1]:+.2f}")
    print(f"Median pool         : GBP{stats['median_pool_gbp']:,.0f}"
          f"   median lines {stats['median_tickets']:,.0f}")
    print()
    print(CAVEAT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
