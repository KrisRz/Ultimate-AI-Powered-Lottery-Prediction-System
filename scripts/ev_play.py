#!/usr/bin/env python3
"""EV-first play advisor: should you play the next draw, and with which lines?

Reads next-draw conditions (estimated jackpot, roll-down flag) from
data/prize_tiers.csv (accumulated from the official XML feed), decides
whether the draw clears the EV threshold, and builds a diversified
portfolio of unpopular lines.

Usage:
  python scripts/ev_play.py                     # advise for next draw
  python scripts/ev_play.py --lines 5 --seed 42
  python scripts/ev_play.py --jackpot 10000000 --roll-down   # what-if
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from lottery.ev import (  # noqa: E402
    DrawConditions,
    DEFAULT_TICKETS_SOLD,
    estimate_tickets_sold,
    should_play,
)
from lottery.portfolio import build_portfolio  # noqa: E402

PRIZE_TIERS_FILE = Path("data/prize_tiers.csv")
OUT_DIR = Path("outputs/predictions")


def next_draw_conditions() -> DrawConditions:
    """Best known conditions for the upcoming draw, from collected data."""
    cond = DrawConditions()
    if PRIZE_TIERS_FILE.exists():
        tiers = pd.read_csv(PRIZE_TIERS_FILE)
        last = tiers.sort_values("draw_number").iloc[-1]
        if pd.notna(last.get("next_jackpot_estimate")):
            cond.jackpot = float(last["next_jackpot_estimate"])
        cond.roll_down = bool(last.get("next_jackpot_roll_down", False))
        estimated = estimate_tickets_sold(tiers)
        if estimated:
            cond.tickets_sold = estimated
    return cond


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=int, default=5, help="Portfolio size")
    parser.add_argument("--jackpot", type=float, default=None, help="Override jackpot (per round)")
    parser.add_argument("--roll-down", action="store_true", help="Force Must-Be-Won roll-down")
    parser.add_argument("--tickets", type=int, default=None,
                        help=f"Assumed lines sold per draw (default: estimated from "
                             f"prize_tiers.csv, else {DEFAULT_TICKETS_SOLD:,})")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum EV (GBP) per line to recommend playing")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Build a portfolio even when the draw is below threshold")
    args = parser.parse_args()

    cond = next_draw_conditions()
    if args.jackpot is not None:
        cond.jackpot = args.jackpot
    if args.roll_down:
        cond.roll_down = True
    if args.tickets is not None:
        cond.tickets_sold = args.tickets

    verdict = should_play(cond, threshold=args.threshold)

    print("=" * 64)
    print("EV ADVISOR - next UK Lotto draw")
    print("=" * 64)
    print(f"Jackpot (per round):  £{cond.jackpot:,.0f}")
    print(f"Rounds per ticket:    {cond.rounds}")
    print(f"Must-Be-Won:          {'YES' if cond.roll_down else 'no'}")
    print(f"Assumed lines sold:   {cond.tickets_sold:,}")
    print(f"Best-line EV:         £{verdict['ev_best_line']:+.3f}  (threshold £{args.threshold:+.2f})")
    print("-" * 64)

    if not verdict["play"] and not args.force:
        print("VERDICT: SKIP this draw - expected loss per £2 line is above")
        print("your threshold. Playing anyway is entertainment, not investment.")
        print("(Use --force to build a portfolio regardless.)")
        print("=" * 64)
        return

    if verdict["play"]:
        print("VERDICT: conditions clear your threshold - if you play, play these:")
    else:
        print("VERDICT: below threshold (forced portfolio):")

    portfolio = build_portfolio(args.lines, cond, seed=args.seed)
    total_ev = sum(p["ev"] for p in portfolio)
    for i, p in enumerate(portfolio, 1):
        nums = " ".join(f"{n:2d}" for n in p["line"])
        print(f"  {i}. {nums}   EV £{p['ev']:+.3f}   popularity x{p['popularity_ratio']:.2f}")
    print("-" * 64)
    print(f"Portfolio: {len(portfolio)} lines, cost £{len(portfolio) * cond.ticket_price:.2f}, "
          f"total EV £{total_ev:+.2f}")
    print("=" * 64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "predictions": [p["line"] for p in portfolio],
        "metadata": {
            "method": "ev_portfolio",
            "verdict": verdict,
            "per_line": [{"line": p["line"], "ev": p["ev"],
                          "popularity_ratio": p["popularity_ratio"]} for p in portfolio],
        },
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUT_DIR / f"ev_portfolio_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(OUT_DIR / "latest.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {OUT_DIR}/ev_portfolio_{ts}.json (+ latest.json for roi_ledger)")


if __name__ == "__main__":
    main()
