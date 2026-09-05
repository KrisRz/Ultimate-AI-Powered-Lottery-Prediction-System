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
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from lottery.ev import (  # noqa: E402
    DrawConditions,
    DEFAULT_TICKETS_SOLD,
    abrams_garibaldi_screen,
    calibrate_fixed_prizes,
    default_portfolio_seed,
    estimate_tickets_sold,
    forecast_must_be_won,
    kelly_stake,
    mbw_type,
    mbw_uplift,
    should_play,
    upcoming_draw_date,
)
from lottery.portfolio import build_portfolio  # noqa: E402

PRIZE_TIERS_FILE = Path("data/prize_tiers.csv")
DRAW_POOLS_FILE = Path("data/draw_pools.csv")
OUT_DIR = Path("outputs/predictions")


def next_draw_conditions(force_roll_down: bool = False,
                         force_ordinary: bool = False,
                         now: datetime | date | None = None) -> DrawConditions:
    """Best known conditions for the upcoming draw, from collected data.

    The roll-down overrides belong here rather than in the caller because the
    sales estimate DEPENDS on the flag - a Must-Be-Won draw sells ~1.27x
    (Sat) / 1.44x (Wed) an ordinary one. Setting `cond.roll_down` after the
    fact would leave a what-if run priced at the wrong sales level, which is
    the same mistake that made the 2026-08-08 alert read +EV.

    `force_ordinary` is the mirror of `force_roll_down`: while the live feed
    flags a roll-down, every what-if inherits it, so "what would an ordinary
    draw at this jackpot be worth?" was unaskable.

    `now` pins the moment the conditions are priced at. Callers that must be
    reproducible pass one derived from the data rather than the clock - the
    site exporter does, because its output is committed and diffed in CI, and
    a wall-clock read would rewrite the file every time the draw date rolls.
    """
    cond = DrawConditions()
    # The draw being priced is the one a ticket bought NOW would enter; its
    # weekday selects the sales baseline and Must-Be-Won uplift (Saturdays
    # sell ~1.59x Wednesdays).
    cond.draw_date = upcoming_draw_date(now)
    if PRIZE_TIERS_FILE.exists():
        tiers = pd.read_csv(PRIZE_TIERS_FILE)
        if len(tiers):
            last = tiers.sort_values("draw_number").iloc[-1]
            if pd.notna(last.get("next_jackpot_estimate")):
                cond.jackpot = float(last["next_jackpot_estimate"])
            # bool(NaN) is True - a missing flag must never fake a Must-Be-Won
            flag = last.get("next_jackpot_roll_down")
            cond.roll_down = not force_ordinary and (force_roll_down or (
                pd.notna(flag) and str(flag).strip().lower() in ("true", "y", "yes", "1")))
            if pd.notna(last.get("rollover_count")):
                cond.rollover_count = int(last["rollover_count"])
            # Sales are an identity, not an estimate, wherever the pools
            # reach: (pool - previous pool) / 8.88%. Winner counts stay the
            # fallback for windows the pools do not cover.
            pools = (pd.read_csv(DRAW_POOLS_FILE)
                     if DRAW_POOLS_FILE.exists() else None)
            estimated = estimate_tickets_sold(tiers, roll_down=cond.roll_down,
                                              draw_date=cond.draw_date,
                                              pools_df=pools)
            if estimated:
                cond.tickets_sold = estimated
            # Fixed-tier prizes come from the data too - a hardcoded table is
            # how the Match 3/2 roll-down prizes once leaked into every draw.
            cond.prizes = calibrate_fixed_prizes(tiers)
    return cond


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=int, default=5, help="Portfolio size")
    parser.add_argument("--jackpot", type=float, default=None,
                        help="Override jackpot (event pool, shared across rounds)")
    parser.add_argument("--roll-down", action="store_true", help="Force Must-Be-Won roll-down")
    parser.add_argument("--ordinary", action="store_true",
                        help="Force an ordinary (non-roll-down) draw in what-if runs")
    parser.add_argument("--tickets", type=int, default=None,
                        help=f"Assumed lines sold per draw (default: estimated from "
                             f"prize_tiers.csv, else {DEFAULT_TICKETS_SOLD:,})")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum EV (GBP) per line to recommend playing")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Bankroll (GBP) the Kelly stake is sized against")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Build a portfolio even when the draw is below threshold")
    args = parser.parse_args()

    cond = next_draw_conditions(force_roll_down=args.roll_down,
                                force_ordinary=args.ordinary)
    what_if = (args.jackpot is not None or args.roll_down or args.ordinary
               or args.tickets is not None)
    if args.jackpot is not None:
        cond.jackpot = args.jackpot
    if args.tickets is not None:
        cond.tickets_sold = args.tickets

    verdict = should_play(cond, threshold=args.threshold)

    print("=" * 64)
    print("EV ADVISOR - next UK Lotto draw")
    print("=" * 64)
    print(f"Jackpot (event pool): £{cond.jackpot:,.0f}")
    print(f"Rounds per ticket:    {cond.rounds}")
    kind = mbw_type(cond.roll_down, cond.rollover_count)
    print(f"Must-Be-Won:          {f'YES ({kind})' if kind else 'no'}")
    mbw = forecast_must_be_won(cond.rollover_count)
    if not cond.roll_down:
        print(f"Rollover:             {mbw['rollover_count']} of {mbw['cap']} - "
              f"Must-Be-Won in {mbw['draws_away']} draw(s), ~{mbw['expected_date']}, "
              f"if nobody wins before")
    day = cond.draw_date.strftime("%A") if cond.draw_date else "unknown day"
    print(f"Assumed lines sold:   {cond.tickets_sold:,}"
          f"{f' ({day} Must-Be-Won uplift x{mbw_uplift(cond.draw_date)[0]})' if cond.roll_down else ''}")
    p = cond.prizes
    print(f"Fixed prizes/round:   5+B £{p.match_5_bonus:,.0f} · 5 £{p.match_5:,.0f} · "
          f"4 £{p.match_4:,.0f} · 3 £{p.match_3:,.0f} · 2 £{p.match_2:,.0f}  [{p.source}]")
    print(f"Best-line EV:         £{verdict['ev_best_line']:+.3f}  (threshold £{args.threshold:+.2f})")
    print(f"Break-even jackpot:   £{verdict['break_even_jackpot']:,.0f}"
          f"{' (roll-down)' if cond.roll_down else ''}")
    ag = abrams_garibaldi_screen(cond)
    if ag:
        # Second opinion for ordinary draws (Abrams & Garibaldi 2010). Their
        # cutoffs are sufficient conditions robust to ANY sales level, so
        # passing is much rarer than our exact break-even.
        status = ("robust +EV even if sales surge" if ag["robust_good_bet"]
                  else "any edge would rest on the sales estimate")
        print(f"A&G second opinion:   entries/jackpot {ag['n_over_j']:.2f} "
              f"(<0.2 wanted), robust cutoff £{ag['jackpot_cutoff'] / 1e6:,.0f}M "
              f"- {status}")
    # On a roll-down the verdict lives or dies on the sales estimate, so show
    # what it does across the plausible range instead of one tidy number.
    sens = verdict.get("sales_sensitivity")
    if sens:
        print(f"Across sales range:   £{sens['ev_low']:+.3f} at {sens['tickets_high']:,} lines "
              f"... £{sens['ev_high']:+.3f} at {sens['tickets_low']:,} (quartiles)")
        holds = ("YES" if sens["robust"]
                 else "NO - only the central sales estimate clears it")
        print(f"Holds across range:   {holds}")
    if verdict["play"]:
        k = kelly_stake(cond, args.bankroll)
        if k["lines_full"] >= 1:
            print(f"Kelly stake:          {k['lines_full']} lines full / "
                  f"{k['lines_half']} half-Kelly on a £{args.bankroll:,.0f} bankroll")
        else:
            # The honest MacLean-Ziemba answer: a real edge that is 81% to
            # lose a given line justifies almost nothing growth-theoretically.
            print(f"Kelly stake:          £{k['stake_full']:.2f} on a "
                  f"£{args.bankroll:,.0f} bankroll (f*={k['kelly_fraction']:.2e}) - "
                  f"the edge is real, but at this bankroll every line is an "
                  f"entertainment stake, not growth")
    print("-" * 64)

    portfolio = []
    if not verdict["play"] and not args.force:
        print("VERDICT: SKIP this draw - expected loss per £2 line is above")
        print("your threshold. Playing anyway is entertainment, not investment.")
        print("(Use --force to build a portfolio regardless.)")
        print("=" * 64)
    else:
        if verdict["play"]:
            print("VERDICT: conditions clear your threshold - if you play, play these:")
        else:
            print("VERDICT: below threshold (forced portfolio):")

        # Same default seed as the alert email: latest.json (what the ledger
        # records via --from-latest) and the mail must propose the SAME lines.
        seed = (args.seed if args.seed is not None
                else default_portfolio_seed(cond.draw_date))
        portfolio = build_portfolio(args.lines, cond, seed=seed)
        total_ev = sum(p["ev"] for p in portfolio)
        for i, p in enumerate(portfolio, 1):
            nums = " ".join(f"{n:2d}" for n in p["line"])
            print(f"  {i}. {nums}   EV £{p['ev']:+.3f}   popularity x{p['popularity_ratio']:.2f}")
        print("-" * 64)
        print(f"Portfolio: {len(portfolio)} lines, cost £{len(portfolio) * cond.ticket_price:.2f}, "
              f"total EV £{total_ev:+.2f}")
        print("=" * 64)

    # A what-if run (--jackpot / --roll-down / --tickets) must NOT touch
    # latest.json: that file is what `roi_ledger.py add --from-latest` records
    # as really played and what the dashboard shows as the live verdict.
    # Exploring "what if the jackpot were £12M" once left a PLAY portfolio
    # sitting there for a draw that is actually a SKIP.
    if what_if:
        print("(what-if run - latest.json left untouched)")
        return

    # Always persist the real verdict - the dashboard reads latest.json, and a
    # SKIP with no file would tell the user to re-run make play forever.
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
    with open(OUT_DIR / "latest.json", "w") as f:
        json.dump(payload, f, indent=2)
    if portfolio:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(OUT_DIR / f"ev_portfolio_{ts}.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved to {OUT_DIR}/ev_portfolio_{ts}.json (+ latest.json for roi_ledger)")
    else:
        print(f"Verdict saved to {OUT_DIR}/latest.json")


if __name__ == "__main__":
    main()
