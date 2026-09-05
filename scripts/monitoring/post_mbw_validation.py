#!/usr/bin/env python3
"""After a Must-Be-Won draw: score the model's forecast against what happened.

Runs at the end of every collection; exits quietly unless the just-collected
draw was a roll-down (~9 a year). When it was, this answers the two questions
plan.md says to ask after every MBW - automatically, because "compare after
each one" is exactly the kind of milestone nobody remembers in November:

1. SALES: measured N (winner counts of the draw itself) vs the N the model
   would have forecast before it (same-weekday baseline x day uplift). The
   1.27/1.44 uplift constants were measured on archive estimates; this file
   accumulates their live scorecard (data/mbw_validation.csv), and at 3-4
   observations `calibrate_mbw_uplift.py` is worth re-running.

2. POOL: the advertised jackpot vs the sum actually redistributed into the
   boosted tiers. The 2026-08-07 archive study found redistribution running
   a systematic ~9% BELOW the recorded pools; if that holds on live data,
   the J/N term is ~9% optimistic and cond.jackpot deserves a haircut.

Report is printed (lands in the workflow log) and emailed when SMTP is
configured - an MBW is rare enough that a scorecard email is signal, not spam.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lottery.ev import (  # noqa: E402
    calibrate_fixed_prizes,
    estimate_tickets_sold,
    exact_lines_sold,
    mbw_uplift,
    P_MATCH_2,
    P_MATCH_3,
    P_MATCH_4,
    TIER_MATCH_2,
    TIER_MATCH_3,
)

PRIZE_TIERS_FILE = Path("data/prize_tiers.csv")
DRAW_POOLS_FILE = Path("data/draw_pools.csv")
VALIDATION_FILE = Path("data/mbw_validation.csv")

TIER_PROBS = {4: P_MATCH_4, 5: P_MATCH_3, 6: P_MATCH_2}


def draw_was_must_be_won(tiers: pd.DataFrame,
                         draw_number: int | None = None) -> bool:
    """Was this draw Must-Be-Won? The row(s) BEFORE it carry the forward flag.

    Named for what it tests. It was `latest_draw_was_rolldown`, which was only
    ever true by coincidence: a Must-Be-Won draw rolls down when nobody wins
    (~73% of them) and pays out normally when somebody does. Both are worth
    scoring - draw 3196 was won outright and is the model's only live
    Saturday sales measurement.
    """
    draws = sorted(tiers["draw_number"].unique())
    if len(draws) < 2:
        return False
    target = int(draw_number) if draw_number is not None else int(draws[-1])
    if target not in draws or draws.index(target) == 0:
        return False
    prev = tiers[tiers["draw_number"] == draws[draws.index(target) - 1]].iloc[-1]
    flag = prev.get("next_jackpot_roll_down")
    if pd.notna(flag) and str(flag).strip().lower() in ("true", "y", "yes", "1"):
        return True
    # JSON-era rows also mark the boosted tiers directly on the draw itself
    latest_rows = tiers[tiers["draw_number"] == target]
    if "tier_roll_down" in latest_rows.columns:
        return bool(latest_rows["tier_roll_down"].fillna(False).any())
    return False


def measured_lines(rows: pd.DataFrame) -> int | None:
    """Median winners/P over the high-count tiers of ONE draw, both rounds."""
    obs = sorted(
        r["winners"] / TIER_PROBS[int(r["tier"])]
        for _, r in rows.iterrows()
        if int(r["tier"]) in TIER_PROBS and r["winners"] > 0
    )
    return int(obs[len(obs) // 2]) if obs else None


def redistributed_sum(rows: pd.DataFrame, prizes) -> float | None:
    """GBP actually paid ABOVE base rates in Match 3 / Match 2, both rounds.

    Prefers the JSON-era per-winner column; falls back to prize_total. Base
    rates come from calibrate_fixed_prizes on the PRECEDING draws, so a
    roll-down cannot contaminate its own baseline.
    """
    base = {TIER_MATCH_3: prizes.match_3, TIER_MATCH_2: prizes.match_2}
    total = 0.0
    seen = False
    for _, r in rows[rows["tier"].isin(base)].iterrows():
        if r["winners"] <= 0:
            continue
        if pd.notna(r.get("prize_per_winner")) and r.get("prize_per_winner", 0) > 0:
            boost = float(r["prize_per_winner"]) - base[int(r["tier"])]
        elif r.get("prize_total", 0) > 0:
            boost = float(r["prize_total"]) / r["winners"] - base[int(r["tier"])]
        else:
            continue
        seen = True
        total += max(boost, 0.0) * r["winners"]
    return total if seen else None


def validate(tiers: pd.DataFrame, pools: pd.DataFrame | None = None,
             draw_number: int | None = None) -> dict | None:
    """The scorecard for a draw, or None if it was not Must-Be-Won.

    Defaults to the latest collected draw, which is how the collector calls it.
    `draw_number` re-scores an earlier one, which is how a correction gets made
    from data rather than by editing the CSV by hand.
    """
    if not draw_was_must_be_won(tiers, draw_number):
        return None
    draws = sorted(tiers["draw_number"].unique())
    latest_no = int(draw_number) if draw_number is not None else int(draws[-1])
    previous_no = draws[draws.index(latest_no) - 1]
    rows = tiers[tiers["draw_number"] == latest_no]
    draw_date = date.fromisoformat(str(rows["draw_date"].iloc[0]))
    before = tiers[tiers["draw_number"] < latest_no]
    prev = before[before["draw_number"] == previous_no].iloc[-1]

    # What the draw sold is an identity where the pools reach it: winner counts
    # measure the same thing with +/-15% of noise, and a scorecard is a
    # single-draw measurement - exactly the case that noise ruins. Draw 3196
    # read 10.92m by winner counts against an exact 9.46m, turning a +2.6%
    # forecast error into a reported -11.1% and a 1.04 uplift into 1.43.
    exact = exact_lines_sold(pools)
    measured = exact.get(latest_no)
    lines_source = "pool identity"
    if measured is None:
        measured = measured_lines(rows)
        lines_source = "winner counts"
    # The forecast is reconstructed with the model as it stands TODAY, on the
    # data that existed before the draw - which is why re-scoring an old draw
    # can move its row: it is scoring the current model, not the one that ran.
    before_pools = (pools[pools["draw_number"] < latest_no]
                    if pools is not None else None)
    predicted = estimate_tickets_sold(before, roll_down=True, draw_date=draw_date,
                                      pools_df=before_pools)
    advertised = (float(prev["next_jackpot_estimate"])
                  if pd.notna(prev.get("next_jackpot_estimate")) else None)
    redistributed = redistributed_sum(rows, calibrate_fixed_prizes(before))
    jackpot_rows = rows[rows["tier"] == 1]
    jackpot_winners = int(jackpot_rows["winners"].sum()) if len(jackpot_rows) else None

    up_installed = mbw_uplift(draw_date)[0]
    baseline = predicted / up_installed if predicted else None
    return {
        "draw_number": int(latest_no),
        "draw_date": draw_date.isoformat(),
        "measured_lines": measured,
        "lines_source": lines_source,
        "predicted_lines": predicted,
        "n_error": (predicted - measured) / measured
        if measured and predicted else None,
        "uplift_installed": up_installed,
        "uplift_measured": measured / baseline if measured and baseline else None,
        "advertised_pool": advertised,
        "jackpot_winners": jackpot_winners,
        "redistributed": redistributed,
        # `is not None`, not truthiness: a measured zero means the jackpot was
        # won outright and nothing rolled down, which is a complete answer.
        # Treating it as missing data lost the one fact that explains the row.
        "pool_ratio": redistributed / advertised
        if redistributed is not None and advertised else None,
    }


def append_scorecard(result: dict, path: Path = VALIDATION_FILE) -> None:
    row = pd.DataFrame([result])
    if path.exists():
        old = pd.read_csv(path)
        row = pd.concat([old, row], ignore_index=True)
        row = row.drop_duplicates(subset="draw_number", keep="last")
    row.sort_values("draw_number").to_csv(path, index=False)


def _pool_line(r: dict) -> str:
    """What happened to the pool - and there are three answers, not two.

    A roll-down that paid out; a jackpot won outright so nothing rolled down;
    or tier data that could not be read. The middle case used to report as
    "redistribution data incomplete", which reads like a failure when it is
    in fact the most informative outcome the scorecard can carry - it is the
    reason the roll-down never happened.
    """
    def pct(x):
        return f"{x:+.1%}" if x is not None else "n/a"

    advertised = r.get("advertised_pool")
    redistributed = r.get("redistributed")

    if redistributed is None:
        return "Pool:         tier data unavailable - could not measure"

    if not redistributed:
        won = r.get("jackpot_winners")
        who = (f"{won} ticket{'' if won == 1 else 's'} matched six"
               if won else "the jackpot was claimed")
        pool = f"£{advertised:,.0f} " if advertised else ""
        return (f"Pool:         nothing rolled down - {who}, so the {pool}"
                "pool was paid out as a jackpot")

    return (f"Pool:         advertised £{advertised:,.0f}, actually "
            f"redistributed £{redistributed:,.0f} "
            f"({pct((r.get('pool_ratio') or 1) - 1)})")


def format_report(r: dict) -> str:
    def pct(x):
        return f"{x:+.1%}" if x is not None else "n/a"

    lines = [
        f"Must-Be-Won draw {r['draw_number']} ({r['draw_date']}) - model scorecard",
        "",
        f"Lines sold:   measured {r['measured_lines']:,} vs forecast "
        f"{r['predicted_lines']:,}  (forecast error {pct(r['n_error'])})"
        f"  [{r.get('lines_source', 'winner counts')}]"
        if r["measured_lines"] and r["predicted_lines"] else
        "Lines sold:   insufficient data",
        f"Sales uplift: measured x{r['uplift_measured']:.3f} vs installed "
        f"x{r['uplift_installed']:.2f}"
        if r["uplift_measured"] else "Sales uplift: n/a",
        _pool_line(r),
        "",
        f"Scorecard history: {VALIDATION_FILE} - at 3-4 live entries, re-run "
        f"scripts/calibrate_mbw_uplift.py and reconsider the constants.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draw", type=int, default=None,
                    help="re-score an earlier draw instead of the latest one")
    args = ap.parse_args()

    if not PRIZE_TIERS_FILE.exists():
        print("[mbw-validation] no prize_tiers.csv - nothing to do")
        return 0
    tiers = pd.read_csv(PRIZE_TIERS_FILE)
    pools = pd.read_csv(DRAW_POOLS_FILE) if DRAW_POOLS_FILE.exists() else None
    result = validate(tiers, pools, args.draw)
    if result is None:
        which = f"draw {args.draw}" if args.draw else "latest draw"
        print(f"[mbw-validation] {which} was not Must-Be-Won - nothing to score")
        return 0
    append_scorecard(result)
    report = format_report(result)
    print(report)
    # Deliberately not emailed. The inbox is reserved for a PLAY verdict, so
    # that a message arriving always means "act". The scorecard is a
    # post-mortem of a draw already settled - it belongs in
    # data/mbw_validation.csv, which it is, and in the workflow log above.
    return 0


if __name__ == "__main__":
    sys.exit(main())
