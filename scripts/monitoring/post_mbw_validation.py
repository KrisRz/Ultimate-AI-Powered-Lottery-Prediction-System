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

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lottery.ev import (  # noqa: E402
    calibrate_fixed_prizes,
    estimate_tickets_sold,
    mbw_uplift,
    P_MATCH_2,
    P_MATCH_3,
    P_MATCH_4,
    TIER_MATCH_2,
    TIER_MATCH_3,
)
from scripts.monitoring.nightly_backtest import maybe_send_email  # noqa: E402

PRIZE_TIERS_FILE = Path("data/prize_tiers.csv")
VALIDATION_FILE = Path("data/mbw_validation.csv")

TIER_PROBS = {4: P_MATCH_4, 5: P_MATCH_3, 6: P_MATCH_2}


def latest_draw_was_rolldown(tiers: pd.DataFrame) -> bool:
    """The row(s) BEFORE the latest draw carry its forward-looking flag."""
    draws = sorted(tiers["draw_number"].unique())
    if len(draws) < 2:
        return False
    prev = tiers[tiers["draw_number"] == draws[-2]].iloc[-1]
    flag = prev.get("next_jackpot_roll_down")
    if pd.notna(flag) and str(flag).strip().lower() in ("true", "y", "yes", "1"):
        return True
    # JSON-era rows also mark the boosted tiers directly on the draw itself
    latest_rows = tiers[tiers["draw_number"] == draws[-1]]
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


def validate(tiers: pd.DataFrame) -> dict | None:
    """The scorecard for the latest draw, or None if it was not a roll-down."""
    if not latest_draw_was_rolldown(tiers):
        return None
    draws = sorted(tiers["draw_number"].unique())
    latest_no = draws[-1]
    rows = tiers[tiers["draw_number"] == latest_no]
    draw_date = date.fromisoformat(str(rows["draw_date"].iloc[0]))
    before = tiers[tiers["draw_number"] < latest_no]
    prev = before[before["draw_number"] == draws[-2]].iloc[-1]

    measured = measured_lines(rows)
    predicted = estimate_tickets_sold(before, roll_down=True, draw_date=draw_date)
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
    if not PRIZE_TIERS_FILE.exists():
        print("[mbw-validation] no prize_tiers.csv - nothing to do")
        return 0
    tiers = pd.read_csv(PRIZE_TIERS_FILE)
    result = validate(tiers)
    if result is None:
        print("[mbw-validation] latest draw was not a roll-down - nothing to score")
        return 0
    append_scorecard(result)
    report = format_report(result)
    print(report)
    maybe_send_email(
        f"LOTTO MBW scorecard: draw {result['draw_number']}", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
