#!/usr/bin/env python3
"""Calibrate the Must-Be-Won sales uplift from historical roll-down draws.

Why this parameter deserves its own script
------------------------------------------
On a roll-down draw the EV is dominated by the redistributed pool, J / N, so
the verdict turns almost entirely on N - the lines sold. N is not observed; it
is estimated from tier winner counts on RECENT draws, which are ordinary ones.
But a Must-Be-Won draw sells materially more than an ordinary draw, so applying
an ordinary-draw N to it inflates EV on exactly the ~9 draws a year the advisor
exists to find. That bug shipped: the 2026-08-08 alert reported EV +£0.038 with
break-even only 3.2% away in N (found 2026-08-06).

Identification
--------------
Roll-down draws are not flagged in the archive, so they are identified by their
signature: a roll-down redistributes the jackpot into Match 3 / Match 2, which
therefore pay above their era base rate (the mode of Match 3's per-winner price).

Lines sold per draw are recovered the same way `lottery.ev.estimate_tickets_sold`
does it - winners / P(tier) over the high-count tiers, median across tiers and
rounds.

The estimate is PAIRED: each roll-down is compared with the ordinary draws
immediately before it, not with the era average. UK Lotto sales drift down over
the decade and roll-downs are not spread evenly across it, so a cross-sectional
comparison confounds the uplift with the trend (it reads ~1.54x).

Window choice matters and is reported for several values. Too narrow (1-2 prior
draws) understates the effect, because the draws just before a roll-down sit at
rollover 3-4 and are already selling hot; ~4 draws is one rollover cycle, which
is the mix `estimate_tickets_sold` actually averages over.

Outputs a report only - it never edits ev.py. Install the figure by hand in
lottery/ev.py MBW_SALES_UPLIFT, as with the popularity weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from lottery.ev import (  # noqa: E402
    MBW_SALES_UPLIFT,
    TWO_ROUND_FIRST_DRAW,
    exact_lines_sold,
    exact_sales_baseline,
    must_be_won_after_cap,
    MBW_SALES_UPLIFT_P25,
    MBW_SALES_UPLIFT_P75,
    MBW_UPLIFT_BY_WEEKDAY,
    P_MATCH_2,
    P_MATCH_3,
    P_MATCH_4,
    TIER_MATCH_3,
)

TIERS_HISTORY_FILE = Path("data/prize_tiers_history.csv")
SALES_HISTORY_FILE = Path("data/sales_history.csv")
DRAW_POOLS_FILE = Path("data/draw_pools.csv")
TIER_PROBS = {4: P_MATCH_4, 5: P_MATCH_3, 6: P_MATCH_2}
TWO_ROUND_ERA = "2026-06-10"
BALLS_59_ERA = "2018-11-01"
BOOST_FACTOR = 1.2          # Match 3 above base * this = roll-down


def load() -> pd.DataFrame:
    if not TIERS_HISTORY_FILE.exists():
        raise SystemExit(f"{TIERS_HISTORY_FILE} not found - "
                         "run scripts/backfill_prize_tiers.py first.")
    df = pd.read_csv(TIERS_HISTORY_FILE)
    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce")
    return df


def rolldown_draws(df: pd.DataFrame) -> set:
    """Draw numbers whose Match 3 paid above the base rate for their era.

    Base is the MODE per era, not the mean: roll-downs are a minority, and the
    single-round era paid £30 (£25 before the 2018 rule change) while the
    two-round era pays £10, so one global base would misclassify a whole era.
    """
    m3 = df[df["tier"] == TIER_MATCH_3].copy()
    m3["era"] = ["two_round" if d >= pd.Timestamp(TWO_ROUND_ERA) else "single"
                 for d in m3["draw_date"]]
    base = m3.groupby("era")["prize_per_winner"].apply(lambda s: s.mode().iloc[0])
    boosted = m3[[row.prize_per_winner > base[row.era] * BOOST_FACTOR
                  for row in m3.itertuples()]]
    return set(int(d) for d in boosted["draw_number"].unique())


def implied_sales(df: pd.DataFrame) -> dict:
    """draw_number -> implied lines sold, median over tiers 4/3/2 and rounds."""
    out = {}
    for draw, g in df.groupby("draw_number"):
        vals = sorted(
            row["winners"] / TIER_PROBS[int(row["tier"])]
            for _, row in g.iterrows()
            if int(row["tier"]) in TIER_PROBS and row["winners"] > 0
        )
        if vals:
            out[int(draw)] = vals[len(vals) // 2]
    return out


def paired_ratios(sales: dict, boosted: set, window: int) -> pd.Series:
    """Each roll-down's sales over the median of the ordinary draws before it."""
    ratios, index = [], []
    for draw in sorted(boosted):
        if draw not in sales:
            continue
        prior = [sales[d] for d in range(draw - window, draw)
                 if d in sales and d not in boosted]
        if len(prior) < max(2, window // 2):
            continue
        ratios.append(sales[draw] / sorted(prior)[len(prior) // 2])
        index.append(draw)
    return pd.Series(ratios, index=index, dtype=float)


def report(df: pd.DataFrame, sales: dict, boosted: set, windows: list) -> None:
    print("=" * 70)
    print("MUST-BE-WON SALES UPLIFT")
    print("=" * 70)
    print(f"Draws in archive:  {df['draw_number'].nunique():,}")
    print(f"Roll-downs found:  {len(boosted)} (Match 3 above era base)")
    print()
    print(f"  {'window':>7} {'n':>5} {'median':>8} {'mean':>8} {'p25':>7} {'p75':>7} {'>1.0':>6}")
    for w in windows:
        r = paired_ratios(sales, boosted, w)
        if len(r) < 3:
            continue
        print(f"  {w:>7} {len(r):>5} {r.median():>8.3f} {r.mean():>8.3f} "
              f"{r.quantile(.25):>7.3f} {r.quantile(.75):>7.3f} {(r > 1).mean():>5.0%}")

    dates = df.groupby("draw_number")["draw_date"].first()
    main_window = 4
    r = paired_ratios(sales, boosted, main_window)
    era = r[[dates.get(d, pd.NaT) >= pd.Timestamp(BALLS_59_ERA) for d in r.index]]

    print(f"\n59-ball era only (from {BALLS_59_ERA}), window={main_window}:")
    print(f"  n={len(era)}  median {era.median():.3f}  "
          f"p25 {era.quantile(.25):.3f}  p75 {era.quantile(.75):.3f}")

    print("\nStability by period (window=4):")
    for lo, hi in [("2015", "2019"), ("2019", "2022"), ("2022", "2024"), ("2024", "2027")]:
        sub = r[[pd.Timestamp(lo) <= dates.get(d, pd.NaT) < pd.Timestamp(hi) for d in r.index]]
        if len(sub) > 2:
            print(f"  {lo}-{hi}: n={len(sub):>3}  median {sub.median():.3f}")

    print("\nInstalled in lottery/ev.py (mixed-baseline fallback):")
    print(f"  MBW_SALES_UPLIFT      {MBW_SALES_UPLIFT:.3f}   (calibrated {era.median():.3f})")
    print(f"  MBW_SALES_UPLIFT_P25  {MBW_SALES_UPLIFT_P25:.3f}   "
          f"(calibrated {era.quantile(.25):.3f})")
    print(f"  MBW_SALES_UPLIFT_P75  {MBW_SALES_UPLIFT_P75:.3f}   "
          f"(calibrated {era.quantile(.75):.3f})")

    day_aware_report(boosted)
    print("\nNote: 'Must-Be-Won' vs 'big jackpot' was settled 2026-08-07 with real")
    print("per-draw sales: pool size does not drive the uplift (log-pool coef -0.016).")
    print("The weekday does - hence the day-aware section above.")
    print("=" * 70)


def day_aware_report(boosted: set) -> None:
    """Per-weekday uplift in the estimator-matched definition: each roll-down's
    lines over the median of same-weekday ordinary draws among the 20 draws
    before it. Real sales (data/sales_history.csv) are the measurement of
    record here - winner-count sales carry per-draw popularity noise that the
    same-day split (half the observations) can no longer average away."""
    if not SALES_HISTORY_FILE.exists():
        print("\nDay-aware uplift: data/sales_history.csv missing - "
              "run `make sales` first.")
        return
    s = pd.read_csv(SALES_HISTORY_FILE, parse_dates=["draw_date"])
    s = s.dropna(subset=["draw_number"]).sort_values("draw_number").reset_index(drop=True)
    s["draw_number"] = s["draw_number"].astype(int)
    s["dow"] = s["draw_date"].dt.weekday
    s["is_rd"] = s["draw_number"].isin(boosted)

    print("\nDay-aware uplift (same-weekday baseline, 20-draw window, real sales):")
    print(f"  {'day':>9} {'n':>4} {'median':>8} {'p25':>7} {'p75':>7}   installed")
    for dow, label in ((5, "Saturday"), (2, "Wednesday")):
        ratios = []
        for i, row in s[s["is_rd"] & (s["dow"] == dow)].iterrows():
            win = s[s.index < i].tail(20)
            win = win[(~win["is_rd"]) & (win["dow"] == dow)]
            if len(win) >= 4:
                ratios.append(row["lines_sold"] / win["lines_sold"].median())
        if len(ratios) < 3:
            print(f"  {label:>9}: not enough observations")
            continue
        r = pd.Series(ratios)
        inst = MBW_UPLIFT_BY_WEEKDAY.get(dow)
        print(f"  {label:>9} {len(r):>4} {r.median():>8.3f} {r.quantile(.25):>7.3f} "
              f"{r.quantile(.75):>7.3f}   {inst}")


def exact_era_uplifts(pools) -> list:
    """The two-round era on exact sales, in the estimator's own definition.

    Everything above measures winner-count sales over an archive that is
    overwhelmingly single-round. That is the right sample for the fallback
    constants and the wrong one for a decision: the 7 June 2026 licence changed
    the jackpot's share of sales and the Saturday base, and Combs & Spry (2024)
    find that every redesign moves the sales response to a jackpot.

    So this section measures only draws since the redesign, and measures them
    off `data/draw_pools.csv` - `(pool - previous pool) / 8.88%`, an identity -
    rather than off winner counts, whose +/-15% per-draw noise is exactly what
    a handful of observations cannot average away.

    The baseline comes from `exact_sales_baseline`, called rather than
    reimplemented: an uplift is only installable if it was measured against the
    same baseline the estimator will multiply. Draws it declines to price (too
    few same-weekday observations before them) are reported as such instead of
    being quietly measured a different way.
    """
    exact = exact_lines_sold(pools)
    must_be_won = sorted(must_be_won_after_cap(pools))
    dates = {int(r["draw_number"]): pd.Timestamp(r["draw_date"]).date()
             for _, r in pools.iterrows()}

    rows = []
    for draw in must_be_won:
        when, measured = dates.get(draw), exact.get(draw)
        if when is None or measured is None:
            # The first draw after a jackpot is won restarts from the minimum,
            # so the difference between its pool and the previous one measures
            # the reset, not any sales.
            rows.append({"draw": draw, "date": when, "lines": None,
                         "baseline": None, "uplift": None,
                         "why": "pool reset - not priceable"})
            continue
        baseline = exact_sales_baseline(pools[pools["draw_number"] < draw], when)
        rows.append({
            "draw": draw, "date": when, "lines": measured, "baseline": baseline,
            "uplift": measured / baseline if baseline else None,
            "why": None if baseline else "too few same-weekday priors",
        })
    return rows


def exact_era_report() -> None:
    if not DRAW_POOLS_FILE.exists():
        print(f"\nExact-pool era report: {DRAW_POOLS_FILE} missing - "
              "run scripts/backfill_draw_pools.py first.")
        return
    rows = exact_era_uplifts(pd.read_csv(DRAW_POOLS_FILE))

    print(f"\nTwo-round era on EXACT sales (draws from {TWO_ROUND_FIRST_DRAW}, "
          f"identity: delta pool / 8.88%):")
    print(f"  {'draw':>6} {'date':>12} {'day':>10} {'lines':>12} "
          f"{'baseline':>12} {'uplift':>8}")
    by_day: dict = {}
    for r in rows:
        day = r["date"].strftime("%A") if r["date"] else ""
        if r["uplift"] is None:
            lines = f"{r['lines']:,}" if r["lines"] else "-"
            print(f"  {r['draw']:>6} {str(r['date'] or ''):>12} {day:>10} "
                  f"{lines:>12} {r['why']:>25}")
            continue
        by_day.setdefault(r["date"].weekday(), []).append(r["uplift"])
        print(f"  {r['draw']:>6} {r['date']!s:>12} {day:>10} "
              f"{r['lines']:>12,} {r['baseline']:>12,} {r['uplift']:>8.3f}")

    print(f"\n  {'day':>10} {'n':>4} {'median':>8}   installed")
    for dow, label in ((5, "Saturday"), (2, "Wednesday")):
        ratios = by_day.get(dow, [])
        inst = MBW_UPLIFT_BY_WEEKDAY.get(dow)
        if not ratios:
            print(f"  {label:>10} {0:>4} {'-':>8}   {inst}   no live observation")
            continue
        r = pd.Series(ratios)
        print(f"  {label:>10} {len(r):>4} {r.median():>8.3f}   {inst}"
              f"{'   n < 4: report only, do not install' if len(r) < 4 else ''}")
    print("\n  Installing a LOWER uplift raises EV on every Must-Be-Won draw, so")
    print("  the burden of proof sits on the side of the change. Wait for n >= 4,")
    print("  and for at least one observation on the day being changed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, nargs="+", default=[2, 4, 6, 8],
                        help="Prior-draw window sizes to report (default: 2 4 6 8)")
    args = parser.parse_args()

    df = load()
    boosted = rolldown_draws(df)
    if not boosted:
        raise SystemExit("No roll-down draws identified - check the archive.")
    report(df, implied_sales(df), boosted, args.windows)
    exact_era_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
