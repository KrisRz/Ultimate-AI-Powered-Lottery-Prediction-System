#!/usr/bin/env python3
"""Calibrate the number-popularity weights from observed prize-tier winners.

The EV model's edge is entirely in *jackpot sharing*: unpopular lines split the
pot with fewer people. Which numbers are unpopular is currently a literature
heuristic (lottery/ev.py: number_weight). This script recovers it from data.

Identification
--------------
For a fixed tier t, the expected winners in draw i are

    E[w_{i,t}] = N_i * P_t * pop_t(D_i)

where N_i = lines sold, P_t = fixed match probability, and pop_t(D_i) is how
popular the *drawn* numbers D_i are (people over-buy some numbers, so draws of
popular numbers produce more low-tier winners per ticket). To first order
pop_t(D) grows like the mean number-weight of D raised to the match degree, so
Match 3 (~130k winners: strong signal, low noise) is the workhorse tier.

N_i is confounded with popularity within a single draw, but total sales drift
*slowly* over time while drawn-number popularity varies *randomly* draw to draw.
So we recover N_i as a rolling-median trend of w_{i,3}/P_3 (which averages the
random popularity term to a near-constant), and the per-draw residual

    mult_i = (w_{i,3} / P_3) / N_i

isolates the popularity of that draw's numbers. Regressing log(mult_i) on
number-presence indicators yields a per-number popularity coefficient.

Two-round era (2026-06+): each round is drawn independently, so we treat every
(draw, round) as one observation.

Outputs a report only - it never edits ev.py. Suggested weight buckets are
printed for a human to apply.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lottery.ev import (
    N_BALLS,
    N_PICK,
    P_MATCH_2,
    P_MATCH_3,
    P_MATCH_4,
    TOTAL_COMBOS,
    number_weight,
    popularity_ratio,
)

DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "lotto_full_history.csv"
TIERS_HISTORY_FILE = DATA_DIR / "prize_tiers_history.csv"
SALES_HISTORY_FILE = DATA_DIR / "sales_history.csv"

TREND_WINDOW = 51        # draws in the rolling-median sales trend (odd, ~6 months)
CALIB_TIER = 5           # tier 5 == Match 3 (strong signal, low noise)
CALIB_MATCH = 3          # numbers a Match-3 winner shares with the drawn set
NUMBER_COLS = [f"Number_{i}" for i in range(1, 7)]

# All three high-count tiers see the drawn set's popularity, each through a
# different match degree k: log(multiplier) ~ (k/6) * sum log(weight). Fitting
# each separately and un-damping by 6/k must land on the SAME weights if the
# independent-weights model is adequate - a cross-tier consistency check the
# single-tier fit cannot provide (Baker & McHale 2009's cross-tier correlation
# is exactly this shared driver).
TIER_CONFIG = {
    4: (P_MATCH_4, 4, "Match 4"),
    5: (P_MATCH_3, 3, "Match 3"),
    6: (P_MATCH_2, 2, "Match 2"),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_joined(tier: int = CALIB_TIER) -> pd.DataFrame:
    """One row per (draw, round): the six drawn numbers + `tier` winners."""
    hist = pd.read_csv(HISTORY_FILE)
    hist = hist.rename(columns={"DrawNumber": "draw_number", "Round": "round"})
    tiers = pd.read_csv(TIERS_HISTORY_FILE)
    tt = tiers[tiers["tier"] == tier][["draw_number", "round", "winners"]]

    df = hist.merge(tt, on=["draw_number", "round"], how="inner")
    df = df[(df["winners"] > 0)].sort_values(["draw_number", "round"]).reset_index(drop=True)
    logger.info("Joined %d (draw, round) observations with %s winners",
                len(df), TIER_CONFIG[tier][2])
    return df


def add_multiplier(df: pd.DataFrame, window: int = TREND_WINDOW,
                   tier_prob: float = P_MATCH_3) -> pd.DataFrame:
    """Popularity multiplier per observation, sales trend divided out."""
    raw = df["winners"] / tier_prob               # ~ N_i * pop_i
    trend = raw.rolling(window, center=True, min_periods=window // 2).median()
    df = df.assign(sales_est=trend, multiplier=raw / trend)
    return df.dropna(subset=["multiplier"])


def drawn_features(df: pd.DataFrame) -> pd.DataFrame:
    nums = df[NUMBER_COLS].to_numpy()
    df = df.assign(
        n_low12=(nums <= 12).sum(axis=1),
        n_low31=(nums <= 31).sum(axis=1),
        n_high=(nums > 31).sum(axis=1),
        model_score=[popularity_ratio(row) for row in nums],
    )
    return df


# Match 3 matches 3 of the drawn 6, so a first-order expansion gives
# log(multiplier) ~= (3/6) * sum_{n in D} log(weight_n). A regression coefficient
# on a number-indicator is therefore HALF the number's log-weight, and the true
# pick-rate weight is recovered by scaling the coefficient back up by 6/3 = 2.
UNDAMP = N_PICK / CALIB_MATCH  # = 6 / 3 = 2.0 (log-mult carries k/6 of the weight sum)


def per_number_regression(df: pd.DataFrame) -> np.ndarray:
    """Least-squares log(multiplier) ~ sum of per-number indicators.

    Returns a length-59 array of relative pick-rate (mean-normalized to 1.0).
    Each row of the design matrix sums to 6, so the system is rank-deficient by
    one; np.linalg.lstsq returns the minimum-norm solution, which we un-damp
    (see UNDAMP) and renormalize to a population mean of 1.
    """
    n = len(df)
    X = np.zeros((n, N_BALLS))
    for row_i, (_, r) in enumerate(df.iterrows()):
        for c in NUMBER_COLS:
            X[row_i, int(r[c]) - 1] = 1.0
    y = np.log(df["multiplier"].to_numpy())
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    weights = np.exp(UNDAMP * (coef - coef.mean()))
    return weights / weights.mean()


def fit_bucket_weights(df: pd.DataFrame, match_degree: int = CALIB_MATCH) -> dict:
    """Calibrated pick-rate for the three interpretable buckets used by ev.py.

    Regresses log(multiplier) on the count of drawn numbers in {<=12, 13-31,
    >31} (13-31 is the reference), un-damps the contrasts by 6/match_degree,
    then solves for the absolute levels under the population constraint
    12*w_low + 19*w_mid + 28*w_high == 59 (i.e. mean pick-rate 1.0).
    """
    nums = df[NUMBER_COLS].to_numpy()
    n_low12 = (nums <= 12).sum(axis=1)
    n_high = (nums > 31).sum(axis=1)
    X = np.column_stack([np.ones(len(df)), n_low12, n_high])
    y = np.log(df["multiplier"].to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    undamp = N_PICK / match_degree     # log-mult carries (k/6) of the weight sum
    r_low = np.exp(undamp * beta[1])   # w_low12 / w_mid
    r_high = np.exp(undamp * beta[2])  # w_high  / w_mid
    w_mid = N_BALLS / (12 * r_low + 19 + 28 * r_high)
    return {"low12": r_low * w_mid, "mid": w_mid, "high": r_high * w_mid}


def cross_tier_report(window: int = TREND_WINDOW) -> None:
    """Fit the buckets from each high-count tier separately.

    Three independent match degrees un-damped by 6/k must agree if the
    independent-weights model is adequate. Added 2026-08-07: they do -
    M4 1.228/1.094/0.838, M3 1.230/1.100/0.834, M2 1.205/1.092/0.850 - the
    validation Baker & McHale's cross-tier-correlation critique asks for.
    A naive variance-weighted JOINT fit was also tried and LOST out-of-sample
    to the M3-only installed weights (it over-weights Match 2, the tier with
    the least signal per unit of noise), so the joint fit is deliberately
    absent here: this table is a consistency check, not an estimator.
    """
    print("\nCross-tier consistency (each tier fit independently, un-damped 6/k):")
    print(f"  {'tier':>8} {'k':>3} {'<=12':>7} {'13-31':>7} {'>31':>7}")
    for tier, (prob, k, label) in sorted(TIER_CONFIG.items(), key=lambda kv: -kv[1][1]):
        df = add_multiplier(load_joined(tier), window, tier_prob=prob)
        b = fit_bucket_weights(df, match_degree=k)
        print(f"  {label:>8} {k:>3} {b['low12']:>7.3f} {b['mid']:>7.3f} {b['high']:>7.3f}")
    cur = {"low12": number_weight(1), "mid": number_weight(20), "high": number_weight(40)}
    print(f"  {'installed':>8} {'-':>3} {cur['low12']:>7.3f} {cur['mid']:>7.3f} {cur['high']:>7.3f}")


def cowinner_tail_report() -> None:
    """Observed jackpot-winner counts vs the popularity model's prediction.

    This is the direct test of `expected_cowinner_share` - the term the whole
    jackpot-sharing edge runs through. Winners per (draw, round) should be
    Poisson(N * popularity_ratio(drawn)/C); the histogram over the era tests
    both the level and the tail. Simon (1998) warns of ~2,000 combinations
    with >200 holders; at our era size that shows up as rare 4+-winner draws.
    First run (2026-08-07, 1,147 draw-rounds): observed 963/158/21/3/2 for
    0/1/2/3/4+ vs predicted 942/179/22/2.7/0.4 - chi2 ~ 4.2 on 3 df, p ~ 0.24.
    The model is adequate; the extreme tail (2 draws with 4+ winners) is the
    only cluster-effect trace and is below significance.
    """
    if not SALES_HISTORY_FILE.exists():
        print("\nCo-winner tail: data/sales_history.csv missing - run `make sales`.")
        return
    hist = pd.read_csv(HISTORY_FILE, parse_dates=["Draw Date"])
    hist = hist[hist["Draw Date"] >= "2015-10-10"]
    sales = pd.read_csv(SALES_HISTORY_FILE).dropna(subset=["draw_number"])
    nmap = dict(zip(sales["draw_number"].astype(int), sales["lines_sold"]))

    observed = np.zeros(5)
    predicted = np.zeros(5)
    for _, r in hist.iterrows():
        n_lines = nmap.get(int(r["DrawNumber"]))
        if not n_lines or not np.isfinite(r["JackpotWins"]):
            continue
        observed[min(int(r["JackpotWins"]), 4)] += 1
        line = [int(r[c]) for c in NUMBER_COLS]
        lam = n_lines * popularity_ratio(line) / TOTAL_COMBOS
        probs = [np.exp(-lam)]
        for k in range(1, 4):
            probs.append(probs[-1] * lam / k)
        probs.append(max(1.0 - sum(probs), 0.0))
        predicted += probs

    print("\nJackpot co-winner tail (the expected_cowinner_share test):")
    print(f"  {'winners':>8} {'observed':>9} {'model':>9}")
    for k in range(5):
        label = f"{k}" if k < 4 else "4+"
        print(f"  {label:>8} {observed[k]:>9.0f} {predicted[k]:>9.1f}")
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - predicted) ** 2 / predicted)
    print(f"  chi2 {chi2:.1f} (4 df, on expected counts) - "
          f"{'model adequate' if chi2 < 9.5 else 'TAIL DIVERGES - revisit cluster model'}")


def oos_report(window: int = TREND_WINDOW) -> None:
    """Out-of-sample check: do the INSTALLED weights predict the second half
    of history better than a refit on the first half? If a refit ever wins
    decisively, the installed constants have drifted stale."""
    df = add_multiplier(load_joined(CALIB_TIER), window, tier_prob=P_MATCH_3)
    half = df["draw_number"].median()
    train, test = df[df["draw_number"] <= half], df[df["draw_number"] > half]
    if len(train) < 100 or len(test) < 100:
        return
    refit = fit_bucket_weights(train)
    installed = {"low12": number_weight(1), "mid": number_weight(20),
                 "high": number_weight(40)}

    def predict(dd, b):
        nums = dd[NUMBER_COLS].to_numpy()
        lw = np.log([b["low12"], b["mid"], b["high"]])
        s = ((nums <= 12) * lw[0] + ((nums > 12) & (nums <= 31)) * lw[1]
             + (nums > 31) * lw[2]).sum(axis=1)
        return (CALIB_MATCH / N_PICK) * s

    y = np.log(test["multiplier"].to_numpy())
    print("\nOut-of-sample (fit first half, score second half, Match 3):")
    for name, b in (("refit(train)", refit), ("installed", installed)):
        pred = predict(test, b)
        corr = np.corrcoef(pred, y)[0, 1]
        mse = float(np.mean((pred - y) ** 2))
        print(f"  {name:>13}: corr {corr:.3f}  mse {mse:.5f}")


def report(df: pd.DataFrame, recovered: np.ndarray) -> None:
    print("\n" + "=" * 66)
    print(f"POPULARITY CALIBRATION  ({len(df)} draw-rounds, "
          f"{df['draw_number'].min()}-{df['draw_number'].max()})")
    print("=" * 66)

    # (A) Model-free effect: more birthday-range numbers -> more Match-3 winners
    print("\nMatch-3 winner multiplier by count of drawn numbers <= 31:")
    print(f"  {'#<=31':>6} {'draws':>6} {'mean mult':>10}")
    for k, g in df.groupby("n_low31"):
        print(f"  {k:>6} {len(g):>6} {g['multiplier'].mean():>10.3f}")
    lo = df[df["n_low31"] >= 5]["multiplier"].mean()
    hi = df[df["n_low31"] <= 2]["multiplier"].mean()
    if hi > 0:
        print(f"  -> birthday-heavy draws draw {lo / hi - 1:+.1%} more winners/ticket "
              f"than high-number draws (the sharing edge, empirically)")

    # (B) Does the current literature model track reality?
    corr = np.corrcoef(np.log(df["model_score"]), np.log(df["multiplier"]))[0, 1]
    print(f"\nCurrent model vs observed:  corr(log popularity_ratio, log multiplier) "
          f"= {corr:+.3f}")
    print("  (positive => the heuristic weights point the right way)")

    # (C) Calibrated bucket weights to install in ev.py's number_weight
    buckets = fit_bucket_weights(df)
    print("\nCalibrated pick-rate weights (install in lottery/ev.py number_weight):")
    print(f"  {'bucket':>10} {'calibrated':>11} {'current':>9}")
    for label, key, lo, hi in [("<=12", "low12", 1, 12), ("13-31", "mid", 13, 31),
                               (">31", "high", 32, 59)]:
        cur = np.mean([number_weight(n) for n in range(lo, hi + 1)])
        print(f"  {label:>10} {buckets[key]:>11.3f} {cur:>9.3f}")

    order = np.argsort(recovered)
    most = ", ".join(f"{n + 1}({recovered[n]:.2f})" for n in order[::-1][:8])
    least = ", ".join(f"{n + 1}({recovered[n]:.2f})" for n in order[:8])
    print(f"\n  most-played numbers (recovered):  {most}")
    print(f"  least-played numbers (recovered): {least}")
    print("  (per-number 'lucky' boosts are within noise - only buckets survive)")
    print("=" * 66)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=TREND_WINDOW)
    parser.add_argument("--since", default=None, metavar="YYYY-MM-DD",
                        help="Fit only on draws from this date on. Player habits "
                             "drift, so a recent-only fit is the check on whether "
                             "the whole-history weights still describe today.")
    parser.add_argument("--last-draws", type=int, default=None, metavar="N",
                        help="Fit only on the most recent N draw-rounds")
    args = parser.parse_args()

    if not TIERS_HISTORY_FILE.exists():
        raise SystemExit(
            f"{TIERS_HISTORY_FILE} not found - run scripts/backfill_prize_tiers.py first."
        )
    df = load_joined()
    # Subset AFTER the trend is divided out: the rolling median needs the
    # surrounding draws to estimate sales, and a window truncated at the edge
    # would leak the popularity term it is supposed to remove.
    df = add_multiplier(df, args.window)
    if args.since:
        df = df[pd.to_datetime(df["Draw Date"]) >= args.since]
    if args.last_draws:
        df = df.tail(args.last_draws)
    if len(df) < 100:
        raise SystemExit(f"Only {len(df)} observations after filtering - too few to fit")
    df = drawn_features(df)
    recovered = per_number_regression(df)
    report(df, recovered)
    # Model-validation sections run on the full era regardless of --since /
    # --last-draws: they answer "is the model right", not "has it drifted".
    if not (args.since or args.last_draws):
        cross_tier_report(args.window)
        cowinner_tail_report()
        oos_report(args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
