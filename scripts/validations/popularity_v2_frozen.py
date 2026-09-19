#!/usr/bin/env python3
"""FROZEN specification of the popularity challenger - popularity-model-v2.

This file is a pre-registration, not a library. It exists so that the final
out-of-sample test cannot be run against a quietly different model than the
one that was chosen. Everything the challenger is - its functional form, its
knots, the damping, the normalisation, the estimator, the training rows and
the fitted coefficients themselves - is written down here BEFORE the test and
pinned by tests/test_popularity_v2_frozen.py.

    Frozen 2026-09-19, on main at 17746e0, before any holdout was scored.

WHY THE COEFFICIENTS AND NOT JUST THE RECIPE
--------------------------------------------
A frozen recipe still leaves a refit free to move: a changed data file, a
different pandas version, one more draw in the training window, and the
"same" specification predicts something else. So the fit is done once, here,
on material that was already fully used, and the NUMBERS are frozen. The
final test then adds no fitting at all - it applies two fixed vectors to rows
neither of them has seen. `--verify` re-derives them from the archive and
fails if anything has drifted.

WHAT IS BEING COMPARED
----------------------
Incumbent  `3-bucket`   log(mult) ~ 1 + #{n<=12} + #{n>31}
Challenger `smooth`     log(mult) ~ 1 + SUM n + SUM max(n-12,0) + SUM max(n-31,0)

Both are OLS on the same rows, the same response, the same estimator. The
incumbent's refit lands on 1.230 / 1.100 / 0.834 against the 1.23 / 1.10 /
0.83 installed in lottery/ev.py since 2026-07-25, so this is genuinely the
production model against a challenger, not a straw man.

NO TUNED HYPERPARAMETERS
------------------------
The challenger has none to tune. Its knots are 12 and 31 because those are
the calendar boundaries the INCUMBENT already uses (months 1-12, days 1-31,
lottery/ev.py number_weight) - they were not searched over, and no other knot
set was ever fitted. Its four coefficients are OLS. The damping 6/3 and the
normalisation to population mean 1.0 are the calibration's, unchanged. That
matters for reading the final test: there is no researcher degree of freedom
left to blame a win or a loss on.

Run:
    PYTHONPATH=. python scripts/validations/popularity_v2_frozen.py --verify
    PYTHONPATH=. python scripts/validations/popularity_v2_frozen.py --power
"""

from __future__ import annotations

import argparse

import numpy as np

# 59 balls, 6 picked. Imported rather than repeated so a rule change breaks
# loudly here instead of silently producing a wrong frozen curve.
from lottery.ev import N_BALLS, N_PICK

FROZEN_AT_COMMIT = "17746e0"
FROZEN_ON = "2026-09-19"

# --- the estimator, unchanged from scripts/calibrate_popularity.py ----------

CALIB_TIER = 5                  # tier 5 == Match 3: ~130k winners, best S/N
CALIB_MATCH = 3                 # a Match-3 winner shares 3 of the drawn 6
UNDAMP = N_PICK / CALIB_MATCH   # = 2.0; log-mult carries k/6 of the weight sum
TREND_WINDOW = 51               # rolling-median sales trend, centred, ~6 months
KNOTS = (12, 31)                # months, days - the incumbent's boundaries

# --- training material (fully used before the freeze; nothing sacred here) ---

TRAIN_FIRST_DRAW = 2066         # first draw with tier data in the 59-ball era
TRAIN_LAST_DRAW = 3195          # last draw in data/prize_tiers_history.csv
TRAIN_OBS = 1147                # (draw, round) rows with Match-3 winners > 0
TRAIN_SOURCES = ("data/lotto_full_history.csv", "data/prize_tiers_history.csv")

# --- the frozen fits --------------------------------------------------------
# OLS (numpy.linalg.lstsq, rcond=None) of log(multiplier) on the design below,
# over all TRAIN_OBS rows. Intercept first.

SMOOTH_BETA = (
    0.5639857369441716,     # intercept
    0.0011457921095295112,  # SUM n
    -0.005699315922191074,  # SUM max(n - 12, 0)
    -0.002632293334837438,  # SUM max(n - 31, 0)
)

BUCKET_BETA = (
    0.335817539808749,      # intercept
    0.05605431387783492,    # #{n <= 12}
    -0.13848899029529033,   # #{n > 31}
)

# The same two fits on the SECOND HALF only (draws 2639-3195, 574 rows) -
# the material the specification contest used as its test set. These are the
# coefficients for the backward test: fit on what has already been scored,
# predict the first half, which never ranked a model. Scoring the first half
# with the full-archive fits above would be leakage, since those fits contain
# it.
SMOOTH_BETA_SECOND_HALF = (
    0.5006408055753555,
    0.0032508417971328614,
    -0.008479414514354045,
    -0.0024895408039364115,
)
BUCKET_BETA_SECOND_HALF = (
    0.40060461646945755,
    0.043848960488656344,
    -0.1531561009596047,
)

# The challenger as a per-number pick-rate curve: exp of the un-damped linear
# predictor, centred, then scaled to population mean 1.0 (59 numbers summing
# to 59). This is the vector that would be installed in lottery/ev.py.
SMOOTH_WEIGHTS = (
    1.187001909917, 1.18972514383, 1.192454625419, 1.195190369016,
    1.19793238899, 1.200680699738, 1.203435315693, 1.206196251321,
    1.208963521121, 1.211737139624, 1.214517121395, 1.217303481034,
    1.206267767914, 1.195332101303, 1.184495574209, 1.173757287863,
    1.163116351644, 1.152571883003, 1.142123007394, 1.131768858199,
    1.121508576657, 1.111341311792, 1.10126622034, 1.091282466686,
    1.081389222788, 1.071585668112, 1.061870989561, 1.052244381411,
    1.042705045242, 1.033252189871, 1.02388503129, 1.009275363919,
    0.994874159778, 0.980678444335, 0.966685285502, 0.952891793028,
    0.939295117902, 0.925892451767, 0.912681026335, 0.899658112821,
    0.886821021376, 0.874167100532, 0.861693736653, 0.849398353398,
    0.837278411189, 0.825331406681, 0.813554872252, 0.801946375489,
    0.790503518686, 0.779223938352, 0.768105304717, 0.757145321257,
    0.746341724216, 0.735692282137, 0.725194795407, 0.714847095796,
    0.704647046013, 0.694592539265, 0.68468149882,
)

# The incumbent, both as installed and as refitted on identical rows.
INSTALLED_WEIGHTS = (1.23, 1.10, 0.83)
BUCKET_WEIGHTS_REFIT = (1.230194087518206, 1.0997284920299815,
                        0.8336724857575674)

# --- the holdout ------------------------------------------------------------
# Draws 3196+ have tier data ONLY in the collector's data/prize_tiers.csv.
# data/prize_tiers_history.csv stops at 3195, so load_joined() has never
# returned them: they trained nothing, scored nothing, and took no part in
# choosing a shape, knots, tier, estimator or split count.
#
# PRIMARY is the subset the pool identity can price - sales exact from
# (pool - previous pool) / 8.88%, lottery.ev.exact_lines_sold - so the
# multiplier's denominator is a measurement rather than a rolling median.
# A draw whose predecessor did not roll over cannot be priced, which is why
# 3197, 3200, 3206 and 3207 are absent.
HOLDOUT_PRIMARY_DRAWS = (3196, 3198, 3199, 3201, 3202, 3203, 3204, 3205)
HOLDOUT_SECONDARY_DRAWS = tuple(range(3196, 3208))   # 3196-3207, trend estimator
HOLDOUT_ROUNDS = (1, 2)

# NOT virgin, and it has to be said: two of the eight primary draws have had
# something equivalent to their popularity multiplier looked at already.
# data/mbw_validation.csv scores Must-Be-Won draws by comparing exact sales
# from the pool identity against the winner-count estimator - and the ratio of
# those two IS the popularity multiplier this test scores. For 3196 the audit
# went further and wrote the number down (§3.2: winner counts read 10.92m
# lines against 9.46m exact, i.e. a multiplier near 1.15, "round two was a
# birthday line"). Nobody has compared the two SPECIFICATIONS on these rows,
# which is the question here, but a human has seen their level. Disclosed
# rather than dropped: the test reports the slice with and without them, and
# the pre-registered veto is read off the full 16 rows.
HOLDOUT_SEEN_BY_SCORECARD = (3196, 3205)

# Draw 3208 onwards is deliberately NOT in the holdout. It had not been drawn
# when this file was written, which makes it the replication slice for later -
# spending it now would leave nothing that has never been looked at.
HOLDOUT_EXCLUDED_FROM = 3208

# The never-SCORED slice: the specification contest only ever tested on rows
# n//2 .. n of the training material (popularity_audit.specification_contest,
# edges = linspace(n//2, n, splits+1)), i.e. draws 2639-3195. Draws 2066-2638
# were training material in every split and were never a test set. They are
# not virgin - they moved the coefficients - but they never ranked a model.
NEVER_SCORED_FIRST_DRAW = 2066      # 2015-10-10
NEVER_SCORED_LAST_DRAW = 2638       # 2021-04-03, 573 rows
CONTEST_SCORED_FIRST_DRAW = 2639    # 2021-04-07 .. 2026-08-05, 574 rows

# --- metric and decision rule, fixed before the test ------------------------

METRIC = "mean squared error of log(multiplier), paired per observation"
DECISION_RULE = (
    "Install smooth only if it beats 3-bucket on the pre-registered slice AND "
    "the paired 90% bootstrap interval for the MSE difference excludes zero. "
    "A tie goes to the incumbent. A loss is accepted as a loss: no re-tuning "
    "against this slice, because a specification adjusted to win on its own "
    "holdout has no holdout left."
)


# --- the specification, as code ---------------------------------------------

def design_smooth(draws: np.ndarray) -> np.ndarray:
    """[1, SUM n, SUM max(n-12,0), SUM max(n-31,0)] per (draw, round)."""
    n = np.asarray(draws, dtype=float)
    cols = [np.ones(len(n)), n.sum(axis=1)]
    for k in KNOTS:
        cols.append(np.maximum(n - k, 0).sum(axis=1))
    return np.column_stack(cols)


def design_bucket(draws: np.ndarray) -> np.ndarray:
    """[1, #{n<=12}, #{n>31}] per (draw, round); 13-31 is the reference."""
    d = np.asarray(draws)
    return np.column_stack([np.ones(len(d)),
                            (d <= 12).sum(axis=1),
                            (d > 31).sum(axis=1)])


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The calibration's own estimator. Kept in one place so the frozen
    coefficients and any refit cannot come from different solvers."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def smooth_weight_table(beta=SMOOTH_BETA) -> np.ndarray:
    """Per-number pick-rate from the smooth coefficients.

    Un-damp by 6/3 (a Match-3 multiplier carries half the summed log weight),
    centre in logs, then scale so the 59 numbers average exactly 1.0 - the
    population constraint every weight vector in this project satisfies.
    """
    ns = np.arange(1, N_BALLS + 1, dtype=float)
    logw = UNDAMP * (beta[1] * ns
                     + beta[2] * np.maximum(ns - KNOTS[0], 0)
                     + beta[3] * np.maximum(ns - KNOTS[1], 0))
    w = np.exp(logw - logw.mean())
    return w * N_BALLS / w.sum()


def bucket_weight_table(beta=BUCKET_BETA) -> tuple:
    """The incumbent's three levels from its coefficients, same constraint."""
    r_low = float(np.exp(UNDAMP * beta[1]))
    r_high = float(np.exp(UNDAMP * beta[2]))
    w_mid = N_BALLS / (12 * r_low + 19 + 28 * r_high)
    return (r_low * w_mid, w_mid, r_high * w_mid)


def smooth_weight_fn(beta=SMOOTH_BETA):
    """The challenger as lottery/ev.py's `number_weight` would see it."""
    table = {int(n): float(w) for n, w in
             zip(range(1, N_BALLS + 1), smooth_weight_table(beta))}
    return lambda n: table[int(n)]


# --- drift guard ------------------------------------------------------------

def refit_from_archive() -> dict:
    """Re-derive both fits from the training material and report the drift.

    Any difference means the frozen numbers no longer describe what the code
    and data produce - a new draw in prize_tiers_history.csv, an estimator
    change, a library change. The test asserts this is zero.
    """
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    from scripts.calibrate_popularity import add_multiplier, load_joined

    df = add_multiplier(load_joined(CALIB_TIER))
    df = df[(df["draw_number"] >= TRAIN_FIRST_DRAW)
            & (df["draw_number"] <= TRAIN_LAST_DRAW)]
    cols = [f"Number_{i}" for i in range(1, N_PICK + 1)]

    def fits(frame):
        draws = frame[cols].to_numpy(int)
        y = np.log(frame["multiplier"].to_numpy())
        return fit_ols(design_smooth(draws), y), fit_ols(design_bucket(draws), y)

    smooth, bucket = fits(df)
    second = df[df["draw_number"] >= CONTEST_SCORED_FIRST_DRAW]
    smooth_2h, bucket_2h = fits(second)
    return {
        "obs": len(df),
        "obs_second_half": len(second),
        "smooth": smooth,
        "bucket": bucket,
        "smooth_second_half": smooth_2h,
        "bucket_second_half": bucket_2h,
    }


# --- what the slice can resolve ---------------------------------------------
# Section 12.6's rule: a tool meant to settle something must first show it can
# detect a planted truth. Here the planted truth is "the smooth curve is
# right", and the question is how often a holdout of n rows would notice.
# Synthetic draws only - no holdout row is read.

def separation(rng: np.random.Generator, n: int = 200_000) -> float:
    """E[(mu_smooth - mu_bucket)^2] over uniformly random draws.

    The systematic gap between the two frozen predictors: the entire effect
    the final test is trying to see, with no noise in it.
    """
    draws = rng.random((n, N_BALLS)).argsort(axis=1)[:, :N_PICK] + 1
    gap = (design_smooth(draws) @ np.array(SMOOTH_BETA)
           - design_bucket(draws) @ np.array(BUCKET_BETA))
    return float(np.mean(gap ** 2))


def power(rng: np.random.Generator, n_obs: int, sigma2: float,
          reps: int = 4000, alpha: float = 0.05) -> dict:
    """How often n_obs rows would rank smooth above 3-bucket, and clear zero.

    Truth = the frozen smooth curve, so the smooth predictor is correct and
    the bucket predictor carries the gap. Noise is Gaussian in log space with
    variance sigma2, set from the out-of-sample MSE the archive actually
    shows. Reports the share of runs where the paired mean difference is
    positive (smooth ahead) and where its one-sided t clears alpha.
    """
    from scipy import stats
    crit = float(stats.t.ppf(1 - alpha, n_obs - 1))   # one-sided, paired
    ahead = 0
    clears = 0
    for _ in range(reps):
        draws = rng.random((n_obs, N_BALLS)).argsort(axis=1)[:, :N_PICK] + 1
        mu_s = design_smooth(draws) @ np.array(SMOOTH_BETA)
        mu_b = design_bucket(draws) @ np.array(BUCKET_BETA)
        eps = rng.normal(0.0, np.sqrt(sigma2), n_obs)
        y = mu_s + eps
        d = (y - mu_b) ** 2 - (y - mu_s) ** 2      # >0 favours smooth
        m = float(d.mean())
        se = float(d.std(ddof=1)) / np.sqrt(n_obs)
        ahead += m > 0
        clears += se > 0 and (m / se) > crit
    return {"n_obs": n_obs, "ahead": ahead / reps, "significant": clears / reps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="re-derive the frozen fits from the archive")
    ap.add_argument("--power", action="store_true",
                    help="what each candidate slice could resolve")
    ap.add_argument("--sigma2", type=float, default=0.11432,
                    help="noise variance; default = the 3-bucket OOS MSE")
    args = ap.parse_args()

    print("=" * 72)
    print(f"POPULARITY-MODEL-V2 - FROZEN SPECIFICATION ({FROZEN_ON}, "
          f"{FROZEN_AT_COMMIT})")
    print("=" * 72)
    w = smooth_weight_table()
    print(f"  smooth beta : {', '.join(f'{b:+.8f}' for b in SMOOTH_BETA)}")
    print(f"  bucket beta : {', '.join(f'{b:+.8f}' for b in BUCKET_BETA)}")
    print(f"  smooth curve: ball 1 {w[0]:.3f}, peak {w.max():.3f} at ball "
          f"{int(w.argmax()) + 1}, ball 59 {w[-1]:.3f}")
    print(f"  bucket refit: {bucket_weight_table()[0]:.3f} / "
          f"{bucket_weight_table()[1]:.3f} / {bucket_weight_table()[2]:.3f}"
          f"   (installed {INSTALLED_WEIGHTS[0]} / {INSTALLED_WEIGHTS[1]} / "
          f"{INSTALLED_WEIGHTS[2]})")
    print(f"  holdout     : draws {HOLDOUT_PRIMARY_DRAWS} x rounds "
          f"{HOLDOUT_ROUNDS} = {2 * len(HOLDOUT_PRIMARY_DRAWS)} rows")
    print()

    if args.verify:
        r = refit_from_archive()
        drift = max(
            np.abs(np.array(r["smooth"]) - np.array(SMOOTH_BETA)).max(),
            np.abs(np.array(r["bucket"]) - np.array(BUCKET_BETA)).max(),
            np.abs(np.array(r["smooth_second_half"])
                   - np.array(SMOOTH_BETA_SECOND_HALF)).max(),
            np.abs(np.array(r["bucket_second_half"])
                   - np.array(BUCKET_BETA_SECOND_HALF)).max(),
        )
        print(f"  refit on {r['obs']} rows (frozen: {TRAIN_OBS}), of which "
              f"{r['obs_second_half']} in the second half")
        print(f"  max |drift| over all four coefficient vectors: {drift:.3e}")
        print("  OK - the frozen numbers still describe the archive"
              if drift < 1e-9 and r["obs"] == TRAIN_OBS
              else "  DRIFT - the freeze no longer matches; do not run the test")
        print()

    if args.power:
        rng = np.random.default_rng(20260919)
        sep = separation(rng)
        print(f"  systematic separation E[(mu_s - mu_b)^2] = {sep:.5f}")
        print(f"  noise variance assumed                   = {args.sigma2:.5f}")
        print()
        print(f"  {'slice':38} {'rows':>6} {'smooth ahead':>13} "
              f"{'clears 0':>10}")
        for label, n in (("holdout, exact sales (3196-3205)", 16),
                         ("holdout, all rounds (3196-3207)", 24),
                         ("never-scored first half (2066-2638)", 573),
                         ("what the contest scored (2639-3195)", 574)):
            p = power(rng, n, args.sigma2)
            print(f"  {label:38} {n:>6} {p['ahead']:>12.0%} "
                  f"{p['significant']:>10.0%}")
        print()
        print("  'smooth ahead' is the sign alone; 'clears 0' is the "
              "pre-registered")
        print("  rule (one-sided paired t at 5%). A slice that cannot clear "
              "zero even")
        print("  when the challenger is TRUE cannot install it - it can only "
              "refute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
