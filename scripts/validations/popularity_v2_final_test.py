#!/usr/bin/env python3
"""The one-shot evaluator for popularity-model-v2. Frozen before it was run.

It fits NOTHING. Both specifications arrive as fixed coefficient vectors from
popularity_v2_frozen.py; this file only applies them to rows they have not
seen and measures which predicts better.

    Primary metric   d_i = (y_i - yhat_bucket,i)^2 - (y_i - yhat_smooth,i)^2
                     d > 0 favours the smooth challenger.
    Reported         effect size (mean d, and MSE for each model), a block
                     bootstrap interval, and the share of draws favouring
                     each model. The p-value is printed last and decides
                     nothing on its own.

THREE OUTCOMES ARE ALLOWED, and "inconclusive" is one of them. A 16-row
prospective holdout that fails to clear an interval has not crowned the
incumbent; it has said "not enough evidence yet, keep collecting".

WHAT THE SLICES ARE (popularity-v2-spec.md section 3)

  retrospective  draws 2066-2638, scored with the coefficients fitted on
                 2639-3195 only. These rows trained coefficients in the
                 original contest and fed the 2026-07-25 production
                 calibration; they never ranked a specification. Strong
                 reverse-temporal validation, NOT an independent holdout,
                 and never on its own grounds for installing anything.

  prospective    draws 3196-3207, which prize_tiers_history.csv does not
                 contain and load_joined() has therefore never returned.
                 The real holdout. Draws 3196 and 3205 are flagged in every
                 report: the Must-Be-Won scorecard compared two sales
                 estimates for them, and the ratio of those estimates IS
                 this multiplier.

  replication    draw 3208 onwards. Not read here, by anything, at all.

TWO NUISANCE CORRECTIONS, both fixed before the run, both common-mode

1. WEEKDAY. 68% of the residual variance in the calibration's multiplier is
   a Wednesday/Saturday offset (Wed -0.257, Sat +0.257 in logs, measured on
   2639-3195): `add_multiplier` divides by a rolling median over a window
   holding BOTH weekdays, which cannot remove a level difference between
   them. It does not bias either fit - the drawn numbers are independent of
   the weekday - but it triples the noise the comparison has to see through.
   The retrospective slice therefore has the weekday means SUBTRACTED, and
   those means come from the fitting half (2639-3195), never from the rows
   being scored. The prospective slice needs no such correction: its
   denominator is that draw's own exact sales, so no weekday level enters.

2. LEVEL. The exact-sales multiplier and the rolling-median multiplier sit
   at slightly different levels (about -0.05 in logs on the 28 training rows
   where both exist). A level shift is common to both models, so it cannot
   change which one predicts better - but it does add variance. The
   evaluator removes it using the MIDPOINT of the two predictions, which is
   symmetric by construction: no version of this correction can favour a
   model, and the uncorrected numbers are printed beside the corrected ones
   every time.

Run (synthetic only - this is what was run before the freeze):
    PYTHONPATH=. python scripts/validations/popularity_v2_final_test.py --self-test
    PYTHONPATH=. python scripts/validations/popularity_v2_final_test.py --power

Run for real, once:
    PYTHONPATH=. python scripts/validations/popularity_v2_final_test.py --final
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lottery.ev import P_MATCH_3, exact_lines_sold
from scripts.validations import popularity_v2_frozen as frozen

# --- pre-registered analysis choices ---------------------------------------

BOOTSTRAP_REPS = 20_000
CI_LEVEL = 0.90
RNG_SEED = 20260919

# The resampling unit is a DRAW, never a row: rounds one and two of the same
# draw share their sales denominator, and on the training material the two
# rounds' residuals correlate at +0.91. Treating 16 rows as 16 independent
# observations would halve the interval for free.
#
# Across draws the paired difference is close to white - |autocorrelation|
# stays under 0.11 at every lag from 1 to 60 on 2639-3195 - but the rolling
# median shares a window of 51 draws with its neighbours, so the
# retrospective slice uses a MOVING BLOCK of draws rather than single draws.
# L = 25 is half that window; L = 1 and L = 51 are printed beside it every
# run, so nobody can pick a length after seeing which one helps.
BLOCK_DRAWS_RETROSPECTIVE = 25
BLOCK_LENGTHS_REPORTED = (1, 25, 51)

# Where a one-shot result is written. Never outputs/predictions/latest.json:
# that file is the advisor's real verdict (CLAUDE.md), and this is research.
RESULT_DIR = Path("outputs/popularity_v2")


# --- interpretation, frozen -------------------------------------------------

SUPPORTS_SMOOTH = "evidence supports smooth"
SUPPORTS_INCUMBENT = "evidence supports incumbent"
INCONCLUSIVE = "inconclusive - collect prospective evidence"


def interpret(d_mean: float, ci: tuple) -> str:
    """The only reading rule, fixed before any real row was scored.

    An interval containing zero is INCONCLUSIVE. It is not a win for the
    incumbent: the incumbent stays installed because it is installed, which
    is a different statement from the data preferring it.
    """
    lo, hi = ci
    if lo > 0 and d_mean > 0:
        return SUPPORTS_SMOOTH
    if hi < 0 and d_mean < 0:
        return SUPPORTS_INCUMBENT
    return INCONCLUSIVE


# --- the metric -------------------------------------------------------------

def paired_difference(y: np.ndarray, mu_smooth: np.ndarray,
                      mu_bucket: np.ndarray, centre: bool = True) -> np.ndarray:
    """d_i, positive where the smooth prediction is closer.

    `centre` removes one common level: the mean gap between y and the
    MIDPOINT of the two predictions. Symmetric in the two models by
    construction - it shifts both residuals by the same constant.
    """
    if centre:
        y = y - np.mean(y - (mu_smooth + mu_bucket) / 2.0)
    return (y - mu_bucket) ** 2 - (y - mu_smooth) ** 2


def draw_mean_t_ci(d: np.ndarray, draw_ids: np.ndarray,
                   level: float = CI_LEVEL) -> tuple:
    """Student interval on the per-DRAW means.

    Aggregating to draws disposes of within-draw dependence exactly rather
    than modelling it, which matters here because rounds one and two share a
    sales denominator (their residuals correlate +0.91 on the training half).
    Measured coverage at the equal-MSE null, synthetic, 400 runs:

        groups  row bootstrap  draw bootstrap  this
          40         88%            88%         91%
           8         85%            81%         90%

    The percentile bootstrap under-covers on eight groups, which is exactly
    the size of the prospective holdout, so this is the primary interval
    there and the bootstrap is printed beside it.
    """
    from scipy import stats
    per = pd.Series(d).groupby(draw_ids).mean().to_numpy()
    k = len(per)
    if k < 2:
        return (float("-inf"), float("inf"))
    half = stats.t.ppf(1 - (1 - level) / 2, k - 1) * per.std(ddof=1) / np.sqrt(k)
    return (float(per.mean() - half), float(per.mean() + half))


def block_bootstrap_ci(d: np.ndarray, draw_ids: np.ndarray, reps: int,
                       level: float, rng: np.random.Generator,
                       block_draws: int = 1) -> tuple:
    """Percentile interval for mean(d), resampling blocks of whole draws."""
    order = np.argsort(draw_ids, kind="stable")
    d = d[order]
    ids = draw_ids[order]
    uniq, starts = np.unique(ids, return_index=True)
    groups = np.split(d, starts[1:])
    n_groups = len(groups)
    if n_groups < 2:
        return (float("-inf"), float("inf"))

    block_draws = max(1, min(block_draws, n_groups))
    n_blocks = int(np.ceil(n_groups / block_draws))
    # Moving blocks: every start position is allowed, so the estimator does
    # not depend on where the archive happens to begin.
    starts_allowed = np.arange(n_groups - block_draws + 1)
    means = np.empty(reps)
    for b in range(reps):
        pick = rng.choice(starts_allowed, size=n_blocks)
        rows = np.concatenate([np.concatenate(groups[s:s + block_draws])
                               for s in pick])
        means[b] = rows.mean()
    alpha = (1.0 - level) / 2.0
    return (float(np.percentile(means, 100 * alpha)),
            float(np.percentile(means, 100 * (1 - alpha))))


def evaluate(y: np.ndarray, draws: np.ndarray, draw_ids: np.ndarray,
             beta_smooth, beta_bucket, rng: np.random.Generator,
             block_draws: int = 1, reps: int = BOOTSTRAP_REPS,
             primary: str = "t") -> dict:
    """Everything the report needs for one slice. No fitting anywhere.

    `primary` names which interval the verdict is read off - "t" for the
    prospective slices, "block" for the retrospective one, both fixed in
    SLICES below. The other interval is computed anyway and printed, so the
    choice cannot be made after the numbers are in.
    """
    mu_s = frozen.design_smooth(draws) @ np.array(beta_smooth)
    mu_b = frozen.design_bucket(draws) @ np.array(beta_bucket)
    d = paired_difference(y, mu_s, mu_b)
    d_raw = paired_difference(y, mu_s, mu_b, centre=False)

    shift = np.mean(y - (mu_s + mu_b) / 2.0)
    yc = y - shift
    per_draw = pd.Series(d).groupby(draw_ids).mean()
    ci_block = block_bootstrap_ci(d, draw_ids, reps, CI_LEVEL, rng, block_draws)
    ci_t = draw_mean_t_ci(d, draw_ids)
    ci = ci_t if primary == "t" else ci_block

    n_draws = len(np.unique(draw_ids))
    se = per_draw.std(ddof=1) / np.sqrt(n_draws) if n_draws > 1 else np.inf
    return {
        "rows": int(len(y)),
        "draws": int(n_draws),
        "mse_smooth": float(np.mean((yc - mu_s) ** 2)),
        "mse_bucket": float(np.mean((yc - mu_b) ** 2)),
        "d_mean": float(np.mean(d)),
        "d_mean_uncentred": float(np.mean(d_raw)),
        "ci": ci,
        "ci_method": primary,
        "ci_t": ci_t,
        "ci_block": ci_block,
        "draws_favouring_smooth": int((per_draw > 0).sum()),
        "t_on_draws": float(per_draw.mean() / se) if se else 0.0,
        "level_shift": float(shift),
        "verdict": interpret(float(np.mean(d)), ci),
    }


# --- the slices -------------------------------------------------------------

def _numbers(frame) -> np.ndarray:
    return frame[[f"Number_{i}" for i in range(1, 7)]].to_numpy(int)


def load_retrospective() -> dict:
    """Draws 2066-2638, weekday-adjusted with means from the fitting half."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    from scripts.calibrate_popularity import add_multiplier, load_joined

    df = add_multiplier(load_joined(frozen.CALIB_TIER))
    df = df.assign(weekday=pd.to_datetime(df["Draw Date"]).dt.weekday,
                   logmult=np.log(df["multiplier"].to_numpy()))
    fit_half = df[df["draw_number"] >= frozen.CONTEST_SCORED_FIRST_DRAW]
    mu_s = frozen.design_smooth(_numbers(fit_half)) @ np.array(
        frozen.SMOOTH_BETA_SECOND_HALF)
    offsets = (pd.Series(fit_half["logmult"].to_numpy() - mu_s)
               .groupby(fit_half["weekday"].to_numpy()).mean())

    scored = df[(df["draw_number"] >= frozen.NEVER_SCORED_FIRST_DRAW)
                & (df["draw_number"] <= frozen.NEVER_SCORED_LAST_DRAW)]
    y = scored["logmult"].to_numpy() - scored["weekday"].map(offsets).to_numpy()
    return {"y": y, "draws": _numbers(scored),
            "draw_ids": scored["draw_number"].to_numpy(),
            "beta_smooth": frozen.SMOOTH_BETA_SECOND_HALF,
            "beta_bucket": frozen.BUCKET_BETA_SECOND_HALF,
            "block": BLOCK_DRAWS_RETROSPECTIVE,
            # 573 draws: enough groups for the block bootstrap, and the
            # rolling median shares a window with its neighbours, so blocks
            # are the honest primary here.
            "primary": "block"}


def load_prospective(exact: bool = True) -> dict:
    """Draws 3196-3207 from the collector's own files, never in the join.

    exact=True  : the 8 draws the pool identity can price, denominator exact.
    exact=False : all 12 draws, denominator from the rolling-median trend -
                  reported always, decisive never.
    """
    tiers = pd.read_csv("data/prize_tiers.csv")
    hist = pd.read_csv("data/lotto_full_history.csv").rename(
        columns={"DrawNumber": "draw_number", "Round": "round"})
    rows = tiers[(tiers["tier"] == frozen.CALIB_TIER)
                 & (tiers["draw_number"] > frozen.TRAIN_LAST_DRAW)
                 & (tiers["draw_number"] < frozen.HOLDOUT_EXCLUDED_FROM)
                 & (tiers["winners"] > 0)]
    df = rows.merge(hist, on=["draw_number", "round"], how="inner")

    if exact:
        lines = exact_lines_sold(pd.read_csv("data/draw_pools.csv"))
        df = df[df["draw_number"].isin(frozen.HOLDOUT_PRIMARY_DRAWS)]
        n = df["draw_number"].map(lines).to_numpy(float)
        y = np.log(df["winners"].to_numpy() / (P_MATCH_3 * n))
    else:
        import logging
        logging.getLogger().setLevel(logging.WARNING)
        from scripts.calibrate_popularity import add_multiplier, load_joined
        joined = add_multiplier(load_joined(frozen.CALIB_TIER))
        raw = df["winners"].to_numpy() / P_MATCH_3
        trend = np.median(joined["sales_est"].to_numpy()[-frozen.TREND_WINDOW:])
        y = np.log(raw / trend)

    return {"y": y, "draws": _numbers(df),
            "draw_ids": df["draw_number"].to_numpy(),
            "beta_smooth": frozen.SMOOTH_BETA,
            "beta_bucket": frozen.BUCKET_BETA,
            "block": 1,
            # 8 or 12 draws: the percentile bootstrap under-covers this small,
            # the t on per-draw means does not (see draw_mean_t_ci).
            "primary": "t"}


# --- synthetic checks: the evaluator must prove it can detect --------------

def _synthetic(rng, n_draws, truth, sigma, shared=0.0, rounds=2):
    """Draws, y and ids with an optional per-draw shared shock.

    `shared` is the dependence the block bootstrap exists for: both rounds
    of a draw move together, exactly as they do when they divide by the same
    sales figure.
    """
    draws = rng.random((n_draws * rounds, 59)).argsort(axis=1)[:, :6] + 1
    ids = np.repeat(np.arange(n_draws), rounds)
    mu = frozen.design_smooth(draws) @ np.array(frozen.SMOOTH_BETA)
    mu_b = frozen.design_bucket(draws) @ np.array(frozen.BUCKET_BETA)
    signal = {"smooth": mu, "bucket": mu_b, "midpoint": (mu + mu_b) / 2.0}[truth]
    eps = rng.normal(0, sigma, len(draws))
    if shared:
        eps = eps + np.repeat(rng.normal(0, shared, n_draws), rounds)
    return draws, signal + eps, ids


def self_test(rng=None, reps=200, sigma=0.179) -> list:
    """Planted truths, then the dependence the bootstrap is there to handle.

    Seeded internally: a reported rate that moves depending on what ran
    before it is a number nobody can check.
    """
    rng = np.random.default_rng(RNG_SEED)
    out = []
    for truth, expect in (("smooth", SUPPORTS_SMOOTH),
                          ("bucket", SUPPORTS_INCUMBENT)):
        hits = 0
        for _ in range(reps):
            draws, y, ids = _synthetic(rng, 150, truth, sigma)
            r = evaluate(y, draws, ids, frozen.SMOOTH_BETA, frozen.BUCKET_BETA,
                         rng, reps=400, primary="t")
            hits += r["verdict"] == expect
        out.append((f"planted {truth}, 150 draws", expect, hits / reps))

    # Equal-MSE null: the truth sits midway, so neither model is right and
    # the interval should cover zero at its nominal rate.
    false_calls = 0
    for _ in range(reps):
        draws, y, ids = _synthetic(rng, 150, "midpoint", sigma)
        r = evaluate(y, draws, ids, frozen.SMOOTH_BETA, frozen.BUCKET_BETA,
                     rng, reps=400, primary="t")
        false_calls += r["verdict"] != INCONCLUSIVE
    out.append(("equal-MSE null, 150 draws", "inconclusive",
                1 - false_calls / reps))

    # Dependence: a per-draw shared shock, the structure the rounds really
    # have. Coverage is checked at BOTH slice sizes and for both intervals,
    # because this is what chose the primary.
    for n_draws in (40, 8):
        cover_t = cover_b = 0
        for _ in range(reps):
            draws, y, ids = _synthetic(rng, n_draws, "midpoint", sigma * 0.5,
                                       shared=sigma)
            r = evaluate(y, draws, ids, frozen.SMOOTH_BETA,
                         frozen.BUCKET_BETA, rng, reps=400)
            cover_t += r["ci_t"][0] <= 0.0 <= r["ci_t"][1]
            cover_b += r["ci_block"][0] <= 0.0 <= r["ci_block"][1]
        out.append((f"shared shock, {n_draws} draws, t interval",
                    "covers 0 at 90%", cover_t / reps))
        out.append((f"shared shock, {n_draws} draws, bootstrap",
                    "covers 0 at 90%", cover_b / reps))
    return out


def power_table(rng=None, sigma2: float = 0.032, reps: int = 600) -> list:
    """How many DRAWS the prospective stream needs before it can speak.

    Seeded internally for the same reason as self_test. Monte Carlo error at
    600 repetitions is about +/-2 percentage points.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    for n_draws in (8, 12, 25, 50, 100, 200):
        wins = 0
        for _ in range(reps):
            draws, y, ids = _synthetic(rng, n_draws, "smooth", np.sqrt(sigma2))
            r = evaluate(y, draws, ids, frozen.SMOOTH_BETA,
                         frozen.BUCKET_BETA, rng, reps=300, primary="t")
            wins += r["verdict"] == SUPPORTS_SMOOTH
        rows.append({"draws": n_draws, "rows": 2 * n_draws,
                     "detects": wins / reps})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic only: can this evaluator detect anything?")
    ap.add_argument("--power", action="store_true",
                    help="synthetic only: draws needed for the stream")
    ap.add_argument("--final", action="store_true",
                    help="READ REAL DATA. One shot, writes a timestamped file.")
    ap.add_argument("--sigma2", type=float, default=0.032,
                    help="noise variance after the weekday artifact is removed")
    args = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)

    if args.self_test:
        print("=" * 72)
        print("SELF-TEST - synthetic data only, no slice is read")
        print("=" * 72)
        for name, expect, rate in self_test(rng):
            print(f"  {name:34} expect {expect:24} {rate:6.0%}")
        print()

    if args.power:
        print("=" * 72)
        print(f"POWER - prospective stream, noise variance {args.sigma2:.4f}")
        print("=" * 72)
        print(f"  {'draws':>7} {'rows':>6} {'says supports smooth':>22}")
        for row in power_table(rng, args.sigma2):
            print(f"  {row['draws']:>7} {row['rows']:>6} {row['detects']:>21.0%}")
        print()

    if args.final:
        return run_final(rng)
    if not (args.self_test or args.power):
        ap.error("nothing to do: pass --self-test, --power or --final")
    return 0


def run_final(rng) -> int:
    """The one shot. Reads the real slices, prints and records the result."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    # One shot means one shot. A second run against the same frozen
    # specification would be the same data answering the same question
    # again, and the only thing a re-run can add is the temptation to
    # prefer whichever answer came second.
    previous = sorted(RESULT_DIR.glob("final-test-*.json"))
    if previous:
        print(f"  REFUSED: this test has already been run - {previous[-1]}")
        print("  Read that file. Re-running needs a new frozen specification,")
        print("  a new slice, and a reason written down before the run.")
        return 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = RESULT_DIR / f"final-test-{stamp}.json"

    report = {"frozen_commit": frozen.FROZEN_AT_COMMIT, "run_at": stamp,
              "slices": {}}

    print("=" * 72)
    print("POPULARITY-MODEL-V2 - FINAL TEST (one shot)")
    print("=" * 72)
    print(f"  metric   : {frozen.METRIC}")
    print(f"  positive d favours SMOOTH; {CI_LEVEL:.0%} block-bootstrap "
          f"interval")
    print()

    for name, data, note in (
        ("retrospective 2066-2638", load_retrospective(),
         "reverse-temporal validation, NOT an independent holdout"),
        ("prospective 3196-3205 (exact sales)", load_prospective(True),
         f"the real holdout; {frozen.HOLDOUT_SEEN_BY_SCORECARD} seen by the "
         "scorecard"),
        ("prospective 3196-3207 (trend)", load_prospective(False),
         "reported always, decisive never"),
    ):
        r = evaluate(data["y"], data["draws"], data["draw_ids"],
                     data["beta_smooth"], data["beta_bucket"], rng,
                     block_draws=data["block"], primary=data["primary"])
        report["slices"][name] = {**r, "note": note}
        print(f"  {name}")
        print(f"     {note}")
        print(f"     rows {r['rows']}, draws {r['draws']}")
        print(f"     MSE  smooth {r['mse_smooth']:.5f}   bucket "
              f"{r['mse_bucket']:.5f}")
        print(f"     d    {r['d_mean']:+.5f}   90% CI "
              f"[{r['ci'][0]:+.5f}, {r['ci'][1]:+.5f}]  ({r['ci_method']})")
        print(f"          t interval [{r['ci_t'][0]:+.5f}, "
              f"{r['ci_t'][1]:+.5f}]   bootstrap [{r['ci_block'][0]:+.5f}, "
              f"{r['ci_block'][1]:+.5f}]")
        print(f"          uncentred d {r['d_mean_uncentred']:+.5f}, level "
              f"shift {r['level_shift']:+.4f}")
        print(f"     draws favouring smooth: {r['draws_favouring_smooth']}"
              f"/{r['draws']}")
        print(f"     -> {r['verdict']}")
        print()

        if data["block"] > 1:
            for L in BLOCK_LENGTHS_REPORTED:
                ci = block_bootstrap_ci(
                    paired_difference(
                        data["y"],
                        frozen.design_smooth(data["draws"]) @ np.array(
                            data["beta_smooth"]),
                        frozen.design_bucket(data["draws"]) @ np.array(
                            data["beta_bucket"])),
                    data["draw_ids"], BOOTSTRAP_REPS, CI_LEVEL, rng, L)
                print(f"     block length {L:>3} draws: CI [{ci[0]:+.5f}, "
                      f"{ci[1]:+.5f}]")
            print()

    out_file.write_text(json.dumps(report, indent=2))
    print(f"  written: {out_file}")
    print("  The installed model is unchanged. This file reports; it never "
          "edits lottery/ev.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
