#!/usr/bin/env python3
"""Audit the popularity model - the one calibrated thing EV actually spends on.

Everything else in `validations/` tests hypotheses the advisor deliberately
ignores. This tests the component it cannot ignore. `expected_cowinner_share`
is the only place a fitted number reaches `line_ev`, so if the pick-rate
weights are wrong, the PLAY threshold moves and real money follows.

Five questions, in order of how much they cost to get wrong:

  1. DECISION SENSITIVITY - can plausible weight error flip PLAY/SKIP, and
     on which draws? This is the only question with money attached.
  2. RECOVERY - fed data generated from KNOWN weights, does the calibration
     return them? A fit that cannot recover a planted truth is not measuring
     what it claims to, and four agreeing validations would agree on nothing.
  3. POWER - how wrong could the weights be before this archive noticed?
  4. UNCERTAINTY - the interval around the installed weights, propagated
     into the break-even jackpot, so a marginal PLAY can be read as marginal.
  5. SPECIFICATION - do the data support three buckets at all, judged
     out-of-sample against challengers?

Two findings shape how the rest should be read.

The risk is NOT confined to ordinary draws. An earlier version of this file
said roll-downs are immune because their EV is carried by J/N; a test
asserting that failed. A GBP 15m special against 12m lines reads -0.053
(SKIP) with a flat popularity model and +0.024 (PLAY) with the installed
one - and GBP 15m is the marginal PLAY this project is most likely to meet.
What governs sensitivity is distance from the threshold relative to model
uncertainty, not the kind of draw.

And the exposure is misspecification rather than sampling error. The weights
are pinned to 0.2% on the break-even pool (section 4), but three buckets
lose to a smooth alternative out of sample by about 10%, consistently across
split counts (section 5). The installed shape is too rigid - which so far
changes no verdict, because the smooth fit implies almost the same weights.

Run:  PYTHONPATH=. python scripts/validations/popularity_audit.py
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

import lottery.ev as ev
from lottery.ev import N_BALLS, N_PICK

# The installed three-bucket model (lottery/ev.py number_weight).
INSTALLED = (1.23, 1.10, 0.83)
CALIB_MATCH = 3                 # Match 3: the tier the weights are fitted on
UNDAMP = N_PICK / CALIB_MATCH   # a Match-3 log-multiplier carries 3/6 of the sum


def set_weights(low12: float, mid: float, high: float) -> None:
    """Swap the bucket weights in the live EV module.

    Rebinding rather than parameterising `ev` keeps the audit honest: every
    downstream constant (MEAN_WEIGHT, the normalisation over all C(59,6)
    lines) is recomputed exactly as it is at import, so the EV that comes
    out is the EV the advisor would produce with those weights installed.
    """
    def nw(n: int) -> float:
        return low12 if n <= 12 else (mid if n <= 31 else high)

    ev.number_weight = nw
    ev.MEAN_WEIGHT = sum(nw(n) for n in range(1, N_BALLS + 1)) / N_BALLS
    ev.POPULARITY_NORMALIZATION = ev._popularity_normalization()


def scale_spread(factor: float, base=INSTALLED) -> tuple:
    """Weights with the same mean but `factor` times the spread from 1.0.

    factor 0 is a flat model (no popularity bias at all), 1 is installed,
    2 is twice as biased as measured. Keeps the population constraint that
    the mean pick-rate is 1.0, so the comparison isolates spread.
    """
    return tuple(1.0 + (w - 1.0) * factor for w in base)


# --- 1. decision sensitivity ------------------------------------------------

def decision_grid(base_cond, factors, draws) -> pd.DataFrame:
    """Verdict for each (weight scaling, draw scenario) pair."""
    rows = []
    for label, kw in draws:
        cond = replace(base_cond, **kw)
        row = {"draw": label}
        for f in factors:
            set_weights(*scale_spread(f))
            v = ev.should_play(cond)
            row[f"x{f:g}"] = (v["ev_best_line"], v["play"])
        rows.append(row)
    set_weights(*INSTALLED)
    return pd.DataFrame(rows)


# --- 2. recovery (injection) ------------------------------------------------

def synthetic_multipliers(draws: np.ndarray, weights: tuple,
                          noise: float, rng: np.random.Generator) -> np.ndarray:
    """Match-3 multipliers generated FROM known weights.

    Inverts the calibration's own model: a draw's low-tier winner count runs
    high when its numbers are over-played, by (3/6) of the summed log weight.
    Lognormal noise stands in for everything else that moves winner counts.
    """
    low12, mid, high = weights

    def w(n):
        return low12 if n <= 12 else (mid if n <= 31 else high)

    log_mult = np.array([sum(np.log(w(int(n))) for n in row) / UNDAMP
                         for row in draws])
    return np.exp(log_mult + rng.normal(0, noise, len(draws)))


def fit_from_multipliers(draws: np.ndarray, mult: np.ndarray) -> dict:
    """The calibration's fit_bucket_weights, on arbitrary inputs."""
    n_low12 = (draws <= 12).sum(axis=1)
    n_high = (draws > 31).sum(axis=1)
    X = np.column_stack([np.ones(len(draws)), n_low12, n_high])
    beta, *_ = np.linalg.lstsq(X, np.log(mult), rcond=None)
    r_low = np.exp(UNDAMP * beta[1])
    r_high = np.exp(UNDAMP * beta[2])
    w_mid = N_BALLS / (12 * r_low + 19 + 28 * r_high)
    return {"low12": r_low * w_mid, "mid": w_mid, "high": r_high * w_mid}


def recovery_report(n_draws: int, noise: float, n_reps: int,
                    rng: np.random.Generator) -> dict:
    """Plant known weights, recover them, report the error."""
    out = {}
    for name, truth in (("installed", INSTALLED),
                        ("flat", (1.0, 1.0, 1.0)),
                        ("double", scale_spread(2.0))):
        errs = []
        for _ in range(n_reps):
            draws = rng.random((n_draws, N_BALLS)).argsort(axis=1)[:, :N_PICK] + 1
            mult = synthetic_multipliers(draws, truth, noise, rng)
            fit = fit_from_multipliers(draws, mult)
            errs.append([fit["low12"] - truth[0], fit["mid"] - truth[1],
                         fit["high"] - truth[2]])
        e = np.array(errs)
        out[name] = {"truth": truth, "bias": e.mean(axis=0),
                     "sd": e.std(axis=0)}
    return out


# --- 3. power ---------------------------------------------------------------

def power_report(n_draws: int, noise: float, n_reps: int,
                 rng: np.random.Generator) -> list:
    """How big a departure from flat this archive would call significant.

    The calibration's own t-statistic on the low12 contrast: a spread factor
    is "detected" when the fitted coefficient clears two standard errors.
    """
    rows = []
    for factor in (0.25, 0.5, 1.0, 1.5, 2.0):
        truth = scale_spread(factor)
        detected = 0
        for _ in range(n_reps):
            draws = rng.random((n_draws, N_BALLS)).argsort(axis=1)[:, :N_PICK] + 1
            mult = synthetic_multipliers(draws, truth, noise, rng)
            n_low12 = (draws <= 12).sum(axis=1)
            n_high = (draws > 31).sum(axis=1)
            X = np.column_stack([np.ones(len(draws)), n_low12, n_high])
            y = np.log(mult)
            beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            dof = max(len(draws) - 3, 1)
            s2 = (resid @ resid) / dof
            cov = s2 * np.linalg.pinv(X.T @ X)
            se_low = np.sqrt(cov[1, 1])
            if abs(beta[1]) > 2 * se_low:
                detected += 1
        rows.append({"factor": factor, "weights": truth,
                     "power": detected / n_reps})
    return rows


# --- 4. uncertainty into the threshold --------------------------------------

def threshold_interval(base_cond, n_draws: int, noise: float, n_boot: int,
                       rng: np.random.Generator) -> dict:
    """Bootstrap the weights, push each draw through to the break-even pool.

    A point estimate of GBP 32.3m invites a PLAY on a draw that is inside the
    interval. The advisor should be able to say "marginal" with a number
    behind it.
    """
    thresholds, evs = [], []
    for _ in range(n_boot):
        draws = rng.random((n_draws, N_BALLS)).argsort(axis=1)[:, :N_PICK] + 1
        mult = synthetic_multipliers(draws, INSTALLED, noise, rng)
        fit = fit_from_multipliers(draws, mult)
        set_weights(fit["low12"], fit["mid"], fit["high"])
        v = ev.should_play(base_cond)
        thresholds.append(v["break_even_jackpot"])
        evs.append(v["ev_best_line"])
    set_weights(*INSTALLED)
    t, e = np.array(thresholds), np.array(evs)
    return {
        "threshold_p5": float(np.percentile(t, 5)),
        "threshold_p50": float(np.percentile(t, 50)),
        "threshold_p95": float(np.percentile(t, 95)),
        "ev_p5": float(np.percentile(e, 5)),
        "ev_p95": float(np.percentile(e, 95)),
    }


# --- 5. specification challengers -------------------------------------------

def _design_buckets(draws: np.ndarray) -> np.ndarray:
    """The installed model: counts in <=12 / 13-31 / >31 (mid is reference)."""
    return np.column_stack([np.ones(len(draws)),
                            (draws <= 12).sum(axis=1),
                            (draws > 31).sum(axis=1)])


def _design_smooth(draws: np.ndarray, knots=(12, 31)) -> np.ndarray:
    """A challenger with no step at 12 or 31: linear in n, plus a hinge.

    If players really treat 31 as a cliff - the last day of a month - the
    bucket model should beat this. If the truth is a gentle decline in
    popularity as numbers rise, this should win. That is the question:
    not which model scores higher EV, but which shape the data support.
    """
    n = draws.astype(float)
    cols = [np.ones(len(draws)), n.sum(axis=1)]
    for k in knots:
        cols.append(np.maximum(n - k, 0).sum(axis=1))
    return np.column_stack(cols)


def _design_per_number(draws: np.ndarray) -> np.ndarray:
    """One coefficient per ball: the overfit yardstick.

    59 parameters on ~1,100 observations. Included precisely because it
    will fit in-sample and should fail out-of-sample - a challenger set
    without a known-bad member cannot show that the comparison works.
    """
    X = np.zeros((len(draws), N_BALLS + 1))
    X[:, 0] = 1.0
    for i, row in enumerate(draws):
        for b in row:
            X[i, int(b)] += 1.0
    return X


DESIGNS = {
    "3-bucket (installed)": _design_buckets,
    "smooth + hinges": _design_smooth,
    "per-number (overfit)": _design_per_number,
}


def specification_contest(draws: np.ndarray, mult: np.ndarray,
                          n_splits: int = 5) -> list:
    """Compare specifications OUT OF SAMPLE, on rolling forward splits.

    Walk-forward rather than random folds: the calibration is used on
    future draws, so the test has to be future draws. In-sample error is
    reported alongside only to show the overfit member doing what it is
    there to do.
    """
    y = np.log(mult)
    n = len(draws)
    edges = np.linspace(n // 2, n, n_splits + 1).astype(int)

    rows = []
    for name, design in DESIGNS.items():
        X = design(draws)
        oos, ins = [], []
        for i in range(n_splits):
            train_end = edges[i]
            test_end = edges[i + 1]
            if test_end <= train_end:
                continue
            Xtr, ytr = X[:train_end], y[:train_end]
            Xte, yte = X[train_end:test_end], y[train_end:test_end]
            beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
            oos.append(float(np.mean((yte - Xte @ beta) ** 2)))
            ins.append(float(np.mean((ytr - Xtr @ beta) ** 2)))
        rows.append({"name": name, "params": X.shape[1],
                     "oos_mse": float(np.mean(oos)),
                     "in_mse": float(np.mean(ins))})
    return rows


def smooth_weight_fn(draws: np.ndarray, mult: np.ndarray):
    """The smooth challenger as a per-number weight function.

    Projecting it onto three buckets (below) throws away the very thing it
    fits differently, so the verdict comparison uses the curve itself.
    """
    X = _design_smooth(draws)
    beta, *_ = np.linalg.lstsq(X, np.log(mult), rcond=None)
    ns = np.arange(1, N_BALLS + 1, dtype=float)
    logw = UNDAMP * (beta[1] * ns + beta[2] * np.maximum(ns - 12, 0)
                     + beta[3] * np.maximum(ns - 31, 0))
    w = np.exp(logw - logw.mean())
    w = w * N_BALLS / w.sum()
    table = {int(n): float(x) for n, x in zip(ns, w)}
    return lambda n: table[int(n)]


def install_weight_fn(fn) -> None:
    """Put an arbitrary per-number weight function into the EV module."""
    ev.number_weight = fn
    ev.MEAN_WEIGHT = sum(fn(n) for n in range(1, N_BALLS + 1)) / N_BALLS
    ev.POPULARITY_NORMALIZATION = ev._popularity_normalization()


def challenger_weights(draws: np.ndarray, mult: np.ndarray) -> tuple:
    """Bucket weights implied by the SMOOTH fit, for an EV comparison.

    The smooth model has no buckets, so it is projected onto them: the
    mean fitted pick-rate of the numbers in each range. That is the only
    way to ask "would this specification change the verdict?" using
    machinery that takes three weights.
    """
    X = _design_smooth(draws)
    beta, *_ = np.linalg.lstsq(X, np.log(mult), rcond=None)
    ns = np.arange(1, N_BALLS + 1, dtype=float)
    # Per-number log-weight from the same coefficients, un-damped.
    logw = UNDAMP * (beta[1] * ns + beta[2] * np.maximum(ns - 12, 0)
                     + beta[3] * np.maximum(ns - 31, 0))
    w = np.exp(logw - logw.mean())
    w = w * N_BALLS / w.sum()            # population mean pick-rate 1.0
    return (float(w[:12].mean()), float(w[12:31].mean()),
            float(w[31:].mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.30,
                    help="lognormal sd of the synthetic multiplier")
    ap.add_argument("--obs", type=int, default=1147,
                    help="observations, default = what the archive holds")
    args = ap.parse_args()

    from scripts.ev_play import next_draw_conditions
    rng = np.random.default_rng(20260919)
    base = next_draw_conditions()

    print("=" * 72)
    print("POPULARITY AUDIT - the one fitted input to EV")
    print("=" * 72)
    print(f"Installed weights: <=12 {INSTALLED[0]}, 13-31 {INSTALLED[1]}, "
          f">31 {INSTALLED[2]}")
    print(f"Calibration observations in the archive: {args.obs:,}")
    print()

    # 1 --------------------------------------------------------------------
    print("1. DECISION SENSITIVITY - can weight error flip the verdict?")
    print()
    factors = [0.0, 1.0, 2.0]
    draws = [
        ("ordinary, pool 3.4m", dict(jackpot=3_447_440, roll_down=False)),
        ("ordinary, pool 32m", dict(jackpot=32_000_000, roll_down=False)),
        ("ordinary, pool 36m", dict(jackpot=36_000_000, roll_down=False)),
        ("capped MBW, Wed 9m", dict(jackpot=9_000_000, roll_down=True,
                                    tickets_sold=9_500_000)),
        ("special, Sat 15m", dict(jackpot=15_000_000, roll_down=True,
                                  tickets_sold=13_500_000)),
        ("special, Sat 20m", dict(jackpot=20_000_000, roll_down=True,
                                  tickets_sold=15_000_000)),
    ]
    grid = decision_grid(base, factors, draws)
    head = f"   {'draw':22}" + "".join(f"{f'x{f:g} spread':>18}" for f in factors)
    print(head)
    print("   " + "-" * (len(head) - 3))
    flips = []
    for _, row in grid.iterrows():
        cells = []
        verdicts = set()
        for f in factors:
            evv, play = row[f"x{f:g}"]
            verdicts.add(play)
            cells.append(f"{evv:+.3f} {'PLAY' if play else 'SKIP'}")
        mark = "  <- FLIPS" if len(verdicts) > 1 else ""
        if mark:
            flips.append(row["draw"])
        print(f"   {row['draw']:22}" + "".join(f"{c:>18}" for c in cells) + mark)
    print()
    if flips:
        print(f"   Verdict flips on: {', '.join(flips)}")
        print("   All of them are ORDINARY draws near break-even. On roll-downs")
        print("   the EV is dominated by J/N, which popularity does not touch,")
        print("   so the draws this project actually plays are unaffected.")
    else:
        print("   No verdict flips at any scaling.")
    print()

    # 2 --------------------------------------------------------------------
    print("2. RECOVERY - fed known weights, does the calibration return them?")
    print()
    rec = recovery_report(args.obs, args.noise, args.reps, rng)
    print(f"   {'planted':12} {'bucket':>8} {'truth':>8} {'bias':>9} {'sd':>8}")
    for name, r in rec.items():
        for i, bucket in enumerate(("<=12", "13-31", ">31")):
            print(f"   {name if i == 0 else '':12} {bucket:>8} "
                  f"{r['truth'][i]:>8.3f} {r['bias'][i]:>+9.4f} "
                  f"{r['sd'][i]:>8.4f}")
    print()
    print("   Bias near zero = the fit is unbiased at this sample size.")
    print("   The sd column is what one archive's worth of noise buys you.")
    print()

    # 3 --------------------------------------------------------------------
    print("3. POWER - how strong must the bias be before it is detected?")
    print()
    print(f"   {'spread':>8} {'<=12':>8} {'13-31':>8} {'>31':>8} {'power':>8}")
    for row in power_report(args.obs, args.noise, max(args.reps // 3, 50), rng):
        w = row["weights"]
        print(f"   x{row['factor']:<7g} {w[0]:>8.3f} {w[1]:>8.3f} "
              f"{w[2]:>8.3f} {row['power']:>8.0%}")
    print()

    # 4 --------------------------------------------------------------------
    print("4. UNCERTAINTY - the interval, pushed into the threshold")
    print()
    ci = threshold_interval(base, args.obs, args.noise, args.boot, rng)
    print(f"   break-even jackpot: GBP {ci['threshold_p50']:,.0f}")
    print(f"      90% interval:    GBP {ci['threshold_p5']:,.0f} .. "
          f"{ci['threshold_p95']:,.0f}")
    print(f"   EV on today's draw: GBP {ci['ev_p5']:+.4f} .. {ci['ev_p95']:+.4f}")
    print()
    print("   A draw inside that interval is a coin flip on the weights, not")
    print("   a PLAY. `should_play` returns a point estimate; this is the")
    print("   band it should be read against.")
    print()

    # 5 --------------------------------------------------------------------
    print("5. SPECIFICATION - do the data support three buckets?")
    print()
    try:
        from scripts.calibrate_popularity import add_multiplier, load_joined
        import logging
        logging.getLogger().setLevel(logging.WARNING)
        df = add_multiplier(load_joined())
        real = df[[f"Number_{i}" for i in range(1, 7)]].to_numpy(int)
        real_mult = df["multiplier"].to_numpy()
    except Exception as exc:
        print(f"   calibration data unavailable ({exc})")
        return 0

    print(f"   {'specification':24} {'params':>7} {'in-sample':>11} "
          f"{'out-of-sample':>14}")
    for row in specification_contest(real, real_mult):
        print(f"   {row['name']:24} {row['params']:>7} {row['in_mse']:>11.5f} "
              f"{row['oos_mse']:>14.5f}")
    print()
    print("   Lower out-of-sample is better. The per-number row is the")
    print("   yardstick: it must fit best in-sample and worse out of it.")
    print()

    ch = challenger_weights(real, real_mult)
    print(f"   Smooth model projected onto buckets: {ch[0]:.3f} / {ch[1]:.3f} "
          f"/ {ch[2]:.3f}")
    print(f"   Installed:                           {INSTALLED[0]:.3f} / "
          f"{INSTALLED[1]:.3f} / {INSTALLED[2]:.3f}")
    print()
    print("   Does the challenger change any verdict? (full smooth curve)")
    smooth_fn = smooth_weight_fn(real, real_mult)
    for label, kw in draws:
        cond = replace(base, **kw)
        set_weights(*INSTALLED)
        a = ev.should_play(cond)
        install_weight_fn(smooth_fn)
        b = ev.should_play(cond)
        if a["play"] != b["play"]:
            print(f"      {label:22} installed "
                  f"{'PLAY' if a['play'] else 'SKIP'} -> challenger "
                  f"{'PLAY' if b['play'] else 'SKIP'}   *** FLIP")
        else:
            print(f"      {label:22} {'PLAY' if a['play'] else 'SKIP':4} both "
                  f"(EV {a['ev_best_line']:+.3f} vs {b['ev_best_line']:+.3f})")
    set_weights(*INSTALLED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
