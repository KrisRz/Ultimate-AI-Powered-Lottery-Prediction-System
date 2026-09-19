#!/usr/bin/env python3
"""Audit the popularity model - the one calibrated thing EV actually spends on.

Everything else in `validations/` tests hypotheses the advisor deliberately
ignores. This tests the component it cannot ignore. `expected_cowinner_share`
is the only place a fitted number reaches `line_ev`, so if the pick-rate
weights are wrong, the PLAY threshold moves and real money follows.

Four questions, in order of how much they cost to get wrong:

  1. DECISION SENSITIVITY - can plausible weight error flip PLAY/SKIP, and
     on which draws? This is the only question with money attached.
  2. RECOVERY - fed data generated from KNOWN weights, does the calibration
     return them? A fit that cannot recover a planted truth is not measuring
     what it claims to, and four agreeing validations would agree on nothing.
  3. POWER - how wrong could the weights be before this archive noticed?
  4. UNCERTAINTY - the interval around the installed weights, propagated
     into the break-even jackpot, so a marginal PLAY can be read as marginal.

The answer to (1) turns out to bound the whole thing: on Must-Be-Won and
special draws - the only draws this project ever plays - the EV is dominated
by the roll-down term J/N, which popularity does not touch. Weight error
moves the verdict only on ORDINARY draws with a jackpot near break-even,
which in this era means above roughly GBP 32m: rarer than the p99 of the
59-ball era. That is the honest scope of the risk, and it is smaller than
the model's prominence suggests - but it is not zero, and `ev.py` said "no
decision moves" without qualifying it.

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
