#!/usr/bin/env python3
"""Is the machine fair? Five tests on the 59-ball era, and what they buy you.

This answers the question every lottery dataset provokes - "with this much
history, surely something is predictable?" - by running the tests that would
detect it and reporting what they find, including when they find nothing.
Nothing is the expected result, and a report that cannot print "no signal"
is not a test.

The tests, in the order a sceptic would ask for them:

  1. Chi-square uniformity - does every ball fall equally often?
  2. Per-ball z-scores - which balls deviate most, and is the WORST one
     surprising once you account for having looked at 59 of them?
  3. Gap analysis - do "overdue" balls come back sooner? This is the
     gambler's fallacy stated as a testable claim, so it gets tested.
  4. Draw-to-draw repeats - does draw t carry information about draw t+1?
     Under independence the overlap is hypergeometric; we compare.
  5. Pair co-occurrence - does any PAIR appear together more than chance,
     judged against the 1,711 pairs we searched, not against one pair.

Tests 2 and 5 are where naive analyses go wrong: searching 59 balls or 1,711
pairs for the most extreme one and then quoting its p-value is a multiple
comparisons error, and it manufactures "signal" from clean random data every
time. Both are judged by Monte Carlo against the same search over simulated
fair draws, which is the only honest yardstick.

Scope: the 59-ball era only (draw 2066, 2015-10-10, onward), BOTH rounds
since the two-round redesign - a drawn ball is a drawn ball whichever round
produced it. Earlier eras had 49 balls and would contaminate every count.

Run:  PYTHONPATH=. python scripts/validations/fairness.py
      PYTHONPATH=. python scripts/validations/fairness.py --sims 50000
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lottery.ev import N_BALLS, N_PICK, TOTAL_COMBOS

# The 59-ball era starts here. Before it the game drew 6 from 49, so mixing
# the eras would show a "deficit" on balls 50-59 that is a rule change, not
# a bias - the single most common way to fake a finding in this data.
ERA_FIRST_DRAW = 2066
HISTORY_FILE = Path("data/lotto_full_history.csv")
NUMBER_COLS = [f"Number_{i}" for i in range(1, N_PICK + 1)]

P_BALL = N_PICK / N_BALLS  # 6/59: chance a given ball appears in a given draw


def load_frame(max_draw: int | None = None) -> pd.DataFrame:
    """The 59-ball era as a frame, both rounds, machine names normalised.

    `max_draw` pins the slice for tests; the collector keeps appending to
    this file, so anything asserting on exact counts must say where it stops.
    """
    df = pd.read_csv(HISTORY_FILE)
    df = df[df["DrawNumber"] >= ERA_FIRST_DRAW]
    if max_draw is not None:
        df = df[df["DrawNumber"] <= max_draw]
    df = df.sort_values(["DrawNumber", "Round"], kind="stable").copy()
    df["MachineNorm"] = df["Machine"].map(normalise_machine)
    df["BallSetStr"] = df["Ball Set"].astype(str)
    return df


def load_draws(max_draw: int | None = None) -> np.ndarray:
    """(n_draws, 6) array of drawn balls, 59-ball era, both rounds."""
    return load_frame(max_draw)[NUMBER_COLS].to_numpy(dtype=int)


def ball_counts(draws: np.ndarray) -> np.ndarray:
    """counts[i] = how often ball i+1 was drawn."""
    return np.bincount(draws.ravel(), minlength=N_BALLS + 1)[1:]


def chi_square_uniformity(draws: np.ndarray, n_sims: int = 0,
                          rng: np.random.Generator | None = None) -> dict:
    """Test 1: are all 59 balls equally likely?

    The textbook chi-square p-value is WRONG here, and knowing why matters.
    Chi-square assumes independent trials, but a draw takes 6 balls WITHOUT
    replacement: once ball 7 is out, the other 58 compete for five slots, so
    the counts are negatively correlated. That shrinks their variance below
    the multinomial assumption, which shrinks the statistic, which inflates
    the p-value. On this archive it reads 0.984 - "suspiciously uniform" -
    when the honest reading is an ordinary 0.7-ish.

    So `p` is reported as the conservative analytic value and `p_mc`, when
    simulations are asked for, as the real one: the same statistic measured
    against draws generated the same way the machine generates them.
    """
    counts = ball_counts(draws)
    expected = counts.sum() / N_BALLS
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    dof = N_BALLS - 1
    out = {
        "chi2": chi2, "dof": dof, "expected": expected,
        "p": float(stats.chi2.sf(chi2, dof)),
        "p_mc": None,
    }
    if n_sims and rng is not None:
        n = len(draws)
        sims = np.empty(n_sims)
        for i in range(n_sims):
            sim = rng.random((n, N_BALLS)).argsort(axis=1)[:, :N_PICK]
            c = np.bincount(sim.ravel(), minlength=N_BALLS)
            sims[i] = ((c - expected) ** 2 / expected).sum()
        out["p_mc"] = float((sims >= chi2).mean())
        out["sim_median_chi2"] = float(np.median(sims))
    return out


def extreme_ball(draws: np.ndarray, n_sims: int, rng: np.random.Generator) -> dict:
    """Test 2: the most deviant ball, judged against having searched all 59.

    A |z| of 2.5 on one pre-chosen ball is notable. The LARGEST |z| among 59
    is not - that is a maximum of 59 draws from a roughly standard normal, and
    it lands near 2.5 routinely. So the Monte Carlo repeats the same search
    (take the max |z| over 59 balls) on fair simulated data and asks how often
    it beats what we saw.
    """
    n_draws = len(draws)
    counts = ball_counts(draws)
    sd = np.sqrt(n_draws * P_BALL * (1 - P_BALL))
    z = (counts - n_draws * P_BALL) / sd
    observed_max = float(np.abs(z).max())

    sim_max = np.empty(n_sims)
    for i in range(n_sims):
        sim = rng.random((n_draws, N_BALLS)).argsort(axis=1)[:, :N_PICK]
        sim_counts = np.bincount(sim.ravel(), minlength=N_BALLS)
        sim_max[i] = np.abs((sim_counts - n_draws * P_BALL) / sd).max()

    order = np.argsort(z)
    return {
        "z": z,
        "hottest": [(int(b + 1), float(z[b])) for b in order[::-1][:5]],
        "coldest": [(int(b + 1), float(z[b])) for b in order[:5]],
        "observed_max_abs_z": observed_max,
        "p_corrected": float((sim_max >= observed_max).mean()),
        "sim_median_max": float(np.median(sim_max)),
    }


def gap_analysis(draws: np.ndarray) -> dict:
    """Test 3: are "overdue" balls more likely to come back?

    For every (draw, ball) we know how many draws it had been missing. If the
    gambler's fallacy were true, the hit rate would RISE with that gap. Under
    independence it is flat at 6/59 in every bucket.
    """
    n_draws = len(draws)
    present = np.zeros((n_draws, N_BALLS), dtype=bool)
    for i, row in enumerate(draws):
        present[i, row - 1] = True

    buckets = [(0, 4), (5, 9), (10, 19), (20, 10 ** 9)]
    hits = defaultdict(int)
    opportunities = defaultdict(int)
    last_seen = np.full(N_BALLS, -1)

    for i in range(n_draws):
        for b in range(N_BALLS):
            if last_seen[b] < 0:          # never seen yet - no gap defined
                continue
            gap = i - last_seen[b] - 1
            for lo, hi in buckets:
                if lo <= gap <= hi:
                    opportunities[(lo, hi)] += 1
                    if present[i, b]:
                        hits[(lo, hi)] += 1
                    break
        for b in np.where(present[i])[0]:
            last_seen[b] = i

    rows, chi2 = [], 0.0
    for lo, hi in buckets:
        n = opportunities[(lo, hi)]
        if n == 0:
            continue
        k = hits[(lo, hi)]
        exp = n * P_BALL
        chi2 += (k - exp) ** 2 / exp + ((n - k) - (n - exp)) ** 2 / (n - exp)
        rows.append({
            "bucket": f"{lo}-{hi if hi < 10 ** 9 else '+'}",
            "n": n, "hits": k, "rate": k / n, "expected": P_BALL,
        })
    return {"rows": rows, "chi2": chi2, "dof": len(rows) - 1,
            "p": float(stats.chi2.sf(chi2, max(len(rows) - 1, 1)))}


def repeat_analysis(draws: np.ndarray) -> dict:
    """Test 4: does draw t predict draw t+1?

    The clean test is the overlap between consecutive draws. Under
    independence that is hypergeometric: 6 "successes" among 59, drawing 6.
    Any memory in the machine - or any dependence introduced by the draw
    procedure - shows up as the observed distribution pulling away from it.
    """
    overlaps = [len(set(draws[i]) & set(draws[i + 1])) for i in range(len(draws) - 1)]
    n = len(overlaps)
    observed = np.bincount(overlaps, minlength=N_PICK + 1)
    expected = np.array([
        stats.hypergeom.pmf(k, N_BALLS, N_PICK, N_PICK) * n
        for k in range(N_PICK + 1)
    ])

    # Pool the sparse tail so chi-square stays valid (expected >= 5).
    obs, exp, labels = [], [], []
    acc_o = acc_e = 0.0
    for k in range(N_PICK + 1):
        acc_o += observed[k]
        acc_e += expected[k]
        if acc_e >= 5 and k < N_PICK:
            obs.append(acc_o); exp.append(acc_e); labels.append(str(k))
            acc_o = acc_e = 0.0
        elif k == N_PICK:
            if acc_e < 5 and obs:
                obs[-1] += acc_o; exp[-1] += acc_e; labels[-1] += "+"
            else:
                obs.append(acc_o); exp.append(acc_e); labels.append(f"{k}+")
    obs, exp = np.array(obs), np.array(exp)
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    dof = len(obs) - 1
    return {
        "n_pairs": n, "labels": labels, "observed": obs, "expected": exp,
        "mean_overlap": float(np.mean(overlaps)),
        "expected_mean": N_PICK * P_BALL,
        "chi2": chi2, "dof": dof, "p": float(stats.chi2.sf(chi2, dof)),
    }


def pair_analysis(draws: np.ndarray, n_sims: int, rng: np.random.Generator) -> dict:
    """Test 5: does any PAIR of balls favour each other?

    1,711 pairs are searched, so the maximum count among them is compared
    with the maximum count from the same search on fair simulated draws -
    never with the expectation for one pair chosen in advance.
    """
    n_draws = len(draws)
    pair_counts = defaultdict(int)
    for row in draws:
        for a, b in combinations(sorted(row), 2):
            pair_counts[(a, b)] += 1
    top = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:5]
    observed_max = top[0][1]
    n_pairs = N_BALLS * (N_BALLS - 1) // 2
    expected = n_draws * (N_PICK * (N_PICK - 1)) / (N_BALLS * (N_BALLS - 1))

    sim_max = np.empty(n_sims)
    for i in range(n_sims):
        sim = rng.random((n_draws, N_BALLS)).argsort(axis=1)[:, :N_PICK] + 1
        counts = defaultdict(int)
        for row in sim:
            for a, b in combinations(sorted(row), 2):
                counts[(a, b)] += 1
        sim_max[i] = max(counts.values())

    return {
        "n_pairs": n_pairs, "expected_per_pair": expected,
        "top": [((int(a), int(b)), c) for (a, b), c in top],
        "observed_max": int(observed_max),
        "p_corrected": float((sim_max >= observed_max).mean()),
        "sim_median_max": float(np.median(sim_max)),
    }


def normalise_machine(name: str) -> str:
    """"Lotto 4" and "Lotto4" are one machine.

    The feed switched format at draw 3191 and started writing the name without
    the space. Left alone it splits every new machine into two entities of
    ~15 draws each, which is how a physical-bias test gets quietly starved of
    the data it needs. Nothing in the EV model reads this column, so the fix
    lives here rather than in the collector.
    """
    return str(name).replace(" ", "")


def detectable_bias(n_draws: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest per-ball bias this many draws could detect, as a multiplier.

    Honest reporting of a null result needs this number. "No signal in 250
    draws" means "no bias big enough for 250 draws to see", and a reader is
    entitled to know where that line sits. Two-sided test on one ball's
    appearance count.
    """
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    sd = np.sqrt(P_BALL * (1 - P_BALL) / n_draws)
    return 1.0 + (z_a + z_b) * sd / P_BALL


def machine_analysis(df: pd.DataFrame, key: str, n_sims: int,
                     rng: np.random.Generator, min_draws: int = 60) -> dict:
    """Test 6: does any machine (or ball set) favour particular numbers?

    This is the only hypothesis on the list with a physical mechanism behind
    it - a worn ball, an unbalanced drum - so it deserves a real test rather
    than dismissal. It is also the one most starved of data: the draws split
    across groups, and each group gets a fraction of the archive.

    Judged the same way as the other searches: the WORST group among those
    tested, against the worst group from the same search on fair data.
    """
    groups, rows = [], []
    for name, sub in df.groupby(key):
        draws = sub[NUMBER_COLS].to_numpy(dtype=int)
        if len(draws) < min_draws:
            continue
        r = chi_square_uniformity(draws)
        groups.append((str(name), len(draws), r["chi2"], r["p"]))
    if not groups:
        return {"rows": [], "skipped": True}

    sizes = [n for _, n, _, _ in groups]
    observed_min_p = min(p for _, _, _, p in groups)

    sim_min_p = np.empty(n_sims)
    for i in range(n_sims):
        worst = 1.0
        for n in sizes:
            sim = rng.random((n, N_BALLS)).argsort(axis=1)[:, :N_PICK] + 1
            worst = min(worst, chi_square_uniformity(sim)["p"])
        sim_min_p[i] = worst

    for name, n, chi2, p in sorted(groups, key=lambda g: g[3]):
        rows.append({"name": name, "n": n, "chi2": chi2, "p": p,
                     "detectable": detectable_bias(n)})
    return {
        "rows": rows, "skipped": False, "n_groups": len(groups),
        "observed_min_p": observed_min_p,
        "p_corrected": float((sim_min_p <= observed_min_p).mean()),
    }


def _verdict(p: float, alpha: float = 0.05) -> str:
    return "no signal" if p >= alpha else "SIGNAL - investigate"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=2000,
                    help="Monte Carlo replications for the corrected tests")
    ap.add_argument("--max-draw", type=int, default=None,
                    help="Pin the slice at this draw number (tests use this)")
    args = ap.parse_args()

    rng = np.random.default_rng(20260919)
    frame = load_frame(args.max_draw)
    draws = frame[NUMBER_COLS].to_numpy(dtype=int)
    n = len(draws)

    print("=" * 70)
    print("IS THE MACHINE FAIR?  59-ball era, both rounds")
    print("=" * 70)
    print(f"Draw-rounds analysed: {n:,}   balls drawn: {n * N_PICK:,}   "
          f"Monte Carlo: {args.sims:,} sims")
    print()

    # 1 -------------------------------------------------------------------
    r = chi_square_uniformity(draws, args.sims, rng)
    print("1. Chi-square uniformity (are all 59 balls equally likely?)")
    print(f"   expected {r['expected']:.1f} appearances per ball")
    print(f"   chi2 {r['chi2']:.1f} on {r['dof']} df")
    print(f"   textbook p = {r['p']:.3f}  <- WRONG here: chi-square assumes")
    print("      independent trials, but a draw takes 6 balls without")
    print("      replacement, which shrinks the statistic and inflates p")
    print(f"   Monte Carlo p = {r['p_mc']:.3f}  (typical chi2 on fair draws: "
          f"{r['sim_median_chi2']:.1f})")
    print(f"   -> {_verdict(r['p_mc'])}")
    print()

    # 2 -------------------------------------------------------------------
    r = extreme_ball(draws, args.sims, rng)
    print("2. Most deviant ball (corrected for searching all 59)")
    print("   hottest: " + ", ".join(f"{b} (z{z:+.2f})" for b, z in r["hottest"]))
    print("   coldest: " + ", ".join(f"{b} (z{z:+.2f})" for b, z in r["coldest"]))
    print(f"   largest |z| observed: {r['observed_max_abs_z']:.2f}")
    print(f"   typical largest |z| on FAIR data: {r['sim_median_max']:.2f}")
    print(f"   corrected p = {r['p_corrected']:.3f}   -> "
          f"{_verdict(r['p_corrected'])}")
    print()

    # 3 -------------------------------------------------------------------
    r = gap_analysis(draws)
    print("3. Gap analysis (do 'overdue' balls come back sooner?)")
    print(f"   {'gap':>8} {'chances':>10} {'hit rate':>10} {'expected':>10}")
    for row in r["rows"]:
        print(f"   {row['bucket']:>8} {row['n']:>10,} {row['rate']:>10.4f} "
              f"{row['expected']:>10.4f}")
    print(f"   chi2 {r['chi2']:.1f} on {r['dof']} df   p = {r['p']:.3f}"
          f"   -> {_verdict(r['p'])}")
    print()

    # 4 -------------------------------------------------------------------
    r = repeat_analysis(draws)
    print("4. Draw-to-draw repeats (does draw t predict draw t+1?)")
    print(f"   mean overlap {r['mean_overlap']:.4f} balls "
          f"(independence says {r['expected_mean']:.4f})")
    print(f"   {'overlap':>8} {'observed':>10} {'expected':>10}")
    for lab, o, e in zip(r["labels"], r["observed"], r["expected"]):
        print(f"   {lab:>8} {o:>10.0f} {e:>10.1f}")
    print(f"   chi2 {r['chi2']:.1f} on {r['dof']} df   p = {r['p']:.3f}"
          f"   -> {_verdict(r['p'])}")
    print()

    # 5 -------------------------------------------------------------------
    r = pair_analysis(draws, args.sims, rng)
    print(f"5. Pair co-occurrence ({r['n_pairs']:,} pairs searched)")
    print(f"   expected per pair: {r['expected_per_pair']:.1f}")
    print("   top: " + ", ".join(f"{a}+{b} ({c}x)" for (a, b), c in r["top"]))
    print(f"   best pair observed: {r['observed_max']}x")
    print(f"   typical best pair on FAIR data: {r['sim_median_max']:.1f}x")
    print(f"   corrected p = {r['p_corrected']:.3f}   -> "
          f"{_verdict(r['p_corrected'])}")
    print()

    # 6 -------------------------------------------------------------------
    # Fewer sims here: each replication re-simulates every group.
    mb_sims = max(args.sims // 10, 50)
    for key, label in (("MachineNorm", "machine"), ("BallSetStr", "ball set")):
        r = machine_analysis(frame, key, mb_sims, rng)
        print(f"6{'ab'[key == 'BallSetStr']}. Per {label} "
              f"(the only hypothesis with a physical mechanism)")
        if r["skipped"]:
            print("   no group has enough draws to test")
            print()
            continue
        print(f"   {'group':>12} {'draws':>7} {'chi2':>8} {'p':>7} "
              f"{'can detect':>11}")
        for row in r["rows"][:6]:
            print(f"   {row['name']:>12} {row['n']:>7,} {row['chi2']:>8.1f} "
                  f"{row['p']:>7.3f} {row['detectable']:>10.0%}")
        print(f"   {r['n_groups']} groups tested; best p = "
              f"{r['observed_min_p']:.3f}")
        print(f"   corrected p = {r['p_corrected']:.3f}   -> "
              f"{_verdict(r['p_corrected'])}")
        print()

    print("=" * 70)
    print("WHAT THIS BUYS YOU")
    print("=" * 70)
    print("Nothing, for picking numbers - and that is the finding, not a")
    print(f"failure of the tests. Every combination stays at 1 in "
          f"{TOTAL_COMBOS:,}.")
    print()

    # The limit of the null result, stated rather than implied. A test that
    # cannot see the bias that would matter has not cleared the hypothesis -
    # it has only bounded it, and the bound belongs in the output.
    biggest = detectable_bias(n)
    needed = (1 / 0.432) ** (1 / N_PICK)   # today's return per GBP staked
    print("The honest limit of all this: with "
          f"{n:,} draw-rounds the sharpest of these")
    print(f"tests would only catch a ball running {biggest - 1:.0%} hot. Turning "
          "an ordinary")
    print(f"draw profitable needs about {needed - 1:.0%} on each of your six - "
          "smaller than")
    print("what the data could show. So this is a bound, not a clearance: it")
    print("rules out a gross fault, never a subtle one. And a subtle one would")
    print("still have to be found among 59 balls before it paid anything,")
    print("which is the search that tests 2 and 5 show manufactures ghosts.")
    print()
    print("What DOES move money is on the other side of the ticket: how many")
    print("people share a jackpot you win. That is measured, it is large, and")
    print("it is what lottery/ev.py already prices -")
    print("  scripts/calibrate_popularity.py, and `make play` for the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
