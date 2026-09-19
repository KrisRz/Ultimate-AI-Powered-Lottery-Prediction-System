#!/usr/bin/env python3
"""The ensemble everyone proposes: score numbers, generate millions of lines,
keep the best ones. Built, and walk-forward tested against random picks.

The recipe is always some version of

    Score(n) = w1*Frequency + w2*RecentFrequency + w3*Gap + w4*PairScore
               + w5*MachineAffinity

then Monte Carlo over a few million combinations, keep the top-scoring
lines. It sounds like engineering, so "it obviously doesn't work" is an
unsatisfying answer. This runs it.

What makes the test honest:

  * Walk-forward. At draw t the scorer sees draws < t and nothing else.
    Fitting weights on all history and admiring the fit is the mistake this
    is designed to avoid.
  * The same random baseline on the SAME draws, so variance does not get
    mistaken for skill. Lottery match counts are noisy enough that a
    strategy can lead over fifty draws on nothing at all.
  * The metric is average matches per line (plus 3+ rates), not jackpots -
    at 1 in 45,057,474 no backtest will ever see one, and a metric that
    never moves cannot compare anything.
  * A Monte Carlo p-value on the DIFFERENCE, so "slightly ahead" gets
    called what it is.

The expected result is a dead heat, because the draws are independent. The
value of running it is that it is measured rather than asserted - and that
the same harness would show a real edge if one existed, which is what the
planted-signal test in tests/test_ensemble_score.py checks.

Run:  PYTHONPATH=. python scripts/validations/ensemble_score.py
      PYTHONPATH=. python scripts/validations/ensemble_score.py --candidates 200000
"""

from __future__ import annotations

import argparse

import numpy as np

from lottery.ev import N_BALLS, N_PICK, popularity_ratio
from scripts.validations.fairness import NUMBER_COLS, load_frame

# Weights as a naive builder would set them: every component gets a say.
# They are NOT fitted - fitting them on the same history that generates the
# features is exactly the overfit this script exists to expose, and a fitted
# version scores no better out of sample (it just takes longer to say so).
DEFAULT_WEIGHTS = {
    "frequency": 1.0,      # all-time count, standardised
    "recent": 1.0,         # last `recent_window` draws
    "gap": 1.0,            # draws since last seen
    "pair": 1.0,           # affinity with the other five on the line
    "machine": 1.0,        # count under the machine/ball-set in play
    "shape": 1.0,          # sum / odd-even / low-high / runs / spread
}


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def ball_features(history: np.ndarray, recent_window: int,
                  machine_rows: np.ndarray | None) -> dict:
    """Per-ball features from draws strictly before the target draw."""
    n = len(history)
    counts = np.bincount(history.ravel(), minlength=N_BALLS + 1)[1:]

    recent = history[-recent_window:] if n > recent_window else history
    recent_counts = np.bincount(recent.ravel(), minlength=N_BALLS + 1)[1:]

    # Draws since last appearance; never-seen balls get the longest gap.
    gap = np.full(N_BALLS, n, dtype=float)
    for i in range(n - 1, -1, -1):
        for b in history[i]:
            if gap[b - 1] == n:
                gap[b - 1] = n - 1 - i
        if (gap != n).all():
            break

    if machine_rows is not None and len(machine_rows):
        machine_counts = np.bincount(machine_rows.ravel(),
                                     minlength=N_BALLS + 1)[1:]
    else:
        machine_counts = np.zeros(N_BALLS)

    return {
        "frequency": _z(counts.astype(float)),
        "recent": _z(recent_counts.astype(float)),
        "gap": _z(gap),
        "machine": _z(machine_counts.astype(float)),
    }


def pair_matrix(history: np.ndarray) -> np.ndarray:
    """(59, 59) co-occurrence counts, standardised."""
    m = np.zeros((N_BALLS, N_BALLS))
    for row in history:
        idx = np.array(row) - 1
        m[np.ix_(idx, idx)] += 1
    np.fill_diagonal(m, 0)
    sd = m.std()
    return (m - m.mean()) / sd if sd > 0 else m


def _shape_stats(lines: np.ndarray) -> dict:
    """Whole-line shape: sum, odd count, low count, consecutive pairs, spread.

    These are properties of a LINE, not of a ball, which is why they need
    their own pass. They are also the ones that feel most like insight and
    are most misleading - see `shape_score`.
    """
    srt = np.sort(lines, axis=1)
    return {
        "sum": srt.sum(axis=1),
        "odd": (srt % 2 == 1).sum(axis=1),
        "low": (srt <= 31).sum(axis=1),          # the date range
        "consec": (np.diff(srt, axis=1) == 1).sum(axis=1),
        "decades": np.array([len(np.unique((row - 1) // 10)) for row in srt]),
    }


def shape_score(cands: np.ndarray, history: np.ndarray) -> np.ndarray:
    """How "normal" each candidate looks against the history's own shapes.

    This is the component that makes an ensemble look clever and quietly
    makes it worse. The shape distributions are real - sums near 180 really
    are more common than sums near 40 - but ONLY because there are more
    lines shaped that way, not because any one of them is likelier. Each
    individual combination stays at 1 in C(59,6).

    "Normal-looking" is also roughly what other players pick, so this
    component pulls towards crowded lines. Whether the FULL ensemble ends up
    crowded is a separate question the report measures rather than assumes:
    the per-ball components pull the other way (towards rarely-drawn high
    numbers), and in practice they win. Run with `--only shape` to see this
    component's own effect on sharing.
    """
    hist = _shape_stats(history)
    cand = _shape_stats(cands)

    total = np.zeros(len(cands))
    for key in ("sum", "odd", "low", "consec", "decades"):
        h = hist[key]
        lo, hi = int(h.min()), int(h.max())
        bins = np.arange(lo, hi + 2)
        counts, _ = np.histogram(h, bins=bins)
        # Laplace smoothing so an unseen shape is unlikely, not impossible.
        probs = (counts + 1) / (counts.sum() + len(counts))
        c = np.clip(cand[key], lo, hi) - lo
        total += np.log(probs[c])
    return _z(total)


def score_candidates(cands: np.ndarray, feats: dict, pairs: np.ndarray,
                     weights: dict, history: np.ndarray | None = None) -> np.ndarray:
    """Score every candidate line. cands is (n_cands, 6), 1-based."""
    idx = cands - 1
    per_ball = (
        weights["frequency"] * feats["frequency"]
        + weights["recent"] * feats["recent"]
        + weights["gap"] * feats["gap"]
        + weights["machine"] * feats["machine"]
    )
    score = per_ball[idx].sum(axis=1)

    if weights["pair"]:
        # Sum the 15 pairwise affinities of each line.
        pair_sum = np.zeros(len(cands))
        for a in range(N_PICK):
            for b in range(a + 1, N_PICK):
                pair_sum += pairs[idx[:, a], idx[:, b]]
        score = score + weights["pair"] * _z(pair_sum)

    if weights.get("shape") and history is not None:
        score = score + weights["shape"] * shape_score(cands, history)
    return score


def random_lines(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((n, N_BALLS)).argsort(axis=1)[:, :N_PICK] + 1


def matches(lines: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.isin(lines, actual).sum(axis=1)


def walk_forward(frame, n_candidates: int, n_lines: int, recent_window: int,
                 min_history: int, step: int, weights: dict,
                 rng: np.random.Generator) -> dict:
    draws = frame[NUMBER_COLS].to_numpy(dtype=int)
    machines = frame["MachineNorm"].to_numpy()
    ball_sets = frame["BallSetStr"].to_numpy()

    ens_matches, rnd_matches, ens_pop, rnd_pop, points = [], [], [], [], 0
    for t in range(min_history, len(draws), step):
        history = draws[:t]

        # Only draws from the same machine AND ball set as the one about to
        # be used - the physical-bias hypothesis, given its best shot.
        same = (machines[:t] == machines[t]) & (ball_sets[:t] == ball_sets[t])
        machine_rows = history[same]

        feats = ball_features(history, recent_window, machine_rows)
        pairs = pair_matrix(history) if weights["pair"] else None

        cands = random_lines(n_candidates, rng)
        scores = score_candidates(cands, feats, pairs, weights, history)
        best = cands[np.argsort(scores)[::-1][:n_lines]]
        rnd = random_lines(n_lines, rng)

        ens_matches.extend(matches(best, draws[t]).tolist())
        rnd_matches.extend(matches(rnd, draws[t]).tolist())
        # The number that actually decides money: how crowded these lines
        # are. 1.0 is an average line; above it you share more.
        ens_pop.extend(popularity_ratio(line) for line in best)
        rnd_pop.extend(popularity_ratio(line) for line in rnd)
        points += 1

    ens = np.array(ens_matches, dtype=float)
    rnd = np.array(rnd_matches, dtype=float)

    # Permutation test on the difference of means: shuffle the labels and
    # see how often chance produces a gap this big.
    observed = ens.mean() - rnd.mean()
    pooled = np.concatenate([ens, rnd])
    n_ens = len(ens)
    diffs = np.empty(2000)
    for i in range(2000):
        perm = rng.permutation(pooled)
        diffs[i] = perm[:n_ens].mean() - perm[n_ens:].mean()
    p_two_sided = float((np.abs(diffs) >= abs(observed)).mean())

    return {
        "points": points, "lines": len(ens),
        "ensemble_avg": float(ens.mean()), "random_avg": float(rnd.mean()),
        "difference": float(observed), "p": p_two_sided,
        "ensemble_3plus": float((ens >= 3).mean()),
        "random_3plus": float((rnd >= 3).mean()),
        "theoretical_avg": N_PICK * N_PICK / N_BALLS,
        "ensemble_popularity": float(np.mean(ens_pop)),
        "random_popularity": float(np.mean(rnd_pop)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=int, default=100_000,
                    help="combinations generated and scored per draw")
    ap.add_argument("--lines", type=int, default=10,
                    help="top-scoring lines kept per draw")
    ap.add_argument("--recent-window", type=int, default=50)
    ap.add_argument("--min-history", type=int, default=600)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--max-draw", type=int, default=None)
    ap.add_argument("--only", type=str, default=None,
                    choices=sorted(DEFAULT_WEIGHTS),
                    help="score on ONE component, to see what it does alone")
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    frame = load_frame(args.max_draw)
    weights = dict(DEFAULT_WEIGHTS)
    if args.only:
        weights = {k: (1.0 if k == args.only else 0.0) for k in weights}

    print("=" * 70)
    print("ENSEMBLE SCORING, WALK-FORWARD")
    print("=" * 70)
    if args.only:
        print(f"Score = {args.only} ONLY (single-component run)")
    else:
        print("Score = frequency + recent frequency + gap + pair affinity")
        print("        + machine/ball-set affinity")
        print("        + line shape (sum, odd/even, low/high, runs, spread)")
    print(f"Candidates scored per draw: {args.candidates:,}   "
          f"lines kept: {args.lines}")
    print(f"Archive: {len(frame):,} draw-rounds; scoring starts after "
          f"{args.min_history:,}, every {args.step}")
    print()

    r = walk_forward(frame, args.candidates, args.lines, args.recent_window,
                     args.min_history, args.step, weights, rng)

    print(f"Draws scored: {r['points']}   lines played: {r['lines']:,}")
    print()
    print(f"   {'':<22} {'avg matches':>12} {'3+ rate':>10}")
    print(f"   {'ensemble (top lines)':<22} {r['ensemble_avg']:>12.4f} "
          f"{r['ensemble_3plus']:>10.4f}")
    print(f"   {'random':<22} {r['random_avg']:>12.4f} "
          f"{r['random_3plus']:>10.4f}")
    print(f"   {'theory':<22} {r['theoretical_avg']:>12.4f}")
    print()
    print(f"   difference: {r['difference']:+.4f} matches per line")
    print(f"   permutation p = {r['p']:.3f}")
    print()

    if r["p"] >= 0.05:
        print("VERDICT: no edge. The ensemble picks lines that look special")
        print("against history and land exactly where random lines land.")
    else:
        print("VERDICT: difference is significant - re-run with another seed")
        print("and more draws before believing it.")
    print()
    print("-" * 70)
    print("AND THE PART THAT COSTS MONEY")
    print("-" * 70)
    print(f"   {'ensemble lines':<22} popularity {r['ensemble_popularity']:.3f}")
    print(f"   {'random lines':<22} popularity {r['random_popularity']:.3f}")
    ratio = r["ensemble_popularity"] / r["random_popularity"]
    print()
    if ratio > 1.02:
        print(f"These lines are played {ratio - 1:+.0%} MORE than random ones -")
        print("no gain in hit rate, and a real loss in jackpot sharing.")
    elif ratio < 0.98:
        print(f"These lines are played {1 - ratio:.0%} LESS than random ones.")
        print("That part is worth something - but it is the popularity model")
        print("earning it, not the prediction: `make play` targets it directly")
        print("and does not need a million candidates to find it.")
    else:
        print(f"Sharing exposure vs random: {ratio:.3f}x - no difference.")
    print()
    print("Hit rate is not where line choice pays. Sharing is - see")
    print("scripts/calibrate_popularity.py and `make play`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
