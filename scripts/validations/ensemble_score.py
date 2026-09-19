#!/usr/bin/env python3
"""The ensemble everyone proposes: score numbers, generate millions of lines,
keep the best ones. Built, and walk-forward tested against random picks.

The recipe is always some version of

    Score(C) = w1*Frequency + w2*RecentFrequency + w3*Gap + w4*PairScore
               + w5*TripleScore + w6*MachineAffinity + w7*Structure

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
  * A percentile against whole RANDOM STRATEGIES run through the same
    backtest. This is the benchmark that settles arguments: a run can clear
    p < 0.05 against the mean while thousands of coin-flippers match it,
    and only the percentile shows that. One run here read p = 0.045 and sat
    at the 95th percentile - i.e. 5,000 random strategies in 100,000 did as
    well.
  * `--nested`: weights fitted on a VALIDATION slice, then frozen before
    the TEST slice is read. Weights are 1.0 by default and never fitted, so
    the default run has no leak to begin with; the flag exists to show that
    even weights chosen to win do not carry to unseen draws.
  * Triples are regularised, not tabulated. C(59,3) = 32,509 cells against
    ~20 observations per draw means the average cell holds half a sighting,
    so raw triple counts are pure overfit; `triple_matrix` needs both a
    minimum count and shrinkage before a cell can move the score.

The expected result is a dead heat, because the draws are independent. The
value of running it is that it is measured rather than asserted - and that
the same harness would show a real edge if one existed, which is what the
planted-signal test in tests/test_ensemble_score.py checks.

Run:  PYTHONPATH=. python scripts/validations/ensemble_score.py
      PYTHONPATH=. python scripts/validations/ensemble_score.py --candidates 200000
"""

from __future__ import annotations

import argparse
from math import comb

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
    "triple": 1.0,         # regularised C(59,3) affinity
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


def triple_matrix(history: np.ndarray, min_obs: int = 3,
                  shrink: float = 20.0) -> dict:
    """Regularised triple scores: C(59,3) = 32,509 cells over ~1,000 draws.

    Raw triple counts are the purest overfit available in this data - each
    draw contributes only C(6,3)=20 triples, so the average cell holds well
    under one observation and the maximum is noise by construction. Two
    guards:

      * `min_obs` - a triple seen fewer times than this scores 0 rather than
        contributing its accident.
      * `shrink` - empirical-Bayes shrinkage towards the expected count,
        (obs - exp) / (obs + shrink), so a cell needs both an excess AND
        support to move the score.

    Returned as a dict keyed by sorted triple; dense storage would be
    32,509 floats for a signal that does not exist.
    """
    from itertools import combinations as _c

    counts: dict[tuple, int] = {}
    for row in history:
        for t in _c(sorted(row), 3):
            counts[t] = counts.get(t, 0) + 1

    n = len(history)
    expected = n * comb(N_PICK, 3) / comb(N_BALLS, 3)
    return {
        t: (c - expected) / (c + shrink)
        for t, c in counts.items() if c >= min_obs
    }


def triple_score(cands: np.ndarray, triples: dict) -> np.ndarray:
    """Sum the 20 regularised triple scores of each candidate line."""
    from itertools import combinations as _c

    out = np.zeros(len(cands))
    if not triples:
        return out
    idx = list(_c(range(N_PICK), 3))
    srt = np.sort(cands, axis=1)
    for i, row in enumerate(srt):
        s = 0.0
        for a, b, c in idx:
            s += triples.get((int(row[a]), int(row[b]), int(row[c])), 0.0)
        out[i] = s
    return _z(out)


def random_strategy_percentile(ens_avg: float, n_draws: int, n_lines: int,
                               n_strategies: int, rng: np.random.Generator,
                               pool_size: int | None = None) -> dict:
    """Where does the ensemble sit among N complete random strategies?

    A p-value against the mean answers "is this better than average?". The
    question that actually matters is "would a room full of random players
    have produced this?" - so run whole strategies through the same
    backtest and read the percentile off their distribution.

    `pool_size` matters more than it looks. A top-scoring portfolio is NOT
    ten independent lines: they are picked by the same score, so they share
    numbers heavily. Measured on this archive, the top ten overlap by 1.64
    balls on average and use only 27 distinct numbers, against 0.56 and 39
    for random ones. Shared numbers mean correlated hits - one lucky ball
    lifts six lines at once - which widens the spread of the portfolio's
    average. Benchmarking that against INDEPENDENT random lines compares it
    to a distribution that is too narrow, and hands the ensemble a
    percentile it did not earn.

    So when `pool_size` is given, each random strategy first draws that many
    distinct balls and then builds its lines from that pool, reproducing the
    concentration rather than assuming it away. Without it the old
    independent-line behaviour is kept, which is correct only for a
    portfolio that really is independent.
    """
    per_strategy = n_draws * n_lines
    if pool_size is None or pool_size >= N_BALLS:
        draws = rng.hypergeometric(N_PICK, N_BALLS - N_PICK, N_PICK,
                                   size=(n_strategies, per_strategy))
        means = draws.mean(axis=1)
    else:
        # Concentration-matched: a pool per strategy, lines drawn from it.
        means = np.empty(n_strategies)
        for i in range(n_strategies):
            pool = rng.choice(N_BALLS, size=pool_size, replace=False) + 1
            lines = pool[rng.random((per_strategy, pool_size))
                         .argsort(axis=1)[:, :N_PICK]]
            # Each draw's six are a fresh random six; counting overlap with
            # a fixed line is symmetric, so sample the drawn sets instead.
            hits = np.empty(per_strategy)
            for j in range(0, per_strategy, n_lines):
                actual = rng.choice(N_BALLS, size=N_PICK, replace=False) + 1
                block = lines[j:j + n_lines]
                hits[j:j + len(block)] = np.isin(block, actual).sum(axis=1)
            means[i] = hits.mean()

    return {
        "n_strategies": n_strategies,
        "pool_size": pool_size,
        "percentile": float((means < ens_avg).mean() * 100),
        "p5": float(np.percentile(means, 5)),
        "p50": float(np.percentile(means, 50)),
        "p95": float(np.percentile(means, 95)),
        "best": float(means.max()),
    }


def portfolio_concentration(lines: np.ndarray) -> dict:
    """How much a portfolio's lines share, which decides its spread."""
    from itertools import combinations as _c
    pairs = list(_c(range(len(lines)), 2))
    overlap = float(np.mean([len(set(lines[a]) & set(lines[b]))
                             for a, b in pairs])) if pairs else 0.0
    return {"mean_overlap": overlap,
            "distinct_balls": int(len(set(lines.ravel())))}


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
                     weights: dict, history: np.ndarray | None = None,
                     triples: dict | None = None) -> np.ndarray:
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

    if weights.get("triple") and triples:
        score = score + weights["triple"] * triple_score(cands, triples)
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
    conc = []
    for t in range(min_history, len(draws), step):
        history = draws[:t]

        # Only draws from the same machine AND ball set as the one about to
        # be used - the physical-bias hypothesis, given its best shot.
        same = (machines[:t] == machines[t]) & (ball_sets[:t] == ball_sets[t])
        machine_rows = history[same]

        feats = ball_features(history, recent_window, machine_rows)
        pairs = pair_matrix(history) if weights["pair"] else None
        triples = triple_matrix(history) if weights.get("triple") else None

        cands = random_lines(n_candidates, rng)
        scores = score_candidates(cands, feats, pairs, weights, history,
                                  triples)
        best = cands[np.argsort(scores)[::-1][:n_lines]]
        rnd = random_lines(n_lines, rng)

        ens_matches.extend(matches(best, draws[t]).tolist())
        rnd_matches.extend(matches(rnd, draws[t]).tolist())
        # The number that actually decides money: how crowded these lines
        # are. 1.0 is an average line; above it you share more.
        ens_pop.extend(popularity_ratio(line) for line in best)
        rnd_pop.extend(popularity_ratio(line) for line in rnd)
        conc.append(portfolio_concentration(best))
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
        "mean_overlap": float(np.mean([c["mean_overlap"] for c in conc])),
        "distinct_balls": float(np.mean([c["distinct_balls"] for c in conc])),
    }


def fit_weights(frame, n_candidates: int, n_lines: int, recent_window: int,
                train_start: int, val_end: int, step: int, n_trials: int,
                rng: np.random.Generator) -> tuple:
    """Search weights on a VALIDATION slice only, then hand them over frozen.

    The objection this answers: if weights are chosen on the same draws the
    result is quoted on, the holdout is not a holdout. So the archive splits
    three ways - train (features only), validation (weights chosen here),
    test (never touched until the weights are frozen).

    The search is random over the simplex, which is enough: the point is not
    to find good weights, it is to show that the best weights findable on
    one slice do not carry to the next.
    """
    best, best_score = None, -np.inf
    for _ in range(n_trials):
        w = {k: float(rng.random()) for k in DEFAULT_WEIGHTS}
        r = walk_forward(frame.iloc[:val_end], n_candidates, n_lines,
                         recent_window, train_start, step, w, rng)
        if r["ensemble_avg"] > best_score:
            best, best_score = w, r["ensemble_avg"]
    return best, best_score


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
    ap.add_argument("--strategies", type=int, default=100_000,
                    help="complete random strategies for the percentile")
    ap.add_argument("--nested", action="store_true",
                    help="fit weights on a validation slice, score on a "
                         "test slice that the fit never saw")
    ap.add_argument("--fit-trials", type=int, default=25)
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

    # "Better than the mean" is a weak question. "Would a room full of
    # random players have produced this?" is the real one.
    pool = int(round(r["distinct_balls"]))
    pc = random_strategy_percentile(r["ensemble_avg"], r["points"],
                                    args.lines, args.strategies, rng,
                                    pool_size=pool)
    pc_naive = random_strategy_percentile(r["ensemble_avg"], r["points"],
                                          args.lines, args.strategies, rng)
    print(f"   Portfolio concentration: lines overlap by "
          f"{r['mean_overlap']:.2f} balls, {r['distinct_balls']:.0f} distinct "
          f"numbers used")
    print(f"   Against {pc['n_strategies']:,} COMPLETE random strategies "
          f"of the SAME concentration ({pool} of 59 balls):")
    print(f"      5th pct {pc['p5']:.4f} | median {pc['p50']:.4f} | "
          f"95th pct {pc['p95']:.4f} | best {pc['best']:.4f}")
    print(f"      ensemble sits at the {pc['percentile']:.1f}th percentile")
    if pc["percentile"] < 95:
        beat = pc["n_strategies"] * (100 - pc["percentile"]) / 100
        print(f"      -> roughly {beat:,.0f} random strategies did as well "
              "or better")
    print(f"   (against INDEPENDENT random lines it would read "
          f"{pc_naive['percentile']:.1f}th - that comparison is too")
    print("    generous, because concentrated portfolios swing wider)")
    print()

    if args.nested:
        n = len(frame)
        val_end = args.min_history + (n - args.min_history) // 2
        print("-" * 70)
        print("NESTED TRAIN / VALIDATION / TEST")
        print("-" * 70)
        print(f"   train: draws < {args.min_history:,} (features only)")
        print(f"   validation: {args.min_history:,}-{val_end:,} "
              f"({args.fit_trials} weight sets tried here)")
        print(f"   test: {val_end:,}-{n:,} (weights frozen before it is read)")
        fitted, val_score = fit_weights(frame, args.candidates, args.lines,
                                        args.recent_window, args.min_history,
                                        val_end, args.step * 2,
                                        args.fit_trials, rng)
        rt = walk_forward(frame, args.candidates, args.lines,
                          args.recent_window, val_end, args.step, fitted, rng)
        print(f"   best weights on validation: "
              + ", ".join(f"{k}={v:.2f}" for k, v in fitted.items()))
        print(f"   their score on validation: {val_score:.4f}")
        print(f"   their score on TEST:       {rt['ensemble_avg']:.4f} "
              f"(random {rt['random_avg']:.4f}, p {rt['p']:.3f})")
        if rt["difference"] <= 0:
            print("   -> the weights that won on validation do not carry over.")
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
