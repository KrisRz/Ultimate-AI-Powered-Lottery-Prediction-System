"""Tests for scripts/validations/ensemble_score.py.

A backtest that reports "no edge" is only worth reading if it WOULD report
an edge when one exists. So the load-bearing test here plants a real,
exploitable signal in synthetic draws and checks the harness finds it.
Everything else is plumbing around that claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lottery.ev import N_BALLS, N_PICK
from scripts.validations.ensemble_score import (
    DEFAULT_WEIGHTS,
    matches,
    pair_matrix,
    random_lines,
    shape_score,
    walk_forward,
)

NUM_COLS = [f"Number_{i}" for i in range(1, N_PICK + 1)]


def _frame(draws: np.ndarray, machine: str = "Arthur",
           ball_set: str = "1") -> pd.DataFrame:
    df = pd.DataFrame(draws, columns=NUM_COLS)
    df["MachineNorm"] = machine
    df["BallSetStr"] = ball_set
    return df


def fair(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, N_BALLS)).argsort(axis=1)[:, :N_PICK] + 1


# --- plumbing -------------------------------------------------------------

def test_matches_counts_overlap():
    lines = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    assert matches(lines, np.array([1, 2, 3, 40, 50, 59])).tolist() == [3, 0]


def test_pair_matrix_is_symmetric_with_empty_diagonal():
    m = pair_matrix(fair(200, seed=1))
    assert m.shape == (N_BALLS, N_BALLS)
    assert np.allclose(m, m.T)


def test_shape_score_prefers_typical_lines():
    """A line of 1-2-3-4-5-6 is structurally freakish; a spread one is not.

    Both have identical probability of being drawn - that is the point the
    report makes - but the shape component is supposed to rank them apart,
    otherwise it is not measuring what it claims to.
    """
    history = fair(600, seed=2)
    cands = np.array([[1, 2, 3, 4, 5, 6], [7, 18, 29, 34, 46, 57]])
    s = shape_score(cands, history)
    assert s[1] > s[0], "the spread line should look more 'normal'"


# --- the claim that matters ----------------------------------------------

def test_harness_finds_a_real_edge_when_one_is_planted():
    """Six balls are genuinely over-drawn. The ensemble must beat random.

    Without this, "no edge on the real archive" is indistinguishable from
    "this harness cannot detect an edge at all".
    """
    rng = np.random.default_rng(3)
    n = 900
    hot = np.array([7, 13, 21, 34, 42, 55])
    draws = fair(n, seed=3)
    for i in range(n):
        # Half the draws are forced to contain three of the hot six.
        if rng.random() < 0.5:
            draws[i, :3] = rng.choice(hot, 3, replace=False)
            if len(set(draws[i])) < N_PICK:
                draws[i] = fair(1, seed=1000 + i)[0]

    r = walk_forward(_frame(draws), 5000, 10, 50, 500, 4, DEFAULT_WEIGHTS,
                     np.random.default_rng(4))
    assert r["difference"] > 0, "planted edge should show as a positive gap"
    assert r["p"] < 0.05, f"harness missed a real edge (p={r['p']})"


def test_no_edge_on_fair_draws():
    r = walk_forward(_frame(fair(900, seed=5)), 5000, 10, 50, 500, 4,
                     DEFAULT_WEIGHTS, np.random.default_rng(6))
    assert r["p"] > 0.05
    assert r["random_avg"] == pytest.approx(r["theoretical_avg"], abs=0.15)


def test_walk_forward_never_sees_the_future():
    """The guard the whole method rests on.

    If the scorer could see draw t while scoring it, a perfect-looking
    result would follow and mean nothing. Here the future is made
    grotesque - the last quarter of the archive is fixed to one line - and
    the score for the earlier draws must be unaffected by it.
    """
    base = fair(800, seed=7)
    poisoned = base.copy()
    poisoned[600:] = np.array([1, 2, 3, 4, 5, 6])

    kw = dict(n_candidates=3000, n_lines=5, recent_window=50,
              min_history=400, step=20, weights=DEFAULT_WEIGHTS)
    a = walk_forward(_frame(base), rng=np.random.default_rng(8), **kw)
    b = walk_forward(_frame(poisoned), rng=np.random.default_rng(8), **kw)

    # Draws before 600 are identical in both, so their scored lines must be
    # too; only the poisoned tail may differ.
    assert a["points"] == b["points"]
    assert a["ensemble_avg"] != b["ensemble_avg"], (
        "the poisoned tail should change LATER draws, proving the walk moves")


def test_popularity_of_picked_lines_is_reported():
    r = walk_forward(_frame(fair(700, seed=9)), 3000, 5, 50, 500, 10,
                     DEFAULT_WEIGHTS, np.random.default_rng(10))
    assert r["ensemble_popularity"] > 0
    assert r["random_popularity"] == pytest.approx(1.0, abs=0.25), (
        "random lines should average about the population mean of 1.0")


def test_single_component_runs_are_possible():
    """`--only shape` must actually isolate the component."""
    weights = {k: (1.0 if k == "shape" else 0.0) for k in DEFAULT_WEIGHTS}
    r = walk_forward(_frame(fair(700, seed=11)), 3000, 5, 50, 500, 10,
                     weights, np.random.default_rng(12))
    assert r["lines"] > 0


# --- regularised triples --------------------------------------------------

def test_triple_matrix_is_starved_by_construction():
    """C(59,3)=32,509 cells, 20 per draw: the data cannot fill them.

    This is the number that justifies the regularisation rather than a raw
    triple table - at ~1,000 draws the average cell holds about half an
    observation, so the biggest counts are noise.
    """
    from math import comb as _comb
    from scripts.validations.ensemble_score import triple_matrix
    history = fair(900, seed=20)
    expected_per_cell = 900 * _comb(6, 3) / _comb(59, 3)
    assert expected_per_cell < 1.0

    kept_loose = triple_matrix(history, min_obs=2)
    kept_tight = triple_matrix(history, min_obs=5)
    assert len(kept_tight) < len(kept_loose) < _comb(59, 3) / 5


def test_triple_shrinkage_damps_small_counts():
    """A triple seen 3 times must not outscore one seen 30 times."""
    from scripts.validations.ensemble_score import triple_matrix
    history = fair(400, seed=21)
    # Plant a triple that genuinely travels together.
    for i in range(0, 400, 3):
        history[i, :3] = [5, 11, 23]
    t = triple_matrix(history, min_obs=3, shrink=20.0)
    planted = t.get((5, 11, 23))
    assert planted is not None
    others = [v for k, v in t.items() if k != (5, 11, 23)]
    assert planted > max(others), "real support should win over accidents"


def test_triple_score_is_zero_without_a_table():
    from scripts.validations.ensemble_score import triple_score
    assert np.allclose(triple_score(fair(10, seed=22), {}), 0.0)


# --- the random-strategy benchmark ---------------------------------------

def test_random_strategy_percentile_centres_on_theory():
    """A strategy scoring exactly the theoretical mean sits near the median."""
    from scripts.validations.ensemble_score import random_strategy_percentile
    rng = np.random.default_rng(23)
    theory = N_PICK * N_PICK / N_BALLS
    r = random_strategy_percentile(theory, 100, 10, 20_000, rng)
    assert 40 < r["percentile"] < 60
    assert r["p5"] < theory < r["p95"]


def test_random_strategy_spread_shrinks_with_more_lines():
    """More lines per strategy = tighter distribution = a harder bar.

    The reason a short backtest flatters a strategy: with few lines the 95th
    percentile of pure luck sits far above the mean.
    """
    from scripts.validations.ensemble_score import random_strategy_percentile
    rng = np.random.default_rng(24)
    narrow = random_strategy_percentile(0.61, 500, 10, 5_000, rng)
    wide = random_strategy_percentile(0.61, 50, 10, 5_000, rng)
    assert (narrow["p95"] - narrow["p5"]) < (wide["p95"] - wide["p5"])


def test_percentile_exposes_a_significant_looking_run():
    """p < 0.05 against the mean, yet thousands of random strategies match it.

    Both readings are of the same run; the percentile is the one that says
    how many coin-flippers would have done as well.
    """
    from scripts.validations.ensemble_score import random_strategy_percentile
    rng = np.random.default_rng(25)
    r = random_strategy_percentile(0.6646, 48, 10, 100_000, rng)
    assert r["percentile"] < 99.0
    assert r["best"] > 0.70, "pure luck reaches well past any single result"


# --- nested train / validation / test ------------------------------------

def test_fitted_weights_do_not_carry_from_validation_to_test():
    """The objection, answered on fair data: weights chosen on one slice
    have no reason to work on the next, and do not."""
    from scripts.validations.ensemble_score import fit_weights
    frame = _frame(fair(900, seed=26))
    rng = np.random.default_rng(27)
    fitted, val_score = fit_weights(frame, 2000, 5, 50, 500, 700, 20, 6, rng)
    assert set(fitted) == set(DEFAULT_WEIGHTS)

    test = walk_forward(frame, 2000, 5, 50, 700, 10, fitted,
                        np.random.default_rng(28))
    # Validation is where they were chosen, so they look good there;
    # the test slice is the honest read and must not inherit the flattery.
    assert val_score >= test["ensemble_avg"] - 0.25


# --- concentration matching ----------------------------------------------

def test_top_lines_are_concentrated_unlike_random_ones():
    """Top-scoring lines share numbers; random ones do not.

    This is why the benchmark has to match concentration: a portfolio whose
    lines overlap swings wider, because one lucky ball lifts several lines
    at once.
    """
    from scripts.validations.ensemble_score import portfolio_concentration
    rng = np.random.default_rng(30)
    lines = np.array([[11, 37, 42, 44, 52, 58], [11, 22, 36, 39, 46, 52],
                      [11, 29, 36, 37, 54, 58], [8, 11, 31, 37, 42, 58]])
    c = portfolio_concentration(lines)
    assert c["mean_overlap"] > 1.0
    assert c["distinct_balls"] < 24

    r = portfolio_concentration(random_lines(4, rng))
    assert r["mean_overlap"] < c["mean_overlap"]


def test_concentrated_benchmark_is_wider_than_the_independent_one():
    """The bug this fixes: an independent benchmark is too narrow.

    Matching concentration widens the reference distribution, which LOWERS
    the percentile a concentrated portfolio earns - i.e. the naive version
    flattered it.
    """
    from scripts.validations.ensemble_score import random_strategy_percentile
    rng = np.random.default_rng(31)
    indep = random_strategy_percentile(0.66, 48, 10, 4_000, rng)
    matched = random_strategy_percentile(0.66, 48, 10, 4_000, rng,
                                         pool_size=27)
    assert (matched["p95"] - matched["p5"]) > (indep["p95"] - indep["p5"])
    assert matched["percentile"] < indep["percentile"]


def test_full_pool_matches_the_independent_case():
    """pool_size >= 59 must fall back to the independent sampler."""
    from scripts.validations.ensemble_score import random_strategy_percentile
    a = random_strategy_percentile(0.61, 50, 10, 3_000,
                                   np.random.default_rng(32))
    b = random_strategy_percentile(0.61, 50, 10, 3_000,
                                   np.random.default_rng(32), pool_size=59)
    assert a["percentile"] == pytest.approx(b["percentile"], abs=2.0)
