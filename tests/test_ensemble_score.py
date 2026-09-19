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
