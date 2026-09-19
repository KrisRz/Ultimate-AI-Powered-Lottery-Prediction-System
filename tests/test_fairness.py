"""Tests for scripts/validations/fairness.py.

The point of these is NOT that the real draws come back clean - a test that
only ever sees clean data cannot tell "the machine is fair" apart from "the
test is broken". So each detector is run twice: once on fair simulated
draws, where it must stay quiet, and once on draws with a planted bias,
where it must fire. Only then does "no signal" on the real archive mean
anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from lottery.ev import N_BALLS, N_PICK
from scripts.validations.fairness import (
    ERA_FIRST_DRAW,
    chi_square_uniformity,
    extreme_ball,
    gap_analysis,
    load_draws,
    load_frame,
    pair_analysis,
    repeat_analysis,
)

# The collector appends to lotto_full_history.csv, so anything asserting on
# counts pins the slice. 3207 is the last draw of 2026-09-16.
PINNED_DRAW = 3207


def fair_draws(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, N_BALLS)).argsort(axis=1)[:, :N_PICK] + 1


def biased_draws(n: int, hot: int = 7, extra: float = 0.35,
                 seed: int = 0) -> np.ndarray:
    """Fair draws, except ball `hot` is forced in on a fraction of them."""
    rng = np.random.default_rng(seed)
    draws = fair_draws(n, seed)
    for i in range(n):
        if rng.random() < extra and hot not in draws[i]:
            draws[i, rng.integers(N_PICK)] = hot
    return draws


# --- the archive itself ---------------------------------------------------

def test_load_draws_is_the_59_ball_era_both_rounds():
    draws = load_draws(max_draw=PINNED_DRAW)
    assert draws.shape == (1171, N_PICK), "59-ball era through draw 3207"
    assert draws.min() >= 1 and draws.max() <= N_BALLS
    # No 49-ball-era contamination: that era could not produce a ball > 49,
    # so its presence proves the era filter held.
    assert (draws > 49).any()


def test_load_draws_respects_the_era_cutoff():
    import pandas as pd
    raw = pd.read_csv("data/lotto_full_history.csv")
    assert (raw["DrawNumber"] < ERA_FIRST_DRAW).any(), "fixture has older eras"
    assert len(load_draws(max_draw=PINNED_DRAW)) < len(raw)


def test_every_ball_appears_in_the_pinned_archive():
    counts = np.bincount(load_draws(max_draw=PINNED_DRAW).ravel(),
                         minlength=N_BALLS + 1)[1:]
    assert (counts > 0).all(), "a ball that never fell would be real news"


# --- detector 1: chi-square ----------------------------------------------

def test_chi_square_quiet_on_fair_draws():
    assert chi_square_uniformity(fair_draws(1200, seed=1))["p"] > 0.05


def test_chi_square_fires_on_a_biased_ball():
    assert chi_square_uniformity(biased_draws(1200, seed=1))["p"] < 0.01


# --- detector 2: most deviant ball, multiple-comparison corrected ---------

def test_extreme_ball_quiet_on_fair_draws():
    rng = np.random.default_rng(2)
    assert extreme_ball(fair_draws(1200, seed=2), 200, rng)["p_corrected"] > 0.05


def test_extreme_ball_fires_on_a_biased_ball():
    rng = np.random.default_rng(3)
    r = extreme_ball(biased_draws(1200, seed=3), 200, rng)
    assert r["p_corrected"] < 0.05
    assert r["hottest"][0][0] == 7, "should finger the ball we planted"


def test_correction_is_what_keeps_the_max_z_honest():
    """The uncorrected reading of the SAME statistic looks significant.

    This is the whole reason the Monte Carlo is there: on clean random data
    the largest |z| among 59 balls sits around 2.5, which a naive two-sided
    normal test reads as p < 0.02 - a finding, from nothing.
    """
    rng = np.random.default_rng(4)
    r = extreme_ball(fair_draws(1200, seed=4), 400, rng)
    assert r["sim_median_max"] > 2.2, "max-of-59 really does run this high"
    assert r["p_corrected"] > 0.05, "corrected test stays quiet"


# --- detector 3: gaps -----------------------------------------------------

def test_gap_analysis_quiet_on_fair_draws():
    assert gap_analysis(fair_draws(1200, seed=5))["p"] > 0.05


def test_gap_analysis_fires_when_overdue_balls_really_do_return():
    """Plant the gambler's fallacy and check the test would catch it."""
    rng = np.random.default_rng(6)
    n = 1500
    draws = fair_draws(n, seed=6)
    last_seen = np.full(N_BALLS + 1, -1)
    for i in range(n):
        overdue = [b for b in range(1, N_BALLS + 1)
                   if last_seen[b] >= 0 and i - last_seen[b] - 1 >= 15]
        if overdue and rng.random() < 0.5:
            pick = overdue[rng.integers(len(overdue))]
            if pick not in draws[i]:
                draws[i, rng.integers(N_PICK)] = pick
        for b in draws[i]:
            last_seen[b] = i
    assert gap_analysis(draws)["p"] < 0.05


# --- detector 4: draw-to-draw dependence ---------------------------------

def test_repeat_analysis_quiet_on_fair_draws():
    r = repeat_analysis(fair_draws(1200, seed=7))
    assert r["p"] > 0.05
    assert r["mean_overlap"] == pytest.approx(r["expected_mean"], abs=0.12)


def test_repeat_analysis_fires_when_draws_echo_each_other():
    rng = np.random.default_rng(8)
    n = 1200
    draws = fair_draws(n, seed=8)
    for i in range(1, n):
        if rng.random() < 0.5:                      # echo one ball forward
            draws[i, rng.integers(N_PICK)] = draws[i - 1][rng.integers(N_PICK)]
    assert repeat_analysis(draws)["p"] < 0.05


# --- detector 5: pairs ----------------------------------------------------

def test_pair_analysis_quiet_on_fair_draws():
    rng = np.random.default_rng(9)
    assert pair_analysis(fair_draws(800, seed=9), 60, rng)["p_corrected"] > 0.05


def test_pair_analysis_fires_on_a_planted_pair():
    rng = np.random.default_rng(10)
    n = 800
    draws = fair_draws(n, seed=10)
    for i in range(n):
        if rng.random() < 0.25:
            draws[i, 0], draws[i, 1] = 11, 12       # a pair that travels together
    r = pair_analysis(draws, 60, rng)
    assert r["p_corrected"] < 0.05
    assert r["top"][0][0] == (11, 12)


# --- the headline claim ---------------------------------------------------

def test_the_real_archive_shows_no_signal():
    """What the report prints, pinned. If this ever fails, read it carefully
    before believing it - and check the collector has not changed the file's
    shape first."""
    draws = load_draws(max_draw=PINNED_DRAW)
    assert chi_square_uniformity(draws)["p"] > 0.05
    assert gap_analysis(draws)["p"] > 0.05
    assert repeat_analysis(draws)["p"] > 0.05


# --- detector 6: machine / ball set --------------------------------------

def test_machine_names_are_normalised():
    """"Lotto 4" and "Lotto4" are the same machine.

    The feed changed format at draw 3191. Without this, every machine
    introduced late splits into two entities and neither has the draws to
    test - the bias hunt starves on a formatting change.
    """
    from scripts.validations.fairness import normalise_machine
    assert normalise_machine("Lotto 4") == normalise_machine("Lotto4")
    assert normalise_machine("Guinevere") == "Guinevere"

    frame = load_frame(max_draw=PINNED_DRAW)
    raw = set(frame["Machine"].unique())
    assert {"Lotto 4", "Lotto4"} <= raw, "both spellings really are in the data"
    assert frame["MachineNorm"].nunique() < len(raw)


def test_detectable_bias_shrinks_with_more_draws():
    from scripts.validations.fairness import detectable_bias
    assert detectable_bias(100) > detectable_bias(1000) > detectable_bias(10000)
    assert detectable_bias(1000) > 1.0


def test_machine_analysis_quiet_on_fair_draws():
    """On fair data the corrected p is uniform, so ONE draw of it says
    nothing - asserting `> 0.05` on a single seed is a test that fails 5% of
    the time by construction, and the first seed tried did exactly that.
    Take the median over several instead: that is stable, and it is the
    claim worth making (the detector is not systematically trigger-happy).
    """
    import pandas as pd
    from scripts.validations.fairness import machine_analysis
    ps = []
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        draws = fair_draws(600, seed=seed)
        df = pd.DataFrame(draws, columns=[f"Number_{i}" for i in range(1, 7)])
        df["G"] = ["a", "b", "c"] * (len(df) // 3)
        ps.append(machine_analysis(df, "G", 120, rng)["p_corrected"])
    assert float(np.median(ps)) > 0.20, f"median corrected p was {ps}"


def test_machine_analysis_fires_on_a_biased_group():
    """One group's drum favours ball 7; the others are clean."""
    import pandas as pd
    from scripts.validations.fairness import machine_analysis
    rng = np.random.default_rng(12)
    clean = fair_draws(400, seed=12)
    dirty = biased_draws(400, hot=7, extra=0.5, seed=13)
    df = pd.concat([
        pd.DataFrame(clean, columns=[f"Number_{i}" for i in range(1, 7)]).assign(G="clean"),
        pd.DataFrame(dirty, columns=[f"Number_{i}" for i in range(1, 7)]).assign(G="dirty"),
    ], ignore_index=True)
    r = machine_analysis(df, "G", 300, rng)
    assert r["p_corrected"] < 0.05
    assert r["rows"][0]["name"] == "dirty", "should name the bad group first"


def test_chi_square_textbook_p_is_biased_by_sampling_without_replacement():
    """Why the report quotes a Monte Carlo p, not scipy's.

    Six balls per draw are taken WITHOUT replacement, so ball counts are
    negatively correlated and their variance falls below the multinomial
    assumption chi-square is built on. The statistic comes out too small and
    the p-value too large. Measured here: fair draws produce a median chi2
    well under the 58 df the textbook expects.
    """
    rng = np.random.default_rng(42)
    r = chi_square_uniformity(fair_draws(1200, seed=42), 400, rng)
    assert r["sim_median_chi2"] < 56.0, (
        "without-replacement draws really do depress chi2 below its df")
    assert r["p_mc"] is not None
    # Same data, two readings: the textbook one is the optimistic one.
    assert r["p"] > r["p_mc"] - 0.10, "analytic p is the conservative reading"


def test_chi_square_monte_carlo_p_still_catches_a_real_bias():
    rng = np.random.default_rng(43)
    r = chi_square_uniformity(biased_draws(1200, seed=43), 200, rng)
    assert r["p_mc"] < 0.05, "the corrected test must not lose its teeth"
