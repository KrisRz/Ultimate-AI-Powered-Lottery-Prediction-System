"""The evaluator is tested on planted data only - never on a holdout row.

Section 12.6's rule: a tool meant to settle something must first show it can
detect a planted truth, and that it stays quiet when there is nothing to
find. Everything here runs on synthetic draws. The one place real files are
touched, the assertions are about WHICH rows the loader selects, never about
what they say - reading the answer early is the failure this whole exercise
is built to avoid.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.validations import popularity_v2_frozen as frozen
from scripts.validations import popularity_v2_final_test as fin

EVALUATOR = Path("scripts/validations/popularity_v2_final_test.py")

# Pinned the same way the specification is: the evaluator that produced the
# result must provably be the one written before the result existed.
EVALUATOR_SHA256 = "4037160e26ef2bcd1da23d4b366976f63aae1ad19ed161ce671d9e5096300263"


def test_evaluator_is_unchanged():
    digest = hashlib.sha256(EVALUATOR.read_bytes()).hexdigest()
    assert digest == EVALUATOR_SHA256, (
        "the evaluator changed. If that happened after the final test ran, "
        "the result belongs to the old file and the new one has no holdout."
    )


def test_interpretation_rule_allows_three_outcomes():
    """An interval covering zero is INCONCLUSIVE - not a win for either."""
    assert fin.interpret(+0.01, (0.002, 0.02)) == fin.SUPPORTS_SMOOTH
    assert fin.interpret(-0.01, (-0.02, -0.002)) == fin.SUPPORTS_INCUMBENT
    assert fin.interpret(+0.01, (-0.002, 0.02)) == fin.INCONCLUSIVE
    assert fin.interpret(-0.01, (-0.02, 0.002)) == fin.INCONCLUSIVE
    # The one that matters most: a positive effect this slice cannot resolve
    # must not be reported as the incumbent winning.
    assert fin.interpret(+0.008, (-0.001, 0.017)) != fin.SUPPORTS_INCUMBENT


def test_detects_a_planted_smooth_truth():
    rng = np.random.default_rng(1)
    draws, y, ids = fin._synthetic(rng, 150, "smooth", 0.179)
    r = fin.evaluate(y, draws, ids, frozen.SMOOTH_BETA, frozen.BUCKET_BETA,
                     rng, reps=500, primary="t")
    assert r["verdict"] == fin.SUPPORTS_SMOOTH
    assert r["d_mean"] > 0
    assert r["mse_smooth"] < r["mse_bucket"]


def test_detects_a_planted_incumbent_truth():
    """The mirror image. A tool that can only find one answer is broken."""
    rng = np.random.default_rng(2)
    draws, y, ids = fin._synthetic(rng, 150, "bucket", 0.179)
    r = fin.evaluate(y, draws, ids, frozen.SMOOTH_BETA, frozen.BUCKET_BETA,
                     rng, reps=500, primary="t")
    assert r["verdict"] == fin.SUPPORTS_INCUMBENT
    assert r["d_mean"] < 0


def test_says_nothing_when_there_is_nothing_to_say():
    """Equal-MSE null: false calls at roughly the nominal 10%, not more."""
    rng = np.random.default_rng(3)
    calls = 0
    runs = 60
    for _ in range(runs):
        draws, y, ids = fin._synthetic(rng, 120, "midpoint", 0.179)
        r = fin.evaluate(y, draws, ids, frozen.SMOOTH_BETA,
                         frozen.BUCKET_BETA, rng, reps=300, primary="t")
        calls += r["verdict"] != fin.INCONCLUSIVE
    assert calls / runs < 0.25


def test_the_interval_respects_the_round_structure():
    """Rounds of one draw share a denominator, so rows are not independent.

    With a per-draw shared shock the primary interval must still cover zero
    at close to its nominal rate on eight draws - the size of the real
    prospective holdout.
    """
    rng = np.random.default_rng(4)
    cover = 0
    runs = 60
    for _ in range(runs):
        draws, y, ids = fin._synthetic(rng, 8, "midpoint", 0.09, shared=0.179)
        r = fin.evaluate(y, draws, ids, frozen.SMOOTH_BETA,
                         frozen.BUCKET_BETA, rng, reps=300, primary="t")
        cover += r["ci_t"][0] <= 0.0 <= r["ci_t"][1]
    assert cover / runs >= 0.85


def test_a_level_shift_cannot_change_the_comparison():
    """The exact-sales and rolling-median multipliers sit at different
    levels. That is common to both models and must not move the verdict."""
    rng = np.random.default_rng(5)
    draws, y, ids = fin._synthetic(rng, 80, "smooth", 0.179)
    a = fin.evaluate(y, draws, ids, frozen.SMOOTH_BETA, frozen.BUCKET_BETA,
                     rng, reps=300, primary="t")
    b = fin.evaluate(y + 0.25, draws, ids, frozen.SMOOTH_BETA,
                     frozen.BUCKET_BETA, rng, reps=300, primary="t")
    assert a["d_mean"] == pytest.approx(b["d_mean"], abs=1e-12)
    assert a["verdict"] == b["verdict"]
    # And the uncentred figure, which is reported beside it, does move -
    # which is why the centred one is primary.
    assert a["d_mean_uncentred"] != pytest.approx(b["d_mean_uncentred"])


def test_small_slices_are_honest_about_being_small():
    """Eight draws with the challenger TRUE: mostly inconclusive.

    This is the number that shaped the plan. If this test ever passes
    easily, the power calculation behind the decision was wrong.
    """
    rng = np.random.default_rng(6)
    resolved = 0
    runs = 60
    for _ in range(runs):
        draws, y, ids = fin._synthetic(rng, 8, "smooth", np.sqrt(0.032))
        r = fin.evaluate(y, draws, ids, frozen.SMOOTH_BETA,
                         frozen.BUCKET_BETA, rng, reps=300, primary="t")
        resolved += r["verdict"] == fin.SUPPORTS_SMOOTH
    assert resolved / runs < 0.5


def test_the_prospective_loader_takes_the_frozen_rows_and_no_others():
    """Structure only. No statistic from these rows is computed here.

    The assertion that matters: nothing from draw 3208 on can enter, however
    much the collector appends tonight.
    """
    data = fin.load_prospective(exact=True)
    assert sorted(set(data["draw_ids"])) == list(frozen.HOLDOUT_PRIMARY_DRAWS)
    assert len(data["y"]) == 2 * len(frozen.HOLDOUT_PRIMARY_DRAWS)
    assert data["draws"].shape == (len(data["y"]), 6)
    assert max(data["draw_ids"]) < frozen.HOLDOUT_EXCLUDED_FROM
    assert data["primary"] == "t"


def test_the_evaluator_refuses_to_run_without_being_asked():
    """--final is the only door to real data, and it is not the default."""
    import sys
    argv = sys.argv
    try:
        sys.argv = ["popularity_v2_final_test.py"]
        with pytest.raises(SystemExit):
            fin.main()
    finally:
        sys.argv = argv


def test_every_loader_is_wired_before_the_one_shot_run():
    """Shapes and row selection only - no statistic is computed here.

    A crash halfway through `--final` would be the worst possible moment to
    discover a broken loader: the one shot would be half spent.
    """
    trend = fin.load_prospective(exact=False)
    assert set(trend["draw_ids"]) == set(frozen.HOLDOUT_SECONDARY_DRAWS)
    assert len(trend["y"]) == 2 * len(frozen.HOLDOUT_SECONDARY_DRAWS)
    assert max(trend["draw_ids"]) < frozen.HOLDOUT_EXCLUDED_FROM

    retro = fin.load_retrospective()
    assert retro["draw_ids"].min() == frozen.NEVER_SCORED_FIRST_DRAW
    assert retro["draw_ids"].max() == frozen.NEVER_SCORED_LAST_DRAW
    assert len(retro["y"]) == 573
    assert retro["beta_smooth"] == frozen.SMOOTH_BETA_SECOND_HALF
    assert retro["primary"] == "block"
    # The weekday correction must leave the rows themselves alone.
    assert np.isfinite(retro["y"]).all()
