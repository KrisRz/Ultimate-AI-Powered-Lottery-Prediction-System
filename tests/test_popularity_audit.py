"""Tests for scripts/validations/popularity_audit.py.

The audit's own claims need the same treatment it gives the model: the
recovery check must fail when the calibration is broken, and the
sensitivity grid must report a flip when one really happens.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

import lottery.ev as ev
from scripts.validations.popularity_audit import (
    INSTALLED,
    fit_from_multipliers,
    scale_spread,
    set_weights,
    synthetic_multipliers,
)

# A Saturday draw, priced from pinned inputs instead of the wall clock.
#
# These conditions used to come from a bare `pinned_conditions()`, which
# reads the clock: on a draw day at 18:30 UTC the upcoming draw becomes the
# NEXT one, the weekday flips Saturday -> Wednesday and the sales baseline
# drops from 8.45m lines to 5.14m. Three tests here were green at 18:20 UTC
# on 2026-09-19 and red at 18:40, having changed nothing - and the
# collector's 21:45 run executes this suite, so the failure would have
# landed on a night with a real draw in it.
#
# What these tests assert is how the POPULARITY MODEL moves a verdict, not
# which draw happens to be next, so both the moment and the sales figure are
# pinned. `next_draw_conditions` already takes `now` for exactly this reason
# (the site exporter pins it so CI diffs stay stable).
PINNED_NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
PINNED_TICKETS = 8_451_176      # the Saturday estimate at that moment


def pinned_conditions():
    """The draw these tests were written against, and always will be."""
    from scripts.ev_play import next_draw_conditions
    return replace(next_draw_conditions(now=PINNED_NOW),
                   tickets_sold=PINNED_TICKETS)


@pytest.fixture(autouse=True)
def restore_weights():
    """Every test rebinds the live EV module, so put it back."""
    yield
    set_weights(*INSTALLED)


def draws(n, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n, ev.N_BALLS)).argsort(axis=1)[:, :ev.N_PICK] + 1


def test_scale_spread_keeps_flat_at_one_and_installed_at_installed():
    assert scale_spread(0.0) == (1.0, 1.0, 1.0)
    assert scale_spread(1.0) == pytest.approx(INSTALLED)
    doubled = scale_spread(2.0)
    assert doubled[0] > INSTALLED[0] and doubled[2] < INSTALLED[2]


def test_set_weights_rebuilds_the_normalisation():
    """Changing weights must move popularity_ratio, not just the lookup.

    POPULARITY_NORMALIZATION is computed over all C(59,6) lines at import;
    forgetting to recompute it would leave ratios silently on the old
    scale, which is the kind of bug that makes an audit agree with itself.
    """
    line = [32, 34, 37, 39, 41, 43]
    set_weights(*INSTALLED)
    installed, norm_installed = ev.popularity_ratio(line), ev.POPULARITY_NORMALIZATION
    set_weights(1.0, 1.0, 1.0)
    flat, norm_flat = ev.popularity_ratio(line), ev.POPULARITY_NORMALIZATION

    assert norm_flat != norm_installed, "the normalisation must be recomputed"
    assert installed < 0.5, "an unpopular line is well below average"

    # Flat weights do NOT give exactly 1.0, and the reason is worth pinning:
    # the raw score for this line is 1.0, but the normalisation averages over
    # all C(59,6) lines INCLUDING the pattern multipliers (runs, arithmetic
    # sequences, birthday-only tickets), which do not depend on number_weight.
    # So even with no per-number bias the mean line still scores above a
    # patternless one.
    assert ev._raw_popularity(line) == pytest.approx(1.0)
    assert norm_flat > 1.0
    assert flat == pytest.approx(1.0 / norm_flat, rel=1e-9)


def test_calibration_recovers_planted_weights():
    """The load-bearing claim: feed known weights in, get them back."""
    d = draws(1200, seed=1)
    for truth in (INSTALLED, (1.0, 1.0, 1.0), scale_spread(2.0)):
        mult = synthetic_multipliers(d, truth, 0.30, np.random.default_rng(2))
        fit = fit_from_multipliers(d, mult)
        assert fit["low12"] == pytest.approx(truth[0], abs=0.06)
        assert fit["mid"] == pytest.approx(truth[1], abs=0.05)
        assert fit["high"] == pytest.approx(truth[2], abs=0.04)


def test_recovery_would_fail_on_a_broken_calibration():
    """Guard the guard: drop the un-damping and the fit must go wrong.

    The 6/3 factor converts a Match-3 regression coefficient back into a
    pick-rate. Without it the recovered spread is visibly too small - so a
    passing recovery test means the factor is actually being applied.
    """
    d = draws(1200, seed=3)
    mult = synthetic_multipliers(d, scale_spread(2.0), 0.30,
                                 np.random.default_rng(4))
    n_low12 = (d <= 12).sum(axis=1)
    n_high = (d > 31).sum(axis=1)
    X = np.column_stack([np.ones(len(d)), n_low12, n_high])
    beta, *_ = np.linalg.lstsq(X, np.log(mult), rcond=None)
    r_low_undamped = np.exp(1.0 * beta[1])      # the bug: factor 1, not 6/3
    correct = fit_from_multipliers(d, mult)
    assert r_low_undamped < correct["low12"] / correct["mid"] - 0.05


def test_popularity_never_flips_a_roll_down_verdict():
    """The finding that bounds the whole risk.

    On a roll-down the EV is carried by J/N, which popularity does not
    enter, so the verdict must be identical from a flat model to a doubled
    one. If this ever fails, the audit's scope statement is wrong.
    """
    from dataclasses import replace
    base = pinned_conditions()
    for pool, sold in ((9_000_000, 9_500_000), (15_000_000, 13_500_000),
                       (20_000_000, 15_000_000)):
        cond = replace(base, jackpot=pool, roll_down=True, tickets_sold=sold)
        verdicts = set()
        for factor in (0.0, 1.0, 2.0):
            set_weights(*scale_spread(factor))
            verdicts.add(ev.should_play(cond)["play"])
        assert len(verdicts) == 1, f"roll-down at {pool} flipped on weights"


def test_popularity_does_flip_a_big_ordinary_draw():
    """The other half of the same finding, pinned so it cannot be forgotten.

    An ordinary draw near break-even IS sensitive to these weights. The
    audit exists because this is true, not because it is not.
    """
    from dataclasses import replace

    cond = replace(pinned_conditions(), jackpot=32_000_000,
                   roll_down=False)
    set_weights(*INSTALLED)
    installed = ev.should_play(cond)["play"]
    set_weights(*scale_spread(2.0))
    doubled = ev.should_play(cond)["play"]
    assert installed is False and doubled is True


# --- decision stability (the grey zone label) -----------------------------

def test_stability_labels_match_the_audit_grid():
    """Every scenario the audit measured, pinned to its label."""
    from dataclasses import replace
    from lottery.ev import should_play

    base = pinned_conditions()
    cases = [
        ({}, "ROBUST SKIP"),                                        # today
        (dict(jackpot=32_000_000), "MODEL-SENSITIVE"),
        (dict(jackpot=36_000_000), "MODEL-SENSITIVE"),
        (dict(jackpot=9_000_000, roll_down=True,
              tickets_sold=9_500_000), "ROBUST SKIP"),
        (dict(jackpot=20_000_000, roll_down=True,
              tickets_sold=15_000_000), "ROBUST PLAY"),
    ]
    for kw, expected in cases:
        v = should_play(replace(base, **kw))
        assert v["model_stability"]["label"] == expected, kw


def test_stability_restores_the_module_after_scanning():
    """It rebinds globals; leaving them changed would poison every later
    call in the process - including the verdict it was asked about."""
    from dataclasses import replace
    from lottery.ev import decision_stability, popularity_ratio

    line = [32, 34, 37, 39, 41, 43]
    before = popularity_ratio(line)
    before_norm = ev.POPULARITY_NORMALIZATION
    decision_stability(replace(pinned_conditions(), jackpot=32_000_000))
    assert popularity_ratio(line) == pytest.approx(before)
    assert ev.POPULARITY_NORMALIZATION == pytest.approx(before_norm)


def test_stability_survives_an_exception_midway():
    """The restore is in a finally block, and that has to stay true."""
    from lottery.ev import decision_stability, popularity_ratio

    line = [32, 34, 37, 39, 41, 43]
    before = popularity_ratio(line)
    with pytest.raises(Exception):
        decision_stability(None)          # None has no .tickets_sold
    assert popularity_ratio(line) == pytest.approx(before)


def test_roll_downs_are_not_automatically_robust():
    """A roll-down near the threshold IS model-sensitive.

    This test was originally written the other way round - asserting every
    roll-down is robust, on the reasoning that J/N dominates its EV - and
    it failed. The GBP 15m special against 12m lines reads -0.053 (SKIP)
    with a flat popularity model and +0.024 (PLAY) with the installed one,
    and the audit calls GBP 15m the marginal PLAY this project is most
    likely to meet. The label tracks distance from the threshold, not the
    kind of draw, and ev.py now says so.
    """
    from dataclasses import replace
    from lottery.ev import should_play

    base = pinned_conditions()
    labels = {}
    for pool in (7_000_000, 9_000_000, 12_000_000, 15_000_000, 20_000_000):
        v = should_play(replace(base, jackpot=pool, roll_down=True,
                                tickets_sold=12_000_000))
        labels[pool] = v["model_stability"]["label"]

    assert labels[15_000_000] == "MODEL-SENSITIVE", (
        "the marginal special is the case this flag exists for")
    # Far from the threshold in either direction, the shape stops mattering.
    assert labels[7_000_000] == "ROBUST SKIP"
    assert labels[20_000_000] == "ROBUST PLAY"


def test_label_tracks_distance_from_threshold_not_draw_type():
    """The same pool, two sales levels, two labels.

    GBP 15m against 13.5m lines is a comfortable SKIP; against 12m lines it
    is a marginal PLAY and the flag fires. Nothing about the draw's TYPE
    changed between them.
    """
    from dataclasses import replace
    from lottery.ev import should_play

    base = pinned_conditions()
    busy = should_play(replace(base, jackpot=15_000_000, roll_down=True,
                               tickets_sold=13_500_000))
    quiet = should_play(replace(base, jackpot=15_000_000, roll_down=True,
                                tickets_sold=12_000_000))
    assert busy["model_stability"]["label"] == "ROBUST SKIP"
    assert quiet["model_stability"]["label"] == "MODEL-SENSITIVE"


# --- specification contest ------------------------------------------------

def test_three_buckets_lose_out_of_sample_to_a_smooth_model():
    """The installed shape is too rigid, and the data say so consistently.

    Not a close call and not a split-count artefact: smooth beats buckets
    at every split count tried, by roughly 10%. Recorded rather than acted
    on - installing it would move published figures and the PLAY
    threshold, which is Kris's call, not the audit's.
    """
    import logging

    from scripts.calibrate_popularity import add_multiplier, load_joined
    from scripts.validations.popularity_audit import specification_contest

    logging.getLogger().setLevel(logging.WARNING)
    df = add_multiplier(load_joined())
    draws_arr = df[[f"Number_{i}" for i in range(1, 7)]].to_numpy(int)
    mult = df["multiplier"].to_numpy()

    for n_splits in (3, 5, 10):
        rows = {r["name"]: r for r in specification_contest(draws_arr, mult,
                                                            n_splits=n_splits)}
        buckets = rows["3-bucket (installed)"]["oos_mse"]
        smooth = rows["smooth + hinges"]["oos_mse"]
        assert smooth < buckets, f"at {n_splits} splits: {smooth} vs {buckets}"


def test_the_overfit_yardstick_behaves_like_one():
    """59 parameters must fit best in-sample - otherwise the contest is
    not measuring what it claims, and 'smooth wins' means nothing."""
    import logging

    from scripts.calibrate_popularity import add_multiplier, load_joined
    from scripts.validations.popularity_audit import specification_contest

    logging.getLogger().setLevel(logging.WARNING)
    df = add_multiplier(load_joined())
    draws_arr = df[[f"Number_{i}" for i in range(1, 7)]].to_numpy(int)
    rows = {r["name"]: r
            for r in specification_contest(draws_arr,
                                           df["multiplier"].to_numpy())}
    per_number = rows["per-number (overfit)"]
    assert per_number["params"] == 60
    assert per_number["in_mse"] < rows["smooth + hinges"]["in_mse"]
    assert per_number["in_mse"] < rows["3-bucket (installed)"]["in_mse"]
    assert per_number["oos_mse"] > per_number["in_mse"]


def test_the_smooth_challenger_flips_a_verdict_the_label_already_flagged():
    """The audit's payoff: a better-supported model moves a decision, and
    the MODEL-SENSITIVE label had already marked that draw.

    If this ever fails because the flag stops firing there, the label has
    lost the case it exists for.
    """
    import logging
    from dataclasses import replace

    from lottery.ev import should_play
    from scripts.calibrate_popularity import add_multiplier, load_joined
    from scripts.validations.popularity_audit import (
        install_weight_fn, smooth_weight_fn,
    )

    logging.getLogger().setLevel(logging.WARNING)
    df = add_multiplier(load_joined())
    draws_arr = df[[f"Number_{i}" for i in range(1, 7)]].to_numpy(int)
    smooth = smooth_weight_fn(draws_arr, df["multiplier"].to_numpy())

    cond = replace(pinned_conditions(), jackpot=32_000_000,
                   roll_down=False)
    set_weights(*INSTALLED)
    installed = should_play(cond)
    assert installed["play"] is False
    assert installed["model_stability"]["label"] == "MODEL-SENSITIVE"

    install_weight_fn(smooth)
    assert should_play(cond)["play"] is True
    set_weights(*INSTALLED)
