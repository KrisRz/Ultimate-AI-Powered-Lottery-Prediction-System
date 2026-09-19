"""The freeze itself is what these tests protect.

`scripts/validations/popularity_v2_frozen.py` is a pre-registration: it says
what the challenger IS, before its one out-of-sample test is run. A frozen
specification that can be edited without anyone noticing is not frozen, so
the file is pinned by content hash, its numbers are re-derived from the
archive, and popularity-v2-spec.md is checked against the code it describes.

If one of these fails after the final test has been read, that is the
failure mode the freeze exists to catch: a specification quietly adjusted to
the result it was supposed to be judged by.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lottery.ev import N_BALLS, exact_lines_sold
from scripts.validations import popularity_v2_frozen as frozen

SPEC_FILE = Path("scripts/validations/popularity_v2_frozen.py")
SPEC_DOC = Path("popularity-v2-spec.md")

# SHA-256 of the frozen specification, taken 2026-09-19 before any holdout
# row was scored. Changing the file means changing this line by hand - which
# is the point: no unconscious edit survives.
SPEC_SHA256 = "4e275f27425fdb7137c00e20e4539ba9e7986547e5bb407bee32d81a4ec5f902"


def test_specification_file_is_unchanged():
    digest = hashlib.sha256(SPEC_FILE.read_bytes()).hexdigest()
    assert digest == SPEC_SHA256, (
        "the frozen specification changed. If this is deliberate, say so in "
        "popularity-v2-spec.md and update SPEC_SHA256 in the same commit - "
        "and note that a challenger edited after seeing its holdout has no "
        "holdout left."
    )


def test_frozen_coefficients_still_come_out_of_the_archive():
    """Re-derive all four vectors. Drift means the freeze describes nothing.

    Pinned by construction rather than by a snapshot query: the join stops at
    prize_tiers_history.csv, which the collector does not write, so draws the
    collector appends cannot move these numbers.
    """
    r = frozen.refit_from_archive()
    assert r["obs"] == frozen.TRAIN_OBS
    assert r["obs_second_half"] == 574
    for name, got in (("smooth", r["smooth"]),
                      ("bucket", r["bucket"]),
                      ("smooth_second_half", r["smooth_second_half"]),
                      ("bucket_second_half", r["bucket_second_half"])):
        want = getattr(frozen, {
            "smooth": "SMOOTH_BETA",
            "bucket": "BUCKET_BETA",
            "smooth_second_half": "SMOOTH_BETA_SECOND_HALF",
            "bucket_second_half": "BUCKET_BETA_SECOND_HALF",
        }[name])
        assert np.allclose(got, want, atol=1e-12), name


def test_weight_curve_matches_the_frozen_table():
    w = frozen.smooth_weight_table()
    assert np.allclose(w, frozen.SMOOTH_WEIGHTS, atol=1e-9)
    # The population constraint every weight vector in this project obeys.
    assert w.mean() == pytest.approx(1.0, abs=1e-12)
    assert len(w) == N_BALLS
    # Shape as frozen: a gentle rise to the last month, then a clean decline.
    assert int(w.argmax()) + 1 == 12
    assert np.all(np.diff(w[11:]) < 0)


def test_bucket_refit_is_the_installed_production_model():
    """The contest must be incumbent vs challenger, not a straw man."""
    refit = frozen.bucket_weight_table()
    assert np.allclose(refit, frozen.BUCKET_WEIGHTS_REFIT, atol=1e-6)
    assert np.allclose(refit, frozen.INSTALLED_WEIGHTS, atol=0.005)


def test_holdout_holds_only_rows_nothing_has_ever_seen():
    """No training draw, and nothing from 3208 on - that is replication.

    prize_tiers_history.csv ends at TRAIN_LAST_DRAW, so a draw above it has
    never reached load_joined() and cannot have trained or scored anything.
    """
    assert frozen.HOLDOUT_PRIMARY_DRAWS
    for d in frozen.HOLDOUT_PRIMARY_DRAWS + frozen.HOLDOUT_SECONDARY_DRAWS:
        assert d > frozen.TRAIN_LAST_DRAW
        assert d < frozen.HOLDOUT_EXCLUDED_FROM
    assert set(frozen.HOLDOUT_PRIMARY_DRAWS) <= set(
        frozen.HOLDOUT_SECONDARY_DRAWS)
    history = pd.read_csv("data/prize_tiers_history.csv")
    assert int(history["draw_number"].max()) == frozen.TRAIN_LAST_DRAW


def test_the_contaminated_rows_are_named_not_hidden():
    """Two primary draws sit in the Must-Be-Won scorecard, whose two sales
    estimates differ BY the popularity multiplier. Naming them is the only
    honest option; dropping them silently would be the dishonest one."""
    scorecard = pd.read_csv("data/mbw_validation.csv")
    seen = {int(d) for d in scorecard["draw_number"]}
    overlap = seen & set(frozen.HOLDOUT_PRIMARY_DRAWS)
    assert overlap == set(frozen.HOLDOUT_SEEN_BY_SCORECARD), (
        "the scorecard now covers a holdout draw that the freeze does not "
        "disclose")


def test_primary_holdout_is_exactly_what_the_pool_identity_can_price():
    """Frozen by rule, not by taste: every priceable draw in the window, and
    only those. A pool revision that changes this SHOULD break the test."""
    pools = pd.read_csv("data/draw_pools.csv")
    priceable = {d for d in exact_lines_sold(pools)
                 if frozen.TRAIN_LAST_DRAW < d < frozen.HOLDOUT_EXCLUDED_FROM}
    assert priceable == set(frozen.HOLDOUT_PRIMARY_DRAWS)


def test_the_document_and_the_code_say_the_same_thing():
    """A spec doc that drifts from its code is worse than no spec doc."""
    # The document is prose: a typographic minus and a decimal comma.
    doc = SPEC_DOC.read_text().replace("\u2212", "-").replace(",", ".")
    for value in frozen.SMOOTH_BETA + frozen.BUCKET_BETA:
        assert f"{value:+.8f}" in doc, value
    for draw in frozen.HOLDOUT_PRIMARY_DRAWS:
        assert str(draw) in doc
    assert "ZAMROŻONE 2026-09-19" in doc
    assert frozen.FROZEN_AT_COMMIT in doc


def test_the_power_report_can_detect_a_planted_truth():
    """Section 12.6's rule, applied to this file's own tool.

    With the challenger planted as the truth and 573 rows, the pre-registered
    threshold must fire nearly always - otherwise a null result on the big
    slice would mean nothing.
    """
    rng = np.random.default_rng(7)
    strong = frozen.power(rng, 573, sigma2=0.11432, reps=40)
    assert strong["significant"] > 0.8
    # And the virgin slice must NOT clear it often - the honest reason the
    # 16-row holdout cannot install anything.
    weak = frozen.power(rng, 16, sigma2=0.11432, reps=40)
    assert weak["significant"] < 0.5


def test_separation_matches_the_gap_the_archive_showed():
    """0.01139 was measured out of sample; 0.01012 is the two frozen
    predictors pulling apart on random draws. Same effect, two routes."""
    rng = np.random.default_rng(11)
    sep = frozen.separation(rng, n=20_000)
    assert 0.008 < sep < 0.013
