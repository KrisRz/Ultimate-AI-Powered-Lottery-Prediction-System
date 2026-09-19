"""Tests for scripts/data_contract.py.

The contract exists because of one specific miss: "Lotto 4" became
"Lotto4" at draw 3191 and nothing noticed, because nothing was wrong -
only different. So the tests that matter are the ones proving the three
states are actually distinguished: a new machine must NOT fail ingest, and
a broken row must.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.data_contract import (
    INVALID,
    KNOWN,
    UNKNOWN_BUT_VALID,
    normalise_machine,
    run,
)


BASE_DATE = pd.Timestamp("2020-01-01")


def frame(rows=3, start=2100, machine="Arthur", ball_set="1"):
    """A minimal well-formed 59-ball-era archive.

    Dates are derived from the draw number, not the row index, so two
    frames concatenated at different starts stay in chronological order -
    otherwise a test about machine names fails on a date rule instead.
    """
    data = []
    for i in range(rows):
        n = start + i
        base = 1 + (i * 7) % 40
        data.append({
            "Draw Date": (BASE_DATE + pd.Timedelta(days=n)).strftime("%Y-%m-%d"),
            **{f"Number_{j}": base + j for j in range(1, 7)},
            "Bonus": 59 - (i % 10),
            "DrawNumber": n,
            "Round": 1,
            "Machine": machine,
            "Ball Set": ball_set,
        })
    df = pd.DataFrame(data)
    return df.sort_values("DrawNumber").reset_index(drop=True)


# --- the three states -----------------------------------------------------

def test_clean_data_is_known():
    f = run(frame(), check_pools=False)
    assert f.status == KNOWN, (f.invalid, f.unknown)


def test_a_new_machine_is_unknown_but_valid_not_invalid():
    """Camelot adding Lotto7 must not break collection.

    This is the whole reason the middle state exists: refusing a legitimate
    new value is as bad as silently swallowing it.
    """
    f = run(frame(machine="Lotto7"), check_pools=False)
    assert f.status == UNKNOWN_BUT_VALID
    assert not f.invalid
    assert any("Lotto7" in m for m in f.unknown)


def test_spelling_drift_is_reported():
    """The 3191 event, as a rule rather than a habit."""
    df = pd.concat([frame(rows=2, start=2100, machine="Lotto 4"),
                    frame(rows=2, start=2102, machine="Lotto4")],
                   ignore_index=True)
    f = run(df, check_pools=False)
    assert any("SPELLING DRIFT" in m for m in f.unknown)
    assert not f.invalid, "different spellings are not malformed data"


def test_normalise_machine_collapses_the_two_spellings():
    assert normalise_machine("Lotto 4") == normalise_machine("Lotto4")
    assert normalise_machine("Guinevere") == "Guinevere"


# --- structural failures are fatal ---------------------------------------

def test_missing_column_is_invalid():
    df = frame().drop(columns=["Bonus"])
    assert run(df, check_pools=False).status == INVALID


def test_ball_out_of_era_range_is_invalid():
    """A 53 before draw 2066 is a 49-ball draw that cannot exist."""
    df = frame(start=1000)
    df.loc[0, "Number_1"] = 53
    f = run(df, check_pools=False)
    assert f.status == INVALID
    assert any("era boundary" in m for m in f.invalid)


def test_repeated_ball_within_a_draw_is_invalid():
    df = frame()
    df.loc[0, "Number_2"] = df.loc[0, "Number_1"]
    assert run(df, check_pools=False).status == INVALID


def test_bonus_among_the_main_six_is_invalid():
    df = frame()
    df.loc[0, "Bonus"] = df.loc[0, "Number_3"]
    assert run(df, check_pools=False).status == INVALID


def test_duplicate_draw_round_is_invalid():
    df = pd.concat([frame(rows=1), frame(rows=1)], ignore_index=True)
    assert run(df, check_pools=False).status == INVALID


def test_a_gap_in_draw_numbers_is_invalid():
    df = frame(rows=3)
    df.loc[2, "DrawNumber"] = df.loc[2, "DrawNumber"] + 5
    f = run(df, check_pools=False)
    assert f.status == INVALID
    assert any("missing draw number" in m for m in f.invalid)


def test_dates_disagreeing_with_draw_order_is_invalid():
    df = frame(rows=3)
    df.loc[2, "Draw Date"] = "2010-01-01"   # a later draw, an earlier date
    assert run(df, check_pools=False).status == INVALID


def test_a_third_round_is_loud_because_ev_sums_over_rounds():
    df = frame(rows=2)
    df.loc[1, "Round"] = 3
    f = run(df, check_pools=False)
    assert f.status == UNKNOWN_BUT_VALID
    assert any("UNKNOWN ROUND" in m for m in f.unknown)


# --- the real archive -----------------------------------------------------

def test_the_live_archive_is_not_invalid():
    """Whatever else it reports, the committed history must be well-formed."""
    f = run()
    assert f.status != INVALID, f.invalid
