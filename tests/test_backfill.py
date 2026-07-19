"""Tests for the Merseyworld archive parser."""

import pytest

from scripts.backfill_history import parse_archive


def _wrap(rows: str) -> str:
    return (
        "<HTML><PRE> UK National Lotto Winning Numbers\n\n"
        "No., Day,DD,MMM,YYYY, N1,N2,N3,N4,N5,N6,BN,   Jackpot, Wins,   Machine  ,Set\n"
        f"{rows}"
        "\nData obtained from http://lottery.merseyworld.com/\n</PRE></HTML>"
    )


VALID_ROW_OLD = "1000, Sat,05,Jun,2004, 01,12,23,34,45,49,07,   5000000,    1,   Merlin  ,  2 \n"
VALID_ROW_NEW_R1 = "3190, Sat,18,Jul,2026, 55,47,52,22,34,32,10,   9559451,    0,   Lotto 4  ,  3 \n"
VALID_ROW_NEW_R2 = "3190, Sat,18,Jul,2026, 55,42,46,20,30,29,38,   9559451,    0,   Lotto 5  ,  4 \n"


def test_parses_valid_rows_and_sorts_numbers():
    df = parse_archive(_wrap(VALID_ROW_OLD))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Draw Date"] == "2004-06-05"
    assert [row[f"Number_{i}"] for i in range(1, 7)] == [1, 12, 23, 34, 45, 49]
    assert row["Bonus"] == 7
    assert row["DrawNumber"] == 1000
    assert row["Round"] == 1


def test_two_round_draws_get_round_numbers():
    df = parse_archive(_wrap(VALID_ROW_NEW_R1 + VALID_ROW_NEW_R2))
    assert len(df) == 2
    assert list(df["Round"]) == [1, 2]
    # Page order determines rounds: Lotto 4 row listed first -> Round 1
    assert df.iloc[0]["Machine"] == "Lotto 4"


def test_duplicate_listing_is_deduped():
    # The archive repeats recent draws in current + expired sections
    df = parse_archive(_wrap(VALID_ROW_OLD + VALID_ROW_OLD))
    assert len(df) == 1


def test_out_of_range_number_is_rejected():
    # 55 is invalid before October 2015 (49-ball era)
    bad = "1000, Sat,05,Jun,2004, 01,12,23,34,45,55,07,   5000000,    1,   Merlin  ,  2 \n"
    with pytest.raises(ValueError, match="invalid rows"):
        parse_archive(_wrap(bad))


def test_duplicate_number_in_draw_is_rejected():
    bad = "1000, Sat,05,Jun,2004, 01,12,23,34,45,45,07,   5000000,    1,   Merlin  ,  2 \n"
    with pytest.raises(ValueError, match="invalid rows"):
        parse_archive(_wrap(bad))


def test_prose_and_separators_are_skipped():
    html = _wrap(
        VALID_ROW_OLD
        + "<HR><B>All lotteries below have exceeded the 180 days expiry date</B><HR>\n"
        + "This page shows all the draws that used any machine.\n"
    )
    df = parse_archive(html)
    assert len(df) == 1


def test_missing_pre_block_raises():
    with pytest.raises(ValueError, match="format changed"):
        parse_archive("<HTML><BODY>nothing here</BODY></HTML>")
