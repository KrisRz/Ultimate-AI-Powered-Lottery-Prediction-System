"""Offline tests for scripts/fetch_sales.py (no network)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts.fetch_sales import (
    attach_draw_numbers,
    parse_sales,
    ticket_price,
    winner_count_estimates,
)
from lottery.ev import P_MATCH_3

SAMPLE = """<HTML><HEAD><TITLE>2026 Lottery Ticket Sales Figures</TITLE>
<BODY><PRE>2026  Lottery Ticket Sales Figures

Day,DD,MMM,YYYY,   Sales  ,   %Chg
Wed, 5,Aug,2026,  10523874,   +2.4
Wed, 5,Aug,2026,  10523874,   +2.4
Sat, 1,Aug,2026,  17072084,   +1.6
Sat, 1,Aug,2026,  17072084,   +1.6
Sat, 5,Oct,2013,  35000000,   +9.9
Wed, 2,Oct,2013,  20000000,    0.0
Sat,19,Nov,1994,  48965792,    0.0

Data obtained from http://lottery.merseyworld.com/
</PRE></BODY></HTML>"""


def test_parse_dedupes_and_orders():
    df = parse_sales(SAMPLE)
    assert list(df["draw_date"]) == [
        date(1994, 11, 19), date(2013, 10, 2), date(2013, 10, 5),
        date(2026, 8, 1), date(2026, 8, 5),
    ]
    assert df["sales_gbp"].tolist() == [
        48965792, 20000000, 35000000, 17072084, 10523874]


def test_lines_follow_price_eras():
    df = parse_sales(SAMPLE).set_index("draw_date")
    assert ticket_price(date(2013, 10, 2)) == 1.0
    assert ticket_price(date(2013, 10, 5)) == 2.0
    assert df.loc[date(2013, 10, 2), "lines_sold"] == 20000000      # 1 GBP era
    assert df.loc[date(2013, 10, 5), "lines_sold"] == 17500000      # 2 GBP era
    assert df.loc[date(2026, 8, 5), "lines_sold"] == 5261937


def test_parse_rejects_unrecognisable_page():
    with pytest.raises(ValueError):
        parse_sales("<HTML><PRE>nothing here</PRE></HTML>")


def test_attach_draw_numbers(tmp_path: Path):
    hist = tmp_path / "hist.csv"
    # 2-round era style: two rows per draw, one per round
    pd.DataFrame({
        "Draw Date": ["2026-08-05", "2026-08-05", "2026-08-01"],
        "DrawNumber": [3195, 3195, 3194],
    }).to_csv(hist, index=False)
    df = parse_sales(SAMPLE)
    df = attach_draw_numbers(df, history_file=hist)
    by_date = df.set_index("draw_date")["draw_number"]
    assert by_date[date(2026, 8, 5)] == 3195
    assert by_date[date(2026, 8, 1)] == 3194
    assert pd.isna(by_date[date(1994, 11, 19)])


def test_winner_count_estimate_inverts_probability(tmp_path: Path):
    n_lines = 8_000_000
    tiers = tmp_path / "tiers.csv"
    pd.DataFrame({
        "draw_number": [3195] * 2,
        "tier": [5, 5],                       # Match 3 in both rounds
        "winners": [int(n_lines * P_MATCH_3)] * 2,
    }).to_csv(tiers, index=False)
    est = winner_count_estimates(tiers_files=(tiers,))
    assert len(est) == 1
    # int() floors the winner count, so the round-trip is only ~1e-5 exact
    assert est.loc[0, "est_lines"] == pytest.approx(n_lines, rel=1e-4)


def test_validate_reports_ratio(tmp_path: Path, capsys):
    n_lines = 8_000_000
    tiers = tmp_path / "tiers.csv"
    pd.DataFrame({
        "draw_number": [3195],
        "tier": [5],
        "winners": [int(n_lines * P_MATCH_3)],
    }).to_csv(tiers, index=False)
    sales = pd.DataFrame({
        "draw_date": [date(2026, 8, 5)],
        "draw_number": pd.array([3195], dtype="Int64"),
        "sales_gbp": [n_lines * 2],
        "lines_sold": [n_lines],
        "pct_chg": [0.0],
    })
    # validate() reads the module-level data paths; point them at tmp_path
    import unittest.mock as mock

    import scripts.fetch_sales as fs
    with mock.patch.object(fs, "TIERS_HISTORY_FILE", tiers), \
         mock.patch.object(fs, "TIERS_LIVE_FILE", tmp_path / "absent.csv"):
        merged = fs.validate(sales)
    assert merged["ratio"].iloc[0] == pytest.approx(1.0, rel=1e-3)
    assert "median 1.000" in capsys.readouterr().out
