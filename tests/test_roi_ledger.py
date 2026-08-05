"""Tests for the ROI ledger add/settle/report cycle."""

from argparse import Namespace
from datetime import date, datetime

import pandas as pd
import pytest

import scripts.roi_ledger as roi
from lottery.ev import UK_TZ


@pytest.fixture
def isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(roi, "DATA_DIR", tmp_path)
    monkeypatch.setattr(roi, "LEDGER_FILE", tmp_path / "ledger.csv")
    monkeypatch.setattr(roi, "FULL_HISTORY_FILE", tmp_path / "lotto_full_history.csv")
    monkeypatch.setattr(roi, "PRIZE_TIERS_FILE", tmp_path / "prize_tiers.csv")
    return tmp_path


def _write_history(tmp_path, rows):
    pd.DataFrame(rows).to_csv(tmp_path / "lotto_full_history.csv", index=False)


HISTORY_ROW_R1 = {
    "Draw Date": "2026-07-18", "Number_1": 22, "Number_2": 32, "Number_3": 34,
    "Number_4": 47, "Number_5": 52, "Number_6": 55, "Bonus": 10, "Jackpot": 0,
    "JackpotWins": 0, "Machine": "Lotto 4", "Ball Set": "3",
    "DrawNumber": 3190, "Round": 1,
}
HISTORY_ROW_R2 = {**HISTORY_ROW_R1, "Number_1": 20, "Number_2": 29, "Number_3": 30,
                  "Number_4": 42, "Number_5": 46, "Number_6": 55, "Bonus": 38,
                  "Machine": "Lotto 5", "Ball Set": "4", "Round": 2}


class TestParseLines:
    def test_parses_semicolon_separated(self):
        lines = roi._parse_lines("6 5 4 3 2 1; 7,8,9,10,11,12")
        assert lines == [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]

    def test_rejects_bad_line(self):
        with pytest.raises(ValueError):
            roi._parse_lines("1 2 3 4 5")
        with pytest.raises(ValueError):
            roi._parse_lines("1 2 3 4 5 60")


class TestNextDrawDate:
    def test_non_draw_day_points_at_the_next_draw(self):
        assert roi._next_draw_date(date(2026, 7, 19)).isoformat() == "2026-07-22"  # Sun -> Wed
        assert roi._next_draw_date(date(2026, 7, 21)).isoformat() == "2026-07-22"  # Tue -> Wed

    def test_on_a_draw_day_lines_belong_to_tonights_draw(self):
        # The bug this replaces: a bare date on a draw day resolved to the NEXT
        # draw, so lines bought on Wednesday afternoon were filed under Saturday
        # and settled against results they were never entered in. A bare date
        # means start-of-day, i.e. sales open.
        assert roi._next_draw_date(date(2026, 7, 22)).isoformat() == "2026-07-22"  # Wed -> Wed
        assert roi._next_draw_date(date(2026, 7, 25)).isoformat() == "2026-07-25"  # Sat -> Sat

    def test_after_sales_close_it_rolls_to_the_following_draw(self):
        after_close = datetime(2026, 7, 22, 21, 0, tzinfo=UK_TZ)   # Wed, draw done
        assert roi._next_draw_date(after_close).isoformat() == "2026-07-25"


class TestAddSettleReport:
    def test_full_cycle_two_rounds(self, isolated_ledger, capsys):
        _write_history(isolated_ledger, [HISTORY_ROW_R1, HISTORY_ROW_R2])

        # Line matches 3 in round 1 (22, 32, 34) and 0 in round 2
        roi.cmd_add(Namespace(
            draw_date="2026-07-18", lines="22 32 34 1 2 3", from_latest=False,
            cost_per_line=2.0,
        ))
        roi.cmd_settle(Namespace())
        ledger = pd.read_csv(isolated_ledger / "ledger.csv")
        row = ledger.iloc[0]
        assert bool(row["settled"]) is True
        assert int(row["matches_r1"]) == 3
        assert int(row["matches_r2"]) == 0
        # New-rules estimate for match-3 (no prize_tiers.csv present): the BASE
        # prize of £10, not the £24 a roll-down draw pays
        assert float(row["prize"]) == pytest.approx(10.0)

        roi.cmd_report(Namespace())
        out = capsys.readouterr().out
        assert "Total spent:    £2.00" in out
        assert "Total won:      £10.00" in out

    def test_settle_uses_actual_tier_prizes_when_available(self, isolated_ledger):
        _write_history(isolated_ledger, [HISTORY_ROW_R1, HISTORY_ROW_R2])
        pd.DataFrame([{
            "draw_number": 3190, "draw_date": "2026-07-18", "round": 1, "tier": 5,
            "winners": 82350, "prize_total": 1976400.0, "rollover": False,
            "rollover_count": 0, "next_jackpot_estimate": 2e6,
            "next_jackpot_roll_down": False,
        }]).to_csv(isolated_ledger / "prize_tiers.csv", index=False)

        roi.cmd_add(Namespace(
            draw_date="2026-07-18", lines="22 32 34 1 2 3", from_latest=False,
            cost_per_line=2.0,
        ))
        roi.cmd_settle(Namespace())
        ledger = pd.read_csv(isolated_ledger / "ledger.csv")
        # Match-3 in round 1 priced from actual tier data; 0 matches in round 2
        assert float(ledger.iloc[0]["prize"]) == pytest.approx(1976400.0 / 82350)
        assert ledger.iloc[0]["prize_source"] == "actual+none"

    def test_unsettled_when_draw_not_in_history(self, isolated_ledger):
        _write_history(isolated_ledger, [HISTORY_ROW_R1])
        roi.cmd_add(Namespace(
            draw_date="2099-01-02", lines="1 2 3 4 5 6", from_latest=False,
            cost_per_line=2.0,
        ))
        roi.cmd_settle(Namespace())
        ledger = pd.read_csv(isolated_ledger / "ledger.csv")
        assert bool(ledger.iloc[0]["settled"]) is False

    def test_unsettled_when_only_one_round_collected(self, isolated_ledger):
        # 2026-format draws must not settle from partial (single-round) data
        _write_history(isolated_ledger, [HISTORY_ROW_R1])
        roi.cmd_add(Namespace(
            draw_date="2026-07-18", lines="22 32 34 1 2 3", from_latest=False,
            cost_per_line=2.0,
        ))
        roi.cmd_settle(Namespace())
        ledger = pd.read_csv(isolated_ledger / "ledger.csv")
        assert bool(ledger.iloc[0]["settled"]) is False
