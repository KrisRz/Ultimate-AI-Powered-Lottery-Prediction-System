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


# --- decision provenance --------------------------------------------------

def test_provenance_columns_are_added_to_an_old_ledger(tmp_path, monkeypatch):
    """A ledger written before provenance existed must still load.

    Old rows keep their tickets and settlements and simply carry no
    verdict; the columns appear so later writes keep a stable order.
    """
    import pandas as pd
    import scripts.roi_ledger as rl

    old = tmp_path / "ledger.csv"
    pd.DataFrame([{
        "added_at": "2026-08-08T12:00:00", "draw_date": "2026-08-08",
        "line": "1 2 3 4 5 6", "cost": 2.0, "settled": True,
        "matches_r1": 2, "bonus_r1": False, "matches_r2": 1,
        "bonus_r2": False, "prize": 0.0, "prize_source": "table",
    }]).to_csv(old, index=False)
    monkeypatch.setattr(rl, "LEDGER_FILE", old)

    ledger = rl._load_ledger()
    assert list(ledger.columns) == rl.LEDGER_COLUMNS
    assert ledger["git_sha"].isna().all()
    assert ledger.loc[0, "line"] == "1 2 3 4 5 6"


def test_provenance_reads_the_saved_verdict(tmp_path, monkeypatch):
    """The row must record WHY, not just what."""
    import json
    import scripts.roi_ledger as rl

    latest = tmp_path / "latest.json"
    latest.write_text(json.dumps({"metadata": {"verdict": {
        "ev_best_line": 0.0241,
        "break_even_jackpot": 14_500_000.0,
        "model_stability": {"label": "MODEL-SENSITIVE",
                            "ev_spec_min": -0.0532, "ev_spec_max": 0.0409},
        "conditions": {"jackpot_event_pool": 15_000_000.0,
                       "tickets_sold": 12_000_000, "roll_down": True,
                       "rounds": 2},
    }}}))
    monkeypatch.setattr(rl, "LATEST_PREDICTIONS", latest)

    p = rl._provenance()
    assert p["jackpot"] == 15_000_000.0
    assert p["model_stability"] == "MODEL-SENSITIVE"
    assert p["ev_spec_min"] == pytest.approx(-0.0532)
    assert p["ev_spec_max"] == pytest.approx(0.0409)
    assert p["roll_down"] is True


def test_provenance_survives_a_missing_or_broken_verdict(tmp_path, monkeypatch):
    """Recording a ticket must never fail because the verdict file is gone.

    Buying is the irreversible act; losing the reason is bad, losing the
    ticket record is worse.
    """
    import scripts.roi_ledger as rl

    monkeypatch.setattr(rl, "LATEST_PREDICTIONS", tmp_path / "absent.json")
    p = rl._provenance()
    assert set(p) == set(rl.PROVENANCE_COLUMNS)
    assert p["ev_best_line"] is None

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    monkeypatch.setattr(rl, "LATEST_PREDICTIONS", broken)
    assert rl._provenance()["ev_best_line"] is None


def test_git_sha_is_recorded_or_blank():
    """Pins the code that produced the verdict; never raises."""
    import scripts.roi_ledger as rl
    sha = rl._git_sha()
    assert isinstance(sha, str)
    if sha:
        assert 6 <= len(sha) <= 12 and all(c in "0123456789abcdef" for c in sha)


def _verdict_file(tmp_path, provenance=None):
    import json
    latest = tmp_path / "latest.json"
    meta = {"verdict": {
        "ev_best_line": -1.03, "break_even_jackpot": 30_819_901.0,
        "model_stability": {"label": "ROBUST SKIP",
                            "ev_spec_min": -1.046, "ev_spec_max": -1.027},
        "conditions": {"jackpot_event_pool": 5_840_169.0,
                       "tickets_sold": 5_100_807, "roll_down": False, "rounds": 2},
    }}
    if provenance is not None:
        meta["provenance"] = provenance
    latest.write_text(json.dumps({"metadata": meta}))
    return latest


def test_git_sha_comes_from_the_verdict_not_from_head(tmp_path, monkeypatch):
    """HEAD at purchase time can be a commit that never priced this draw -
    open problem B from the 2026-09-19 handoff."""
    import scripts.roi_ledger as rl
    monkeypatch.setattr(rl, "LATEST_PREDICTIONS", _verdict_file(tmp_path, {
        "advice": "MARGINAL", "draw_date": "2026-10-07",
        "git_sha": "abc1234", "git_dirty": False}))
    monkeypatch.setattr(rl, "_git_sha", lambda: "HEADSHA")
    p = rl._provenance(date(2026, 10, 7))
    assert p["git_sha"] == "abc1234"
    assert p["advice"] == "MARGINAL"
    assert p["provenance_status"] == rl.PROVENANCE_COMPLETE


def test_a_dirty_tree_is_recorded_as_such(tmp_path, monkeypatch):
    import scripts.roi_ledger as rl
    monkeypatch.setattr(rl, "LATEST_PREDICTIONS", _verdict_file(tmp_path, {
        "advice": "SKIP", "draw_date": "2026-09-30",
        "git_sha": "abc1234", "git_dirty": True}))
    assert rl._provenance(date(2026, 9, 30))["git_sha"] == "abc1234+dirty"


def test_a_verdict_for_another_draw_is_refused(tmp_path, monkeypatch):
    import scripts.roi_ledger as rl
    monkeypatch.setattr(rl, "LATEST_PREDICTIONS", _verdict_file(tmp_path, {
        "advice": "SKIP", "draw_date": "2026-09-26",
        "git_sha": "abc1234", "git_dirty": False}))
    p = rl._provenance(date(2026, 9, 30))
    assert p["provenance_status"] == rl.PROVENANCE_MISSING
    assert p["ev_best_line"] is None and p["git_sha"] is None
    assert p["mismatch"] == "2026-09-26"


def test_a_verdict_saved_before_provenance_falls_back_to_head(tmp_path, monkeypatch):
    import scripts.roi_ledger as rl
    monkeypatch.setattr(rl, "LATEST_PREDICTIONS", _verdict_file(tmp_path))
    monkeypatch.setattr(rl, "_git_sha", lambda: "HEADSHA")
    p = rl._provenance(date(2026, 9, 30))
    assert p["git_sha"] == "HEADSHA" and p["advice"] is None
    assert p["provenance_status"] == rl.PROVENANCE_COMPLETE


def test_the_real_ledger_still_reads_and_reports(capsys):
    """Backward compatibility on the file that holds real money: the local
    ledger (no `advice` column) must load and report the same totals."""
    import scripts.roi_ledger as rl
    if not rl.LEDGER_FILE.exists():
        pytest.skip("no local ledger on this machine (CI)")
    ledger = rl._load_ledger()
    assert "advice" in ledger.columns
    assert list(ledger.columns[:len(rl.LEDGER_COLUMNS)]) == rl.LEDGER_COLUMNS


@pytest.mark.parametrize("sha", ["8201337", "1e45678", "0000000"])
def test_a_numeric_looking_sha_survives_a_round_trip(tmp_path, monkeypatch, sha):
    """pandas would read these as 8201337, inf and 0, and the next write
    would record a commit that never existed."""
    import scripts.roi_ledger as rl
    ledger = tmp_path / "ledger.csv"
    monkeypatch.setattr(rl, "LEDGER_FILE", ledger)
    row = {c: None for c in rl.LEDGER_COLUMNS}
    row.update({"added_at": "2026-09-26T12:00:00", "draw_date": "2026-09-26",
                "line": "1 2 3 4 5 6", "cost": 2.0, "settled": False, "git_sha": sha})
    pd.DataFrame([row]).to_csv(ledger, index=False)
    once = rl._load_ledger()
    once.to_csv(ledger, index=False)
    assert rl._load_ledger().loc[0, "git_sha"] == sha
