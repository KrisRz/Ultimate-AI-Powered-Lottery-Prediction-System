"""Tests for the EV advisor CLI - what it writes, and when it must not."""

import json
import sys

import pandas as pd
import pytest

from scripts import ev_play


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """Run the CLI in an empty tree so it writes nowhere near real outputs."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    return tmp_path


def _run(*argv):
    sys.argv = ["ev_play.py", *argv]
    ev_play.main()


class TestLatestJsonIsOnlyTheRealVerdict:
    """`roi_ledger.py add --from-latest` records latest.json as really played,
    so a hypothetical run must never be able to leave a PLAY portfolio there."""

    def test_real_run_writes_latest(self, isolated_cwd):
        _run()
        payload = json.load(open(isolated_cwd / "outputs/predictions/latest.json"))
        assert payload["metadata"]["verdict"]["play"] is False

    @pytest.mark.parametrize("override", (
        ["--jackpot", "50000000"],
        ["--roll-down"],
        ["--tickets", "1000"],
    ))
    def test_what_if_run_writes_nothing(self, isolated_cwd, override):
        _run(*override)
        assert not (isolated_cwd / "outputs/predictions/latest.json").exists()

    def test_what_if_cannot_overwrite_an_existing_verdict(self, isolated_cwd):
        _run()
        before = (isolated_cwd / "outputs/predictions/latest.json").read_text()
        _run("--jackpot", "50000000", "--roll-down")   # a screaming PLAY
        assert (isolated_cwd / "outputs/predictions/latest.json").read_text() == before


class TestNextDrawConditions:
    def test_reads_jackpot_prizes_and_sales_from_collected_data(self, isolated_cwd):
        rows = []
        for draw in (3191, 3192, 3193):
            for rnd in (1, 2):
                for tier, winners, per_winner in (
                    (3, 40, 1_000.0), (4, 3_400, 50.0),
                    (5, 78_000, 10.0), (6, 730_000, 1.0),
                ):
                    rows.append({
                        "draw_number": draw, "draw_date": "2026-07-25", "round": rnd,
                        "tier": tier, "winners": winners,
                        "prize_total": winners * per_winner, "rollover": True,
                        "rollover_count": 2, "next_jackpot_estimate": 4_442_277.0,
                        "next_jackpot_roll_down": False,
                    })
        pd.DataFrame(rows).to_csv(isolated_cwd / "data/prize_tiers.csv", index=False)

        cond = ev_play.next_draw_conditions()
        assert cond.jackpot == pytest.approx(4_442_277.0)
        assert cond.roll_down is False
        assert cond.prizes.match_3 == 10.0
        assert cond.prizes.match_2 == 1.0
        assert cond.prizes.source.startswith("observed")
        assert 6_000_000 < cond.tickets_sold < 9_000_000
