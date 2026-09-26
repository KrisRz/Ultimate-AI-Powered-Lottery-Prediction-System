"""The terminal is a front end, not a second implementation.

The proof is the golden files: `lotto play` must print exactly what
`make play` printed on the same frozen inputs, `lotto ticket --yes` exactly
what `make ticket` did, and both must save the same latest.json. A SKIP must
never turn into a ticket without either a human's yes or an explicit --yes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import lotto
from tests.test_ev_play_golden import GOLDEN_DIR, freeze

WHEEL_GOLDEN = GOLDEN_DIR.parent / "wheel_play"
SAT_NOON = (2026, 9, 26, 12, 0)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    freeze(tmp_path, monkeypatch, SAT_NOON)
    # No terminal: the override and the ledger must not wait for input.
    monkeypatch.setattr(lotto, "_interactive", lambda: False)
    return tmp_path


def _run(capsys, *argv) -> tuple[int, str]:
    code = lotto.main(list(argv), root=None)
    return code, capsys.readouterr().out


def _latest(tmp_path):
    return json.loads((tmp_path / "outputs" / "predictions" / "latest.json").read_text())


def test_play_is_make_play(frozen, capsys):
    code, out = _run(capsys, "play")
    assert code == 0
    assert out == (GOLDEN_DIR / "ordinary_saturday_open.txt").read_text()
    assert _latest(frozen) == json.loads(
        (GOLDEN_DIR / "ordinary_saturday_open.json").read_text())


def test_ticket_with_yes_is_make_ticket(frozen, capsys):
    code, out = _run(capsys, "ticket", "--yes")
    assert code == 0
    assert out.startswith("⚠ MODEL VERDICT: ROBUST SKIP")
    golden = (GOLDEN_DIR / "force_five_lines.txt").read_text()
    assert golden in out
    assert _latest(frozen) == json.loads((GOLDEN_DIR / "force_five_lines.json").read_text())
    assert "not recorded" in out


def test_a_skip_without_yes_generates_nothing(frozen, capsys):
    code, out = _run(capsys, "ticket")
    assert code == 2
    assert "pass --yes" in out
    assert not (frozen / "outputs").exists()


def test_wheel_is_make_wheel_and_leaves_the_verdict_on_file(frozen, capsys):
    code, out = _run(capsys, "wheel", "--yes")
    assert code == 0
    printout = (WHEEL_GOLDEN / "default_pool_12.txt").read_text()
    assert printout.rsplit("Saved to", 1)[0] in out
    saved = _latest(frozen)["metadata"]["provenance"]
    assert saved["draw_date"] == "2026-09-26" and saved["advice"] == "SKIP"
    assert _latest(frozen)["predictions"] == []


def test_record_copies_the_verdict_into_the_ledger(frozen, capsys):
    code, _ = _run(capsys, "ticket", "--yes", "--record")
    assert code == 0
    ledger = pd.read_csv(frozen / "data" / "ledger.csv", dtype={"git_sha": str})
    assert len(ledger) == 5
    assert set(ledger["draw_date"]) == {"2026-09-26"}
    assert set(ledger["advice"]) == {"SKIP"}
    assert set(ledger["git_sha"]) == {"0000000"}       # the verdict's, not HEAD
    assert set(ledger["provenance_status"]) == {"complete"}


def test_whatif_never_saves(frozen, capsys):
    code, out = _run(capsys, "whatif", "--jackpot", "8636787", "--roll-down")
    assert code == 0
    assert out.strip().endswith("(what-if - nothing saved)")
    assert not (frozen / "outputs").exists()


def test_status_says_the_decision(frozen, capsys):
    code, out = _run(capsys, "status")
    assert code == 0
    assert "Next draw:    Saturday 26 Sep 2026" in out
    assert "DECISION:     SKIP" in out


class TestCheck:
    def test_scores_a_line(self, frozen, capsys):
        code, out = _run(capsys, "check", "1", "2", "3", "4", "5", "6")
        assert code == 0
        assert "Popularity:   x42.81" in out
        assert "identical for every line" in out

    @pytest.mark.parametrize("bad", [["1", "2", "3"], ["1", "1", "2", "3", "4", "5"],
                                     ["0", "2", "3", "4", "5", "6"], ["a", "b", "c", "d", "e", "f"]])
    def test_rejects_a_bad_line(self, frozen, capsys, bad):
        code, _ = _run(capsys, "check", *bad)
        assert code == 1


class TestSyncGuard:
    def test_refuses_off_main(self):
        assert "branch" in lotto.sync_refusal({"branch": "feature/x", "dirty": False})

    def test_refuses_a_dirty_tree(self):
        assert "uncommitted" in lotto.sync_refusal({"branch": "main", "dirty": True})

    def test_allows_clean_main(self):
        assert lotto.sync_refusal({"branch": "main", "dirty": False}) is None


def test_analysis_menu_is_read_only_and_excludes_the_frozen_evaluator():
    scripts = {script for script, *_ in lotto.ANALYSES.values()}
    root = Path(__file__).parent.parent
    assert all((root / s).exists() for s in scripts)
    banned = ("popularity_v2_final_test", "new_predict", "nightly_backtest",
              "validations/backtest.py", "fetch_data", "backfill", "export_site_data")
    assert not [s for s in scripts for b in banned if b in s]


def test_the_terminal_does_not_import_the_legacy_predictor():
    import subprocess
    import sys
    probe = ("import sys, scripts.lotto; print(','.join(m for m in "
             "('tensorflow', 'scripts.new_predict') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, env={"PYTHONPATH": "."})
    assert out.stdout.strip() == ""


def test_the_menu_runs_an_option_and_exits(frozen, capsys, monkeypatch):
    answers = iter(["4", "1 2 3 4 5 6", "", "3", "", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert lotto.menu() == 0
    out = capsys.readouterr().out
    assert out.count("DECISION:     SKIP") == 3          # the header, each loop
    assert "Popularity:   x42.81" in out                 # [4] check
    assert "EV ADVISOR - next UK Lotto draw" in out      # [3] make play


def test_the_menu_generates_only_after_a_yes(frozen, capsys, monkeypatch):
    monkeypatch.setattr(lotto, "_interactive", lambda: True)
    answers = iter(["1", "n", "", "0"])                  # ticket, then decline
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert lotto.menu() == 0
    assert not (frozen / "outputs" / "predictions").exists() or not any(
        (frozen / "outputs" / "predictions").glob("ev_portfolio_*"))
