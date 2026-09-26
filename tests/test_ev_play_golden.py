"""Characterisation of `scripts/ev_play.py`, frozen before its refactor.

The advisor is being split into advise() / render() / save() so a terminal
front end can call the same code. That split must change nothing, and "the
tests still pass" is not evidence of that - so each scenario here pins BOTH
what is printed, byte for byte, and what is saved: the verdict's numbers
(EV, break-even, sales, stability, sensitivity) and the lines.

Every input is frozen: collected data truncated at draw 3209, the clock,
and the working directory, so nothing here reads the live tail or writes the
real outputs/predictions/latest.json.

Regenerate ONLY for a deliberate behaviour change, and say so in the commit:
    UPDATE_GOLDEN=1 PYTHONPATH=. python -m pytest tests/test_ev_play_golden.py
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

import lottery.ev as ev
from scripts import ev_play

GOLDEN_DIR = Path(__file__).parent / "golden" / "ev_play"
THROUGH = 3209
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"

# (name, argv, moment in London)
SCENARIOS = [
    ("ordinary_saturday_open", [], (2026, 9, 26, 12, 0)),
    ("ordinary_after_close_stale", [], (2026, 9, 26, 22, 5)),
    ("force_five_lines", ["--force"], (2026, 9, 26, 12, 0)),
    ("force_three_lines_seeded", ["--force", "--lines", "3", "--seed", "7"],
     (2026, 9, 26, 12, 0)),
    ("whatif_wednesday_mbw_skip", ["--jackpot", "8636787", "--roll-down"],
     (2026, 10, 7, 10, 0)),
    ("whatif_mbw_play", ["--jackpot", "8636787", "--roll-down", "--tickets",
                         "6000000"], (2026, 10, 7, 10, 0)),
    ("whatif_big_ordinary", ["--jackpot", "20000000", "--ordinary"],
     (2026, 9, 26, 12, 0)),
]


def _frozen_datetime(moment: _dt.datetime):
    class Frozen(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return moment.astimezone(tz) if tz else moment.replace(tzinfo=None)
    return Frozen


def _frozen_date(today: _dt.date):
    class Frozen(_dt.date):
        @classmethod
        def today(cls):
            return today
    return Frozen


def freeze(tmp_path, monkeypatch, when):
    """Collected data truncated at THROUGH, a frozen clock and code version,
    and tmp_path as the working directory. Shared with the terminal's tests."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    for name in ("prize_tiers.csv", "draw_pools.csv"):
        df = pd.read_csv(Path(__file__).parent.parent / "data" / name)
        df[df["draw_number"] <= THROUGH].to_csv(data / name, index=False)
    monkeypatch.chdir(tmp_path)

    moment = _dt.datetime(*when, tzinfo=ev.UK_TZ)
    monkeypatch.setattr(ev, "datetime", _frozen_datetime(moment))
    monkeypatch.setattr(ev, "date", _frozen_date(moment.date()))
    monkeypatch.setattr(ev_play, "datetime", _frozen_datetime(moment))
    # The commit changes with every commit; what is pinned is that it is saved.
    monkeypatch.setattr(ev_play, "code_version",
                        lambda: {"git_sha": "0000000", "git_dirty": False})
    return moment


def run_ev_play(tmp_path, monkeypatch, argv, when) -> tuple[str, dict | None]:
    """ev_play.main() on frozen data and a frozen clock; (stdout, latest.json)."""
    freeze(tmp_path, monkeypatch, when)
    monkeypatch.setattr(sys, "argv", ["ev_play.py", *argv])

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ev_play.main()
    latest = tmp_path / "outputs" / "predictions" / "latest.json"
    return out.getvalue(), (json.loads(latest.read_text()) if latest.exists() else None)


@pytest.mark.parametrize("name,argv,when", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_ev_play_is_unchanged(name, argv, when, tmp_path, monkeypatch):
    stdout, latest = run_ev_play(tmp_path, monkeypatch, argv, when)
    text_file = GOLDEN_DIR / f"{name}.txt"
    json_file = GOLDEN_DIR / f"{name}.json"
    if UPDATE:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        text_file.write_text(stdout)
        if latest is None:
            json_file.unlink(missing_ok=True)
        else:
            json_file.write_text(json.dumps(latest, indent=2, sort_keys=True) + "\n")
        pytest.skip("golden files regenerated")

    assert stdout == text_file.read_text()
    if latest is None:
        assert not json_file.exists(), "this scenario used to save latest.json"
        return
    want = json.loads(json_file.read_text())
    # The numbers first, by name, so a failure says WHICH figure moved.
    got_v, want_v = latest["metadata"]["verdict"], want["metadata"]["verdict"]
    for key in ("play", "ev_best_line", "break_even_jackpot", "model_stability",
                "sales_sensitivity", "conditions", "reference_line"):
        assert got_v.get(key) == want_v.get(key), key
    assert latest["predictions"] == want["predictions"]
    assert latest == want


def test_every_scenario_has_its_golden_files():
    if UPDATE:
        pytest.skip("regenerating")
    for name, _, _ in SCENARIOS:
        assert (GOLDEN_DIR / f"{name}.txt").exists(), name


@pytest.fixture(autouse=True)
def _no_stray_outputs():
    """The harness must never touch the real outputs/ directory."""
    real = Path("outputs/predictions/latest.json").resolve()
    before = real.stat().st_mtime if real.exists() else None
    yield
    after = real.stat().st_mtime if real.exists() else None
    assert before == after, "a golden run wrote the real latest.json"
