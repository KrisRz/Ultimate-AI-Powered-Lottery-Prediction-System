"""Characterisation of `scripts/wheel_play.py`, frozen before its refactor.

Same contract as tests/test_ev_play_golden.py: printout byte for byte and
the saved portfolio, on data truncated at 3209 and a frozen clock.
    UPDATE_GOLDEN=1 PYTHONPATH=. python -m pytest tests/test_wheel_play_golden.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

import pytest

from scripts import wheel_play
from tests.test_ev_play_golden import THROUGH, _frozen_date, _frozen_datetime

GOLDEN_DIR = Path(__file__).parent / "golden" / "wheel_play"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"
SCENARIOS = [
    ("default_pool_12", []),
    ("eight_lines", ["--lines", "8"]),
    ("pool_11", ["--pool-size", "11"]),
]


def run_wheel(tmp_path, monkeypatch, argv):
    import datetime as _dt

    import pandas as pd

    import lottery.ev as ev
    from scripts import ev_play
    data = tmp_path / "data"
    data.mkdir()
    for name in ("prize_tiers.csv", "draw_pools.csv"):
        df = pd.read_csv(Path("data") / name)
        df[df["draw_number"] <= THROUGH].to_csv(data / name, index=False)
    monkeypatch.chdir(tmp_path)
    moment = _dt.datetime(2026, 9, 26, 12, 0, tzinfo=ev.UK_TZ)
    monkeypatch.setattr(ev, "datetime", _frozen_datetime(moment))
    monkeypatch.setattr(ev, "date", _frozen_date(moment.date()))
    monkeypatch.setattr(ev_play, "datetime", _frozen_datetime(moment))
    monkeypatch.setattr(wheel_play, "datetime", _frozen_datetime(moment))
    monkeypatch.setattr(sys, "argv", ["wheel_play.py", *argv])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        wheel_play.main()
    saved = list((tmp_path / "outputs" / "predictions").glob("wheel_portfolio_*.json"))
    assert len(saved) == 1
    assert not (tmp_path / "outputs" / "predictions" / "latest.json").exists()
    return out.getvalue(), json.loads(saved[0].read_text())


@pytest.mark.parametrize("name,argv", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_wheel_play_is_unchanged(name, argv, tmp_path, monkeypatch):
    stdout, saved = run_wheel(tmp_path, monkeypatch, argv)
    text_file, json_file = GOLDEN_DIR / f"{name}.txt", GOLDEN_DIR / f"{name}.json"
    if UPDATE:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        text_file.write_text(stdout)
        json_file.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
        pytest.skip("golden files regenerated")
    assert stdout == text_file.read_text()
    want = json.loads(json_file.read_text())
    assert saved["predictions"] == want["predictions"]
    assert saved == want
