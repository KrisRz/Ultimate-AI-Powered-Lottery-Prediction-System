"""Tests for the +EV email alert.

This path fires on the ~9 draws a year worth playing, and the cloud collector
runs it with no ev_play.py and no outputs/ directory - so the email has to
stand on its own. Everything here guards that.
"""

from datetime import date

import pytest

from lottery.ev import DrawConditions, should_play
from scripts.monitoring import ev_alert

MBW = DrawConditions(jackpot=12_800_000, roll_down=True, tickets_sold=7_457_262)
DRAW = date(2026, 8, 8)


def _alert(cond=MBW, draw=DRAW, n_lines=5):
    return ev_alert.build_alert(cond, should_play(cond), draw, n_lines)


class TestAlertIsSelfContained:
    def test_subject_names_the_draw_and_the_edge(self):
        subject, _ = _alert()
        assert "2026-08-08" in subject
        assert "+0.47" in subject          # the fixture's EV, not a round number

    def test_body_carries_the_lines_to_play(self):
        _, body = _alert()
        assert "Lines to play (5 x £2 = £10.00):" in body
        numbers = [ln for ln in body.splitlines() if "EV £+" in ln and "Best-line" not in ln]
        assert len(numbers) == 5

    def test_body_carries_a_ready_to_paste_record_command(self):
        _, body = _alert()
        assert f"--draw-date {DRAW}" in body
        assert body.count(";") >= 4          # five lines, semicolon-separated

    def test_body_states_the_conditions_that_justify_playing(self):
        _, body = _alert()
        assert "Must-Be-Won:          YES" in body
        assert "£12,800,000" in body
        assert "Break-even jackpot:" in body


class TestRetryDoesNotContradictTheFirstEmail:
    """collect.yml runs twice per draw (evening + next-morning retry). Two
    emails proposing different lines would be worse than one."""

    def test_same_draw_gives_the_same_lines(self):
        assert _alert()[1] == _alert()[1]

    def test_different_draws_give_different_lines(self):
        a = _alert(draw=date(2026, 8, 8))[1]
        b = _alert(draw=date(2026, 8, 12))[1]
        assert a != b


class TestAlertSurvivesAPortfolioFailure:
    def test_email_still_goes_out_without_lines(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("constraints unsatisfiable")
        monkeypatch.setattr(ev_alert, "build_portfolio", boom)

        subject, body = _alert()
        assert "+EV ALERT" in subject
        assert "Could not build a portfolio" in body
        assert "£12,800,000" in body          # the verdict still reaches you


class TestSkipStaysSilent:
    def test_ordinary_draw_sends_nothing(self, monkeypatch, capsys):
        ordinary = DrawConditions(jackpot=4_442_277, tickets_sold=7_457_262)
        monkeypatch.setattr(ev_alert, "next_draw_conditions", lambda: ordinary)
        sent = []
        monkeypatch.setattr(ev_alert, "maybe_send_email",
                            lambda *a: sent.append(a))
        monkeypatch.delenv("EV_ALERT_TEST", raising=False)

        ev_alert.main()
        assert sent == []
        assert "SKIP" in capsys.readouterr().out
