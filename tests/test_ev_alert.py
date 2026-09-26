"""Tests for the +EV email alert.

This path fires on the one or two draws a year worth playing, and the cloud collector
runs it with no ev_play.py and no outputs/ directory - so the email has to
stand on its own. Everything here guards that.
"""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from lottery.ev import DrawConditions, exact_sales_baseline, mbw_uplift, should_play
from scripts.monitoring import ev_alert

@pytest.fixture(autouse=True)
def _not_the_heartbeat_run(monkeypatch):
    """collect.yml runs this suite inside the Sunday EventBridge run, where
    GitHub sets GITHUB_EVENT_NAME=workflow_dispatch - so any test calling
    main() would otherwise take the heartbeat branch on Sunday mornings only."""
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)


MBW = DrawConditions(jackpot=12_800_000, roll_down=True, tickets_sold=7_457_262)
DRAW = date(2026, 8, 8)


def _alert(cond=MBW, draw=DRAW, n_lines=5):
    return ev_alert.build_alert(cond, should_play(cond), draw, n_lines)


class TestAlertIsSelfContained:
    def test_subject_names_the_draw_and_the_edge(self):
        subject, _ = _alert()
        assert "2026-08-08" in subject
        assert "+0.47" in subject          # the fixture's EV, not a round number

    def test_subject_says_whether_the_edge_survives_the_sales_estimate(self):
        """Read on a phone, hours before sales close: "+EV whatever it sells"
        and "+EV only if sales land near the estimate" are different decisions,
        and the second was twelve lines down the body."""
        robust, _ = _alert()
        assert "PLAY (robust)" in robust

        # A pool just over break-even: +EV centrally, gone by the p75 of sales.
        marginal_cond = DrawConditions(jackpot=9_300_000, roll_down=True,
                                       tickets_sold=7_399_061,
                                       draw_date=date(2026, 9, 9))
        verdict = should_play(marginal_cond)
        assert verdict["play"] and not verdict["sales_sensitivity"]["robust"]
        subject, _ = _alert(marginal_cond, date(2026, 9, 9))
        assert "marginal" in subject and "robust" not in subject

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


class TestOperatorSecondOpinion:
    AGREES = {"phase": "MUST_BE_WON", "must_be_won": True,
              "jackpot": 12_800_000.0, "sales_close": None}

    def _with(self, operator):
        return ev_alert.build_alert(MBW, should_play(MBW), DRAW, 5, operator=operator)

    def test_agreement_is_stated_and_the_subject_untouched(self):
        subject, body = self._with(self.AGREES)
        assert subject.startswith("LOTTO +EV ALERT")
        assert "Operator's page:      agrees" in body

    def test_disagreement_leads_the_subject(self):
        subject, body = self._with({**self.AGREES, "phase": "INITIAL",
                                    "must_be_won": False})
        assert subject.startswith("CHECK FEEDS - LOTTO +EV ALERT")
        assert "DISAGREES" in body
        assert "Lines to play" in body            # still actionable

    def test_an_unreachable_page_does_not_block_the_mail(self):
        subject, body = self._with({})
        assert subject.startswith("LOTTO +EV ALERT")
        assert "unreachable" in body

    def test_main_consults_the_page_on_a_play(self, monkeypatch):
        monkeypatch.setattr(ev_alert, "next_draw_conditions", lambda: MBW)
        monkeypatch.setattr(ev_alert, "fetch_operator_page", lambda: self.AGREES)
        sent = []
        monkeypatch.setattr(ev_alert, "maybe_send_email", lambda *a: sent.append(a))
        monkeypatch.delenv("EV_ALERT_TEST", raising=False)
        ev_alert.main()
        assert sent and "Operator's page:      agrees" in sent[0][1]


def test_the_alert_does_not_load_the_legacy_predictor():
    """The PLAY email used to import TensorFlow by way of nightly_backtest ->
    backtest -> new_predict. An import failure anywhere in that legacy stack
    would have taken the one output that matters down with it."""
    import subprocess
    import sys
    probe = ("import sys; import scripts.monitoring.ev_alert; "
             "print(','.join(m for m in ('tensorflow', 'scripts.new_predict', "
             "'scripts.validations.backtest') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, env={"PYTHONPATH": "."})
    assert out.stdout.strip() == ""


# Pools as collected up to draw 3209. Pinned, not the live file: the collector
# appends every draw and would move the measured uplift under these tests.
POOLS = pd.read_csv("data/draw_pools.csv").query("draw_number <= 3209")
BEFORE_3205 = POOLS.query("draw_number < 3205")
D3205 = date(2026, 9, 9)


def _mbw_3205(**over):
    """Draw 3205 as the model priced it: a Wednesday capped Must-Be-Won at
    £7.81M, sales at the installed x1.44 on the exact Wednesday baseline."""
    baseline = exact_sales_baseline(BEFORE_3205, D3205)
    fields = dict(jackpot=7_807_591, roll_down=True, rollover_count=5,
                  tickets_sold=int(baseline * mbw_uplift(D3205)[0]),
                  draw_date=D3205)
    fields.update(over)
    return DrawConditions(**fields)


class TestMarginalTier:
    """The installed uplift is one-round-era; every two-round Must-Be-Won on
    exact sales has come in under it. The MARGINAL mail makes that
    disagreement visible without installing the lower constant."""

    def test_draw_3205_would_have_raised_a_marginal(self):
        cond = _mbw_3205()
        assert not should_play(cond)["play"]           # what the model said
        alt = ev_alert.at_measured_uplift(cond, BEFORE_3205)
        assert alt["n"] == 2 and alt["uplift"] == pytest.approx(1.127, abs=1e-3)
        assert alt["cond"].tickets_sold < cond.tickets_sold
        assert alt["verdict"]["play"]

    def test_uses_the_highest_measured_uplift(self):
        uplift, n = ev_alert.measured_uplift(POOLS)
        assert n == 3 and uplift == pytest.approx(1.152, abs=1e-3)

    def test_ordinary_and_special_draws_are_not_repriced(self):
        assert ev_alert.at_measured_uplift(_mbw_3205(roll_down=False), POOLS) is None
        assert ev_alert.at_measured_uplift(_mbw_3205(special_event=True), POOLS) is None
        assert ev_alert.at_measured_uplift(_mbw_3205(), None) is None

    def test_never_raises_the_sales_assumed(self):
        low = _mbw_3205(tickets_sold=1_000_000)
        assert ev_alert.at_measured_uplift(low, POOLS) is None

    def test_subject_says_both_verdicts(self):
        cond = _mbw_3205()
        alt = ev_alert.at_measured_uplift(cond, BEFORE_3205)
        subject, body = ev_alert.build_marginal_alert(
            cond, should_play(cond), alt, D3205, 5)
        assert subject.startswith("LOTTO MARGINAL: 2026-09-09")
        assert "installed model: SKIP" in subject and "robust" not in subject
        assert body.startswith("MARGINAL - the installed model says SKIP")
        assert "Lines to play (5 x £2" in body

    def test_main_sends_marginal_instead_of_staying_silent(self, monkeypatch, capsys):
        cond = _mbw_3205()
        monkeypatch.setattr(ev_alert, "next_draw_conditions", lambda: cond)
        monkeypatch.setattr(ev_alert.pd, "read_csv", lambda *a, **k: BEFORE_3205)
        monkeypatch.setattr(ev_alert, "fetch_operator_page", lambda: {})
        sent = []
        monkeypatch.setattr(ev_alert, "maybe_send_email", lambda *a: sent.append(a))
        monkeypatch.delenv("EV_ALERT_TEST", raising=False)
        ev_alert.main()
        assert len(sent) == 1 and "MARGINAL" in sent[0][0]
        assert "[ev-alert] MARGINAL" in capsys.readouterr().out


class TestWeeklyHeartbeat:
    SUN_0605 = datetime(2026, 9, 27, 6, 5, tzinfo=timezone.utc)

    def test_due_only_on_the_eventbridge_sunday_morning_run(self):
        assert ev_alert.heartbeat_due("workflow_dispatch", self.SUN_0605)
        # GitHub's own cron fires the same retry around 11:00 UTC
        assert not ev_alert.heartbeat_due("schedule", self.SUN_0605)
        assert not ev_alert.heartbeat_due(
            "workflow_dispatch", self.SUN_0605.replace(hour=11))
        # the Thursday retry and the draw-night runs stay silent
        assert not ev_alert.heartbeat_due(
            "workflow_dispatch", datetime(2026, 9, 24, 6, 5, tzinfo=timezone.utc))
        assert not ev_alert.heartbeat_due(None, self.SUN_0605)

    def _main(self, monkeypatch, event, now):
        ordinary = DrawConditions(jackpot=2_000_000, tickets_sold=5_100_000,
                                  draw_date=date(2026, 9, 30))
        monkeypatch.setattr(ev_alert, "next_draw_conditions", lambda: ordinary)
        monkeypatch.setattr(ev_alert, "heartbeat_due",
                            lambda e: e == "workflow_dispatch" and now.weekday() == 6)
        monkeypatch.setenv("GITHUB_EVENT_NAME", event)
        monkeypatch.delenv("EV_ALERT_TEST", raising=False)
        sent = []
        monkeypatch.setattr(ev_alert, "maybe_send_email", lambda *a: sent.append(a))
        ev_alert.main()
        return sent

    def test_sunday_skip_sends_one_status_mail(self, monkeypatch):
        sent = self._main(monkeypatch, "workflow_dispatch", self.SUN_0605)
        assert len(sent) == 1
        subject, body = sent[0]
        assert subject.startswith("LOTTO weekly:") and "SKIP" in subject
        assert "If this mail stops arriving on Sundays" in body

    def test_other_runs_stay_silent_on_skip(self, monkeypatch):
        assert self._main(monkeypatch, "schedule", self.SUN_0605) == []

    def test_stale_data_leads_the_subject(self):
        cond = DrawConditions(jackpot=2_000_000, tickets_sold=5_100_000)
        subject, body = ev_alert.build_heartbeat(
            cond, should_play(cond), date(2026, 9, 30), date(2026, 9, 23),
            date(2026, 9, 26), None, None)
        assert "DATA BEHIND" in subject
        assert "NOT COLLECTED:        the 2026-09-26 draw" in body
