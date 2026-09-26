"""The emailed ticket: valid lines, the same lines as ./lotto, and honest text."""

import pytest

from scripts import ev_play
from scripts.monitoring import ticket_mail
from tests.test_ev_play_golden import freeze

SAT_NOON = (2026, 9, 26, 12, 0)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    freeze(tmp_path, monkeypatch, SAT_NOON)


def _valid(line):
    return len(line) == 6 and len(set(line)) == 6 and all(1 <= n <= 59 for n in line)


def test_five_valid_lines_the_same_as_lotto_ticket(frozen):
    subject, body, tickets = ticket_mail.build_ticket_mail("portfolio", 5)
    assert len(tickets) == 5 and all(_valid(t) for t in tickets)
    assert tickets == [p["line"] for p in ev_play.advise(force=True).portfolio]
    assert "model says SKIP" in subject
    assert "Entertainment, not investment" in body
    assert "no pattern to use" in body                 # never sold as prediction
    assert '--lines "' + "; ".join(" ".join(map(str, t)) for t in tickets) in body


def test_the_wheel_is_six_valid_tickets(frozen):
    _, body, tickets = ticket_mail.build_ticket_mail("wheel")
    assert len(tickets) == 6 and all(_valid(t) for t in tickets)
    assert "guaranteed Match 3" in body


def test_a_variant_is_a_different_reproducible_set(frozen):
    base = ticket_mail.build_ticket_mail("portfolio", 5, 0)[2]
    one = ticket_mail.build_ticket_mail("portfolio", 5, 1)[2]
    assert one != base
    assert one == ticket_mail.build_ticket_mail("portfolio", 5, 1)[2]


def test_a_gate_problem_is_in_the_mail(frozen):
    _, body, _ = ticket_mail.build_ticket_mail(gate=["pool file missing"])
    assert "WARNING: the data gate reported: pool file missing" in body


@pytest.mark.parametrize("env", [{"TICKET_KIND": "lottery"}, {"TICKET_LINES": "0"},
                                 {"TICKET_LINES": "11"}, {"TICKET_VARIANT": "-1"}])
def test_bad_input_sends_nothing(frozen, monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sent = []
    monkeypatch.setattr(ticket_mail, "maybe_send_email", lambda *a: sent.append(a))
    assert ticket_mail.main() == 1 and sent == []


def test_main_sends_one_mail(frozen, monkeypatch):
    for k in ("TICKET_KIND", "TICKET_LINES", "TICKET_VARIANT"):
        monkeypatch.delenv(k, raising=False)
    sent = []
    monkeypatch.setattr(ticket_mail, "maybe_send_email", lambda *a: sent.append(a))
    assert ticket_mail.main() == 0
    assert len(sent) == 1 and sent[0][0].startswith("LOTTO ticket: 5 lines")
