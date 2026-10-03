#!/usr/bin/env python3
"""Email alert when the next draw clears the EV threshold (PLAY verdict).

Runs at the end of the post-draw routine. Four kinds of mail, and nothing
else:

- PLAY: the draw clears the threshold on the installed model.
- MARGINAL: a Must-Be-Won draw the installed model calls SKIP, but which
  clears it at the sales uplift MEASURED in the two-round era. The installed
  constant stays until the evidence rule allows it to move (n >= 4, see
  CLAUDE.md); this makes the disagreement visible instead of silent. Draw 3205
  was +GBP 0.06 a line after the fact while the model said SKIP.
- HEARTBEAT: one short status mail on Sunday morning, sent by the
  EventBridge-dispatched run only. Months of correct silence were
  indistinguishable from a broken pipeline; a missing Sunday mail now means
  something upstream of this step failed.
- BUDGET: a Must-Be-Won draw the model calls SKIP, on the morning
  EventBridge run only. A standing playing rule, not a verdict - these draws
  return about 80% of the stake against under half for an ordinary one, so
  a fixed stake on each keeps a ticket in play every few weeks at the least
  expected cost. The subject says it is -EV. On a Sunday it stands in for
  the status mail.

The email carries the lines themselves. This path fires on the ~9 draws a
year that are actually worth playing, and the cloud collector runs only this
script - never ev_play.py, and outputs/ is gitignored, so there is no
portfolio file for it to point at. An alert that says "run make play" is
useless in the one situation it exists for: you, away from the Mac, with a
Must-Be-Won draw closing in hours.

Lines are seeded from the draw date, so the evening run and the next-morning
retry produce the SAME portfolio instead of two different ones.

SMTP configuration via environment variables (e.g. in ~/.lotto_env,
sourced by post_draw.sh - never commit credentials):
  SMTP_SERVER   e.g. smtp.gmail.com  (SSL port 465; Gmail: use an App Password)
  SMTP_USER     your@gmail.com
  SMTP_PASS     app password
  EMAIL_TO      recipient
  EMAIL_FROM    optional, defaults to SMTP_USER
Optional: EV_ALERT_LINES (portfolio size, default 5).
"""

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lottery.ev import (  # noqa: E402
    MARGINAL,
    PLAY,
    DrawConditions,
    at_measured_uplift,  # noqa: F401 - re-exported; tests read it here
    default_portfolio_seed,
    kelly_stake,
    mbw_type,
    measured_uplift,  # noqa: F401 - re-exported; tests read it here
)
from lottery.portfolio import build_portfolio  # noqa: E402
from scripts.ev_play import (  # noqa: E402
    PRIZE_TIERS_FILE,
    advise,
)
from scripts.monitoring.notify import maybe_send_email  # noqa: E402
from scripts.monitoring.operator_page import (  # noqa: E402
    compare as compare_with_operator,
    fetch_operator_page,
)

DEFAULT_LINES = 5


def _portfolio_block(cond: DrawConditions, draw_date: date, n_lines: int) -> str:
    """The lines to buy and the command that records them, or a note saying
    why there are none - a portfolio problem must never swallow the mail."""
    try:
        # Same draw -> same lines: the retry run cannot contradict the first
        # email, and (same seed in ev_play) latest.json proposes the SAME
        # portfolio this email carries.
        portfolio = build_portfolio(n_lines, cond,
                                    seed=default_portfolio_seed(draw_date))
        lines = [p["line"] for p in portfolio]
        picks = "\n".join(
            f"  {' '.join(f'{n:2d}' for n in p['line'])}    EV £{p['ev']:+.3f}"
            for p in portfolio
        )
        cost = len(portfolio) * cond.ticket_price
        record = "; ".join(" ".join(str(n) for n in line) for line in lines)
        return (
            f"\nLines to play ({len(portfolio)} x £{cond.ticket_price:.0f} = "
            f"£{cost:.2f}):\n{picks}\n\n"
            f"After buying, record them:\n"
            f'  python scripts/roi_ledger.py add --draw-date {draw_date} '
            f'--lines "{record}"\n'
        )
    except Exception as exc:
        return (
            f"\n(Could not build a portfolio here: {exc}. Run `make play` "
            f"for the lines.)\n"
        )


def build_alert(cond: DrawConditions, verdict: dict, draw_date: date,
                n_lines: int = DEFAULT_LINES, operator: dict | None = None) -> tuple:
    """(subject, body) for a PLAY verdict, portfolio included.

    `operator` is the operator's own page (scripts/monitoring/operator_page),
    read as a second opinion on the jackpot and the Must-Be-Won flag. None
    means it was not consulted; {} means it could not be read.
    """
    # Whether the edge survives the sales estimate belongs in the SUBJECT, not
    # twelve lines down the body. This alert arrives on a phone, hours before
    # sales close, and the difference between "+EV whatever the draw sells" and
    # "+EV only if sales land near the central estimate" is the difference
    # between a decision and a coin toss. Draw 3196 was the second kind.
    sens = verdict.get("sales_sensitivity")
    if sens and sens.get("robust"):
        strength = "PLAY (robust)"
    elif sens:
        strength = "PLAY - marginal, central estimate only"
    else:
        strength = "PLAY"
    # A PLAY that only survives one shape of the popularity model is a
    # different decision from one that survives all of them, and the
    # difference belongs where it will be read - not twelve lines down.
    stab = verdict.get("model_stability") or {}
    if stab.get("label") == "MODEL-SENSITIVE":
        strength += " / model-sensitive"
    subject = (f"LOTTO +EV ALERT: {strength}, {draw_date} draw, EV "
               f"£{verdict['ev_best_line']:+.2f} per line")

    # A second source for the two facts the verdict stands on. A disagreement
    # leads the subject rather than suppressing the mail: the PLAY email is
    # the output that has to survive a bad feed, and a human with the
    # operator's page open can settle it in a minute.
    operator_line = ""
    if operator is not None:
        agrees, operator_line = compare_with_operator(cond.roll_down, cond.jackpot,
                                                      operator)
        operator_line += "\n"
        if agrees is False:
            subject = "CHECK FEEDS - " + subject

    portfolio_block = _portfolio_block(cond, draw_date, n_lines)

    # The sales estimate is the one number that decides a roll-down verdict, and
    # it is estimated, not observed. The email is read away from the Mac with
    # hours to sales close, so it has to carry the uncertainty itself - a lone
    # "+£0.04" reads as a clean PLAY even when it sits 3% from break-even.
    if sens:
        caveat = (
            f"Across plausible sales: £{sens['ev_low']:+.2f} at "
            f"{sens['tickets_high']:,} lines ... £{sens['ev_high']:+.2f} at "
            f"{sens['tickets_low']:,} lines\n"
            f"Holds across that range: "
            f"{'YES' if sens['robust'] else 'NO - thin edge, central estimate only'}\n"
        )
    else:
        caveat = ""

    # Sized, not just flagged: where the edge lives (Match 3/2 on a roll-down)
    # decides how much a rational bankroll puts on it. Half-Kelly is the
    # discount for N being estimated. Failure here must not eat the alert.
    try:
        bankroll = float(os.environ.get("EV_ALERT_BANKROLL", "1000"))
        k = kelly_stake(cond, bankroll)
        if k["lines_full"] >= 1:
            kelly_line = (f"Kelly stake:          {k['lines_full']} lines full / "
                          f"{k['lines_half']} half-Kelly "
                          f"(£{bankroll:,.0f} bankroll)\n")
        else:
            kelly_line = (f"Kelly stake:          £{k['stake_full']:.2f} at a "
                          f"£{bankroll:,.0f} bankroll - a real edge, but every "
                          f"line is an entertainment stake, not growth\n")
    except Exception:
        kelly_line = ""

    body = (
        f"The next UK Lotto draw clears your EV threshold.\n\n"
        f"Draw:                 {draw_date}\n"
        f"Jackpot (event pool): £{cond.jackpot:,.0f}\n"
        f"Must-Be-Won:          "
        f"{f'YES ({mbw_type(cond.roll_down, cond.rollover_count)})' if cond.roll_down else 'no'}\n"
        f"Estimated lines sold: {cond.tickets_sold:,}\n"
        f"Best-line EV:         £{verdict['ev_best_line']:+.2f} "
        f"(per £{cond.ticket_price:.0f} ticket, both rounds)\n"
        f"Break-even jackpot:   £{verdict['break_even_jackpot']:,.0f}\n"
        + operator_line + caveat + kelly_line
        + portfolio_block +
        f"\nEV is an average over a lottery-sized variance: a +EV draw is a good "
        f"bet, not a likely win.\n"
    )
    return subject, body


def build_marginal_alert(cond: DrawConditions, verdict: dict, alt: dict,
                         draw_date: date, n_lines: int = DEFAULT_LINES,
                         operator: dict | None = None) -> tuple:
    """(subject, body) for a draw that is +EV only at the measured uplift.

    The lines and figures are the PLAY mail's, priced at the measured sales;
    the paragraph in front says, before anything else, that the installed
    model disagrees and why that is a judgement call rather than a signal.
    """
    subject, body = build_alert(alt["cond"], alt["verdict"], draw_date, n_lines,
                                operator=operator)
    # Rebuilt rather than relabelled: "PLAY (robust)" describes the sales band
    # around the MEASURED figure, and next to "MARGINAL" it reads as a
    # contradiction. Only a feed disagreement keeps its prefix.
    prefix = "CHECK FEEDS - " if subject.startswith("CHECK FEEDS - ") else ""
    subject = (f"{prefix}LOTTO MARGINAL: {draw_date} draw, EV "
               f"£{alt['verdict']['ev_best_line']:+.2f} per line at measured sales "
               f"(installed model: SKIP £{verdict['ev_best_line']:+.2f})")
    preface = (
        f"MARGINAL - the installed model says SKIP, the measured data says PLAY.\n\n"
        f"Installed model:  EV £{verdict['ev_best_line']:+.2f} a line at "
        f"{cond.tickets_sold:,} lines sold (Must-Be-Won uplift "
        f"x{alt['installed']:.2f}, one-round era)\n"
        f"Measured uplift:  EV £{alt['verdict']['ev_best_line']:+.2f} a line at "
        f"{alt['cond'].tickets_sold:,} lines sold (x{alt['uplift']:.3f}, the "
        f"highest of {alt['n']} two-round Must-Be-Won draws on exact sales)\n\n"
        f"Every two-round Must-Be-Won so far sold less than the installed uplift "
        f"assumes, and draw 3205 was +£0.06 a line after the fact while the model "
        f"said SKIP. But {alt['n']} observations are not enough to install the "
        f"lower figure. Treat this as a thin edge you may take, not a verdict.\n\n"
        + "-" * 60 + "\n"
    )
    return subject, preface + body


def budget_due(event_name: str | None, now: datetime | None = None) -> bool:
    """Whether this run sends the Must-Be-Won budget mail.

    Only the morning retry dispatched by EventBridge (Thu/Sun 06:05 UTC). It
    prices the next draw three to four days ahead, and it is the one run of the
    four per draw that both has the last draw collected and fires on time -
    every run sending would mean four copies of the same lines.
    """
    now = now or datetime.now(timezone.utc)
    return event_name == "workflow_dispatch" and now.hour < HEARTBEAT_BEFORE_HOUR


def build_budget_mail(cond: DrawConditions, verdict: dict, draw_date: date,
                      n_lines: int = DEFAULT_LINES,
                      measured: dict | None = None) -> tuple:
    """(subject, body) for a Must-Be-Won draw the model calls SKIP.

    A playing policy, not a verdict: Must-Be-Won draws are the cheapest way to
    buy a lottery ticket. In the two-round era, at realised sales, they
    returned -£0.38, -£0.35, -£0.35 and +£0.06 a £2 line (3184, 3190, 3196,
    3205) against -£0.96 to -£1.15 for every ordinary draw - so a fixed stake
    on every one of them keeps a ticket in play every three to four weeks at
    well under half the expected loss of an ordinary draw. The subject
    says it is still -EV, because it is.
    """
    ev = verdict["ev_best_line"]
    price = cond.ticket_price
    back = (price + ev) / price
    cost = n_lines * price
    subject = (f"LOTTO budget: {draw_date} Must-Be-Won, {n_lines} lines £{cost:.0f} "
               f"- still -EV (£{ev:+.2f} a line)")
    measured_line = ""
    if measured is not None:
        measured_line = (f"At measured sales:    £{measured['verdict']['ev_best_line']:+.2f} "
                         f"a line (x{measured['uplift']:.3f} uplift, "
                         f"n={measured['n']})\n")
    body = (
        f"BUDGET PLAY - your standing rule: a fixed £{cost:.0f} on every "
        f"Must-Be-Won draw.\n"
        f"The model says SKIP: this draw is still slightly -EV. It is the "
        f"cheapest ticket\nof the cycle, not a good bet.\n\n"
        f"Draw:                 {draw_date}\n"
        f"Jackpot (event pool): £{cond.jackpot:,.0f}\n"
        f"Must-Be-Won:          YES ({mbw_type(cond.roll_down, cond.rollover_count)})\n"
        f"Estimated lines sold: {cond.tickets_sold:,}\n"
        f"Best-line EV:         £{ev:+.2f} a £{price:.0f} line - about "
        f"{back:.0%} of the stake comes back on average\n"
        + measured_line +
        f"Ordinary draw:        about -£1.00 to -£1.15 a line, under half back\n"
        f"Break-even jackpot:   £{verdict['break_even_jackpot']:,.0f}\n"
        + _portfolio_block(cond, draw_date, n_lines) +
        f"\nIf nobody matches six, the jackpot rolls down into the lower tiers, "
        f"which is\nwhy this draw pays Match 3-5 far better than an ordinary "
        f"one. The jackpot\nitself is still 1 in 22.5 million a line.\n"
    )
    return subject, body


HEARTBEAT_WEEKDAY = 6        # Sunday (Mon=0)
HEARTBEAT_BEFORE_HOUR = 10   # UTC; EventBridge dispatches the retry at 06:05


def heartbeat_due(event_name: str | None, now: datetime | None = None) -> bool:
    """Whether this run sends the weekly status mail.

    Only the Sunday-morning retry dispatched by EventBridge (a
    `workflow_dispatch`, on time to the minute): GitHub's own cron fires the
    same retry around 11:00 UTC, and two mails a week would teach you to
    ignore both. If EventBridge stops dispatching, the mail stops - which is
    the alarm this exists to be.
    """
    now = now or datetime.now(timezone.utc)
    return (event_name == "workflow_dispatch"
            and now.weekday() == HEARTBEAT_WEEKDAY
            and now.hour < HEARTBEAT_BEFORE_HOUR)


def build_heartbeat(cond: DrawConditions, verdict: dict, draw_date: date,
                    latest_collected: date | None, missing: date | None,
                    outlook: dict | None, outlook_measured: dict | None) -> tuple:
    """(subject, body) for the weekly status mail."""
    status = ("DATA BEHIND" if missing else "OK")
    subject = (f"LOTTO weekly: {status}, {draw_date} draw SKIP "
               f"(EV £{verdict['ev_best_line']:+.2f})")
    lines = [
        "Weekly status - the pipeline ran and priced the next draw.",
        "No PLAY this week. If this mail stops arriving on Sundays, the",
        "collector did not run: check GitHub Actions (collect.yml).",
        "",
        f"Latest draw collected: {latest_collected or 'unknown'}",
    ]
    if missing:
        lines.append(f"NOT COLLECTED:        the {missing} draw - figures below "
                     f"are one draw stale")
    lines += [
        "",
        f"Next draw:            {draw_date}",
        f"Jackpot (event pool): £{cond.jackpot:,.0f}",
        f"Rollovers:            {cond.rollover_count}",
        f"Best-line EV:         £{verdict['ev_best_line']:+.2f} a £2 line",
        f"Break-even jackpot:   £{verdict['break_even_jackpot']:,.0f}",
    ]
    if outlook:
        lines += [
            "",
            f"Next Must-Be-Won:     ~{outlook['expected_date']} "
            f"({outlook['draws_away']} draws away, if nobody wins first)",
            f"  projected pool:     £{outlook['projected_pool']:,.0f} vs "
            f"break-even £{outlook['break_even_jackpot']:,.0f}",
            f"  EV (installed):     £{outlook['ev_best_line']:+.2f} a line",
        ]
        if outlook_measured:
            lines.append(
                f"  EV (measured x{outlook_measured['uplift']:.3f}): "
                f"£{outlook_measured['verdict']['ev_best_line']:+.2f} a line")
    lines += ["", "Expect a PLAY or MARGINAL mail one to two draws a year, and a",
              "budget mail with lines before every Must-Be-Won draw."]
    return subject, "\n".join(lines) + "\n"


def _latest_collected() -> date | None:
    if not PRIZE_TIERS_FILE.exists():
        return None
    tiers = pd.read_csv(PRIZE_TIERS_FILE)
    return pd.to_datetime(tiers["draw_date"]).max().date() if len(tiers) else None


def main() -> None:
    if os.environ.get("EV_ALERT_TEST") == "1":
        maybe_send_email(
            "LOTTO: test alertu",
            "Dziala! Alerty mailowe sa skonfigurowane poprawnie.\n"
            "Prawdziwy mail przyjdzie przy werdykcie PLAY albo MARGINAL, "
            "a status co niedziele rano.",
        )
        print("[ev-alert] TEST email attempted (sent only if SMTP env is configured)")
        return

    # One record for the whole decision - the same advise() that `make play`
    # and the terminal print, so the mail cannot classify the draw its own way.
    a = advise()
    cond, verdict, draw_date = a.cond, a.verdict, a.cond.draw_date
    n_lines = int(os.environ.get("EV_ALERT_LINES", DEFAULT_LINES))
    for note in a.notes:
        print(f"[ev-alert] {note}")
    if a.advice == PLAY:
        subject, body = build_alert(cond, verdict, draw_date, n_lines,
                                    operator=fetch_operator_page())
        print(body)
        maybe_send_email(subject, body)
        print(f"[ev-alert] PLAY (EV £{verdict['ev_best_line']:+.2f}) - alert attempted "
              "(sent only if SMTP env is configured)")
        return
    if a.advice == MARGINAL:
        subject, body = build_marginal_alert(cond, verdict, a.measured, draw_date,
                                             n_lines, operator=fetch_operator_page())
        print(body)
        maybe_send_email(subject, body)
        print(f"[ev-alert] MARGINAL (EV £{verdict['ev_best_line']:+.2f} installed, "
              f"£{a.measured['verdict']['ev_best_line']:+.2f} measured) - alert attempted")
        return

    event = os.environ.get("GITHUB_EVENT_NAME")
    # A stale counter cannot say whether the next draw is Must-Be-Won.
    if cond.roll_down and not a.stale and budget_due(event):
        subject, body = build_budget_mail(cond, verdict, draw_date, n_lines,
                                          measured=a.measured)
        print(body)
        maybe_send_email(subject, body)
        # Stands in for the Sunday status mail: it proves the run as well.
        print(f"[ev-alert] BUDGET (EV £{verdict['ev_best_line']:+.2f}, "
              f"Must-Be-Won) - mail attempted")
        return

    print(f"[ev-alert] SKIP (EV £{verdict['ev_best_line']:+.2f}) - no alert sent")
    if not heartbeat_due(event):
        return
    # A stale counter makes the Must-Be-Won forecast fiction; the subject
    # already says DATA BEHIND.
    outlook = None if a.stale else a.outlook
    outlook_measured = None if a.stale else a.outlook_measured
    subject, body = build_heartbeat(cond, verdict, draw_date, _latest_collected(),
                                    a.stale, outlook, outlook_measured)
    print(body)
    maybe_send_email(subject, body)
    print("[ev-alert] weekly heartbeat attempted")


if __name__ == "__main__":
    main()
