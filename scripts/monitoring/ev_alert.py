#!/usr/bin/env python3
"""Email alert when the next draw clears the EV threshold (PLAY verdict).

Runs at the end of the post-draw routine. Sends nothing on SKIP days,
so an email in your inbox means: rare conditions, look at the dashboard.

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
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lottery.ev import DrawConditions, should_play, upcoming_draw_date  # noqa: E402
from lottery.portfolio import build_portfolio  # noqa: E402
from scripts.ev_play import next_draw_conditions  # noqa: E402
from scripts.monitoring.nightly_backtest import maybe_send_email  # noqa: E402

DEFAULT_LINES = 5


def build_alert(cond: DrawConditions, verdict: dict, draw_date: date,
                n_lines: int = DEFAULT_LINES) -> tuple:
    """(subject, body) for a PLAY verdict, portfolio included."""
    subject = (f"LOTTO +EV ALERT: {draw_date} draw, EV "
               f"£{verdict['ev_best_line']:+.2f} per line")

    lines = []
    try:
        # Same draw -> same lines, so a retry run cannot contradict the first
        # email with a different portfolio.
        portfolio = build_portfolio(n_lines, cond,
                                    seed=int(draw_date.strftime("%Y%m%d")))
        lines = [p["line"] for p in portfolio]
        picks = "\n".join(
            f"  {' '.join(f'{n:2d}' for n in p['line'])}    EV £{p['ev']:+.3f}"
            for p in portfolio
        )
        cost = len(portfolio) * cond.ticket_price
        record = "; ".join(" ".join(str(n) for n in line) for line in lines)
        portfolio_block = (
            f"\nLines to play ({len(portfolio)} x £{cond.ticket_price:.0f} = "
            f"£{cost:.2f}):\n{picks}\n\n"
            f"After buying, record them:\n"
            f'  python scripts/roi_ledger.py add --draw-date {draw_date} '
            f'--lines "{record}"\n'
        )
    except Exception as exc:  # never let a portfolio problem swallow the alert
        portfolio_block = (
            f"\n(Could not build a portfolio here: {exc}. Run `make play` "
            f"for the lines.)\n"
        )

    body = (
        f"The next UK Lotto draw clears your EV threshold.\n\n"
        f"Draw:                 {draw_date}\n"
        f"Jackpot (event pool): £{cond.jackpot:,.0f}\n"
        f"Must-Be-Won:          {'YES' if cond.roll_down else 'no'}\n"
        f"Estimated lines sold: {cond.tickets_sold:,}\n"
        f"Best-line EV:         £{verdict['ev_best_line']:+.2f} "
        f"(per £{cond.ticket_price:.0f} ticket, both rounds)\n"
        f"Break-even jackpot:   £{verdict['break_even_jackpot']:,.0f}\n"
        + portfolio_block +
        f"\nEV is an average over a lottery-sized variance: a +EV draw is a good "
        f"bet, not a likely win.\n"
    )
    return subject, body


def main() -> None:
    if os.environ.get("EV_ALERT_TEST") == "1":
        maybe_send_email(
            "LOTTO: test alertu",
            "Dziala! Alerty mailowe sa skonfigurowane poprawnie.\n"
            "Prawdziwy mail przyjdzie tylko przy werdykcie PLAY (+EV) "
            "albo przy awarii zbierania danych.",
        )
        print("[ev-alert] TEST email attempted (sent only if SMTP env is configured)")
        return

    cond = next_draw_conditions()
    verdict = should_play(cond, threshold=0.0)
    if not verdict["play"]:
        print(f"[ev-alert] SKIP (EV £{verdict['ev_best_line']:+.2f}) - no alert sent")
        return

    n_lines = int(os.environ.get("EV_ALERT_LINES", DEFAULT_LINES))
    subject, body = build_alert(cond, verdict, upcoming_draw_date(), n_lines)
    print(body)
    maybe_send_email(subject, body)
    print(f"[ev-alert] PLAY (EV £{verdict['ev_best_line']:+.2f}) - alert attempted "
          "(sent only if SMTP env is configured)")


if __name__ == "__main__":
    main()
