#!/usr/bin/env python3
"""Email alert when the next draw clears the EV threshold (PLAY verdict).

Runs at the end of the post-draw routine. Sends nothing on SKIP days,
so an email in your inbox means: rare conditions, look at the dashboard.

SMTP configuration via environment variables (e.g. in ~/.lotto_env,
sourced by post_draw.sh - never commit credentials):
  SMTP_SERVER   e.g. smtp.gmail.com  (SSL port 465; Gmail: use an App Password)
  SMTP_USER     your@gmail.com
  SMTP_PASS     app password
  EMAIL_TO      recipient
  EMAIL_FROM    optional, defaults to SMTP_USER
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lottery.ev import should_play  # noqa: E402
from scripts.ev_play import next_draw_conditions  # noqa: E402
from scripts.monitoring.nightly_backtest import maybe_send_email  # noqa: E402


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

    subject = f"LOTTO +EV ALERT: next draw EV £{verdict['ev_best_line']:+.2f} per line"
    body = (
        f"The next UK Lotto draw clears your EV threshold.\n\n"
        f"Jackpot (per round): £{cond.jackpot:,.0f}\n"
        f"Must-Be-Won:         {'YES' if cond.roll_down else 'no'}\n"
        f"Estimated lines sold: {cond.tickets_sold:,}\n"
        f"Best-line EV:        £{verdict['ev_best_line']:+.2f} (per £2 ticket, both rounds)\n\n"
        f"Generate the portfolio:  make play\n"
        f"Record what you buy:     python scripts/roi_ledger.py add --from-latest\n"
    )
    maybe_send_email(subject, body)
    print(f"[ev-alert] PLAY (EV £{verdict['ev_best_line']:+.2f}) - alert attempted "
          "(sent only if SMTP env is configured)")


if __name__ == "__main__":
    main()
