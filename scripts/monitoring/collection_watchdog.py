#!/usr/bin/env python3
"""Watchdog: verify the latest draw's data was actually collected.

Runs (in CI) the morning after each draw. The official XML only serves the
LATEST draw, so a missed collection window (Wed->Sat, ~72h) loses that draw's
prize-tier data permanently - this check makes any failure loud while the
data is still recoverable.

Exits non-zero when data is missing (which also triggers GitHub's own
workflow-failure notification) and attempts an SMTP alert if configured.
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.monitoring.nightly_backtest import maybe_send_email  # noqa: E402

PRIZE_TIERS_FILE = Path("data/prize_tiers.csv")


def most_recent_draw_date(today: date | None = None) -> date:
    """Most recent Wednesday (2) or Saturday (5) strictly before today."""
    d = (today or date.today()) - timedelta(days=1)
    while d.weekday() not in (2, 5):
        d -= timedelta(days=1)
    return d


def main() -> int:
    expected = most_recent_draw_date()

    if not PRIZE_TIERS_FILE.exists():
        problem = f"{PRIZE_TIERS_FILE} does not exist at all"
    else:
        tiers = pd.read_csv(PRIZE_TIERS_FILE)
        latest = max(
            datetime.strptime(d, "%Y-%m-%d").date() for d in tiers["draw_date"]
        )
        if latest >= expected:
            print(f"[watchdog] OK - draw {expected} present (latest: {latest}, "
                  f"{len(tiers)} tier rows total)")
            return 0
        problem = f"latest collected draw is {latest}, expected {expected}"

    msg = (
        f"Lotto data collection FAILED: {problem}.\n\n"
        f"The official XML only serves the latest draw - fix this before the "
        f"next draw or the prize-tier data for {expected} is lost.\n"
        f"Manual recovery: PYTHONPATH=. python -c "
        f"'from scripts.fetch_data import download_fresh_data; download_fresh_data()'\n"
    )
    print(f"[watchdog] ALERT - {problem}")
    maybe_send_email("LOTTO COLLECTION FAILURE", msg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
