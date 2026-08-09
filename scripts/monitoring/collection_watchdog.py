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
        # A truncated/empty file must alert like missing data, not crash
        # before the email goes out.
        try:
            tiers = pd.read_csv(PRIZE_TIERS_FILE)
            latest = max(
                datetime.strptime(d, "%Y-%m-%d").date() for d in tiers["draw_date"]
            )
        except Exception as exc:
            problem = f"could not read {PRIZE_TIERS_FILE}: {exc}"
        else:
            if latest >= expected:
                print(f"[watchdog] OK - draw {expected} present (latest: {latest}, "
                      f"{len(tiers)} tier rows total)")
                return 0
            problem = f"latest collected draw is {latest}, expected {expected}"

    msg = (
        f"Lotto data collection FAILED: {problem}.\n\n"
        f"Since 2026-08 a rerun self-heals gaps: the JSON API serves every "
        f"draw of the last ~180 days by number, and the collector backfills "
        f"anything missing since the last collected draw. So:\n"
        f"  1. PYTHONPATH=. python -c "
        f"'from scripts.fetch_data import download_fresh_data; download_fresh_data()'\n"
        f"  2. Still missing (API dead / gap older than 180 days)? The archive "
        f"scraper recovers winner counts:\n"
        f"     PYTHONPATH=. python scripts/backfill_prize_tiers.py\n"
        f"     (writes data/prize_tiers_history.csv - note its prize amounts "
        f"carry roll-down boosts, so trust the winner counts, not the prizes)\n"
        f"An alert firing at all now means BOTH the draw-night run and this "
        f"morning's self-heal failed - look at the workflow logs, not just the data.\n"
    )
    print(f"[watchdog] ALERT - {problem}")
    # Not emailed. Exiting non-zero reds the workflow run, and GitHub already
    # notifies on a failed run - a second message about the same failure only
    # trains you to ignore both.
    return 1


if __name__ == "__main__":
    sys.exit(main())
