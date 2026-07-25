#!/usr/bin/env python3
"""Backfill historical per-tier winner counts from lottery.co.uk.

The official National-Lottery XML feed only carries the *latest* draw's prize
breakdown, so data/prize_tiers.csv (the live calibration fuel) starts in 2026.
This script recovers the same per-tier winner counts for every past draw from
lottery.co.uk's server-rendered result pages, whose numbers match the official
feed exactly (verified against draw 3191).

Output: data/prize_tiers_history.csv - one row per (draw, round, tier) with the
number of winners and the prize per winner. Kept SEPARATE from the live
prize_tiers.csv so the two provenances never mix; the popularity-calibration
step reads both. Draw numbers/dates come from data/lotto_full_history.csv, so we
only ever fetch pages we know exist.

Pages are cached on disk (default: scratchpad) so re-runs are free, and the
script is resumable - draws already in the output CSV are skipped. Winner counts
are never fabricated: a page that fails to parse is logged and skipped (or aborts
the run under --strict).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data")
FULL_HISTORY_FILE = DATA_DIR / "lotto_full_history.csv"
OUTPUT_FILE = DATA_DIR / "prize_tiers_history.csv"

# First draw of the current 59-ball game. Its prize structure / player
# popularity is what the EV model calibrates against, so this is the default
# lower bound; pass --since-draw 1 to also pull the 49-ball era.
ERA_59_START_DRAW = 2066

BASE_URL = "https://www.lottery.co.uk/lotto/results-{d:02d}-{m:02d}-{y:04d}"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)

# lottery.co.uk category label -> our tier number (matches prize_tiers.csv)
CATEGORY_TIER = {
    "Match 6": 1,
    "Match 5 plus Bonus": 2,
    "Match 5": 3,
    "Match 4": 4,
    "Match 3": 5,
    "Match 2": 6,
}

OUTPUT_COLUMNS = [
    "draw_number", "draw_date", "round", "tier", "category",
    "winners", "prize_per_winner",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _to_int(text: str) -> int:
    """Digits only: strips £, commas, and words like 'Rollover'/'Rolldown'."""
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else 0


def _to_money(text: str) -> float:
    cleaned = re.sub(r"[^\d.]", "", (text or "").replace(",", ""))
    return float(cleaned) if cleaned else 0.0


def parse_breakdown(html: str, draw_number: int, draw_date: str) -> list[dict]:
    """Extract per-(round, tier) winner rows from a result page.

    Handles both layouts: single-round (data-title 'Winners') for pre-2026 draws
    and two-round ('Round 1 Winners' / 'Round 2 Winners') since 2026-06. Raises
    ValueError if the page has no recognizable breakdown table.

    Parsed with stdlib regex (no bs4/lxml dependency): the markup is
    machine-generated and uniform - each cell carries a data-title="..." and the
    tier label sits in a <strong>. We first slice out the breakdown table so no
    other data-title table on the page can leak in.
    """
    apos = html.find('id="prizeBreakdown"')
    region = html[apos:] if apos != -1 else html
    tstart = region.find("<table")
    tend = region.find("</table>", tstart)
    if tstart == -1 or tend == -1:
        raise ValueError("no prize-breakdown table on page")
    table = region[tstart:tend]

    rows: list[dict] = []
    for tr in re.split(r"<tr\b", table)[1:]:
        cat_m = re.search(r"<strong>\s*([^<]+?)\s*</strong>", tr)
        if cat_m is None:
            continue
        category = cat_m.group(1).strip()
        tier = CATEGORY_TIER.get(category)
        if tier is None:
            continue  # totals row, free lucky dip, etc.

        cells = {
            title.strip(): re.sub(r"<[^>]+>", " ", val)
            for title, val in re.findall(
                r'data-title="([^"]+)"[^>]*>(.*?)</td>', tr, re.DOTALL
            )
        }
        prize = _to_money(cells.get("Prize Per Winner") or cells.get("Prize") or "")

        if "Round 1 Winners" in cells:  # two-round era
            per_round = {1: cells.get("Round 1 Winners"), 2: cells.get("Round 2 Winners")}
        elif "Winners" in cells:  # single-round era
            per_round = {1: cells.get("Winners")}
        else:
            raise ValueError(f"row for tier {tier} has no winners column: {cells}")

        for rnd, wtext in per_round.items():
            rows.append({
                "draw_number": draw_number,
                "draw_date": draw_date,
                "round": rnd,
                "tier": tier,
                "category": category,
                "winners": _to_int(wtext),
                "prize_per_winner": prize,
            })

    if not rows:
        raise ValueError("breakdown table parsed to zero tier rows")

    # Sanity guard: Match 3 always has thousands of winners; a tiny number means
    # we parsed the wrong table or the page is a placeholder.
    m3 = sum(r["winners"] for r in rows if r["tier"] == 5)
    if m3 < 100:
        raise ValueError(f"implausible Match-3 total ({m3}) - likely a bad parse")
    return rows


def fetch_page(date_iso: str, cache_dir: Path, session: requests.Session,
               delay: float, use_cache: bool) -> str:
    y, m, d = (int(x) for x in date_iso.split("-"))
    cache_file = cache_dir / f"{date_iso}.html"
    if use_cache and cache_file.exists() and cache_file.stat().st_size > 1000:
        return cache_file.read_text(encoding="utf-8", errors="replace")

    url = BASE_URL.format(d=d, m=m, y=y)
    time.sleep(delay)  # be polite - one page per `delay` seconds
    resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=25)
    resp.raise_for_status()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text, encoding="utf-8")
    return resp.text


def draws_to_fetch(since_draw: int) -> list[tuple[int, str]]:
    """(draw_number, date_iso) for every draw >= since_draw, one per event."""
    hist = pd.read_csv(FULL_HISTORY_FILE)
    events = (
        hist[hist["DrawNumber"] >= since_draw]
        .drop_duplicates(subset="DrawNumber")
        .sort_values("DrawNumber")
    )
    return [(int(r.DrawNumber), str(r._2)) for r in events[["DrawNumber", "Draw Date"]].itertuples()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since-draw", type=int, default=ERA_59_START_DRAW,
                        help=f"lowest draw number to fetch (default {ERA_59_START_DRAW}, 59-ball era)")
    parser.add_argument("--limit", type=int, default=None, help="stop after N draws (for testing)")
    parser.add_argument("--delay", type=float, default=1.2, help="seconds between live requests")
    parser.add_argument("--cache-dir", default=None,
                        help="HTML cache dir (default: scratchpad or ./.tier_cache)")
    parser.add_argument("--no-cache", action="store_true", help="always refetch, ignore cache")
    parser.add_argument("--strict", action="store_true", help="abort on first parse failure")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else DATA_DIR.parent / ".tier_cache"

    targets = draws_to_fetch(args.since_draw)
    done: set[int] = set()
    if OUTPUT_FILE.exists():
        done = set(pd.read_csv(OUTPUT_FILE)["draw_number"].astype(int))
        logger.info("Resuming: %d draws already in %s", len(done), OUTPUT_FILE)
    todo = [(n, d) for n, d in targets if n not in done]
    if args.limit:
        todo = todo[: args.limit]
    logger.info("Fetching %d draws (%d total in era, %d already done)",
                len(todo), len(targets), len(done))

    session = requests.Session()
    all_rows: list[dict] = []
    failures: list[tuple[int, str, str]] = []
    for i, (draw_no, date_iso) in enumerate(todo, 1):
        try:
            html = fetch_page(date_iso, cache_dir, session, args.delay, not args.no_cache)
            rows = parse_breakdown(html, draw_no, date_iso)
            all_rows.extend(rows)
        except Exception as e:  # noqa: BLE001 - report and continue
            failures.append((draw_no, date_iso, str(e)))
            logger.error("Draw %s (%s) FAILED: %s", draw_no, date_iso, e)
            if args.strict:
                break
        if i % 50 == 0:
            logger.info("... %d/%d draws processed (%d rows so far)", i, len(todo), len(all_rows))

    if all_rows:
        new_df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
        if OUTPUT_FILE.exists():
            old = pd.read_csv(OUTPUT_FILE)
            new_df = pd.concat([old, new_df], ignore_index=True)
        new_df = new_df.drop_duplicates(subset=["draw_number", "round", "tier"], keep="last")
        new_df = new_df.sort_values(["draw_number", "round", "tier"])
        new_df.to_csv(OUTPUT_FILE, index=False)
        logger.info("Wrote %s (%d rows, %d draws)",
                    OUTPUT_FILE, len(new_df), new_df["draw_number"].nunique())

    if failures:
        logger.warning("%d draws failed to parse:", len(failures))
        for n, d, msg in failures[:10]:
            logger.warning("  draw %s (%s): %s", n, d, msg)
    return 1 if (failures and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
