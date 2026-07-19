#!/usr/bin/env python3
"""Backfill full UK Lotto draw history from the Merseyworld archive.

Produces:
  data/lotto_full_history.csv   - every draw since 1994 (both the 49-ball and
                                  59-ball eras), with jackpot and jackpot-winner
                                  counts per draw
  data/merged_lottery_data.csv  - the 59-ball era only (draws since 2015-10-10),
                                  in the column layout the rest of the pipeline
                                  expects

Invalid rows are rejected loudly - numbers are never fabricated.
"""

import argparse
import logging
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data")
FULL_HISTORY_FILE = DATA_DIR / "lotto_full_history.csv"
MERGED_FILE = DATA_DIR / "merged_lottery_data.csv"

MERSEYWORLD_URL = (
    "https://lottery.merseyworld.com/cgi-bin/lottery"
    "?days=2&Machine=Z&Ballset=0&order=0&show=1&year=0&display=CSV"
)
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# First 59-ball draw was on 2015-10-10 (before that: 49 balls)
ERA_59_START = date(2015, 10, 10)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_archive(url: str = MERSEYWORLD_URL) -> str:
    logger.info("Downloading full draw history from Merseyworld...")
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    return resp.text


def parse_archive(html: str) -> pd.DataFrame:
    """Parse the CSV table embedded in Merseyworld's <PRE> block."""
    m = re.search(r"<PRE>(.*?)</PRE>", html, re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError("No <PRE> block found in archive response - format changed?")
    lines = m.group(1).splitlines()

    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("No.")), None
    )
    if header_idx is None:
        raise ValueError("Header row not found in archive response - format changed?")

    rows, bad = [], []
    for ln in lines[header_idx + 1:]:
        ln = ln.strip()
        # Data rows always start with the draw number; anything else is prose,
        # a repeated header, or an <HR> separator in the archive page
        if not ln or not ln[0].isdigit():
            continue
        parts = [p.strip() for p in ln.split(",")]
        # No., Day, DD, MMM, YYYY, N1..N6, BN, Jackpot, Wins, Machine, Set
        if len(parts) != 16:
            bad.append(ln)
            continue
        try:
            draw_no = int(parts[0])
            draw_date = datetime.strptime(f"{parts[2]} {parts[3]} {parts[4]}", "%d %b %Y").date()
            numbers = sorted(int(p) for p in parts[5:11])
            bonus = int(parts[11])
            jackpot = int(parts[12])
            jackpot_wins = int(parts[13])
            machine = parts[14]
            ball_set = parts[15]
        except (ValueError, IndexError):
            bad.append(ln)
            continue

        max_ball = 59 if draw_date >= ERA_59_START else 49
        if (
            len(set(numbers)) != 6
            or not all(1 <= n <= max_ball for n in numbers)
            or not 1 <= bonus <= max_ball
            or bonus in numbers
        ):
            bad.append(ln)
            continue

        rows.append(
            {
                "_parse_idx": len(rows),
                "Draw Date": draw_date.isoformat(),
                "Number_1": numbers[0], "Number_2": numbers[1], "Number_3": numbers[2],
                "Number_4": numbers[3], "Number_5": numbers[4], "Number_6": numbers[5],
                "Bonus": bonus,
                "Jackpot": jackpot,
                "JackpotWins": jackpot_wins,
                "Machine": machine,
                "Ball Set": ball_set,
                "DrawNumber": draw_no,
            }
        )

    if bad:
        for ln in bad[:5]:
            logger.error("Rejected row: %s", ln)
        raise ValueError(f"{len(bad)} invalid rows in archive - refusing to continue")

    df = pd.DataFrame(rows)
    # The archive page lists the most recent ~180 days twice (current + expired
    # sections) - drop identical repeats (ignoring page position)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "_parse_idx"])

    # Since 2026-06-10 each draw event has TWO rounds (two machines, two ball
    # sets) and a ticket is checked against both. Page order within a draw
    # number is Round 1 first, matching the official XML's set order.
    df = df.sort_values(["DrawNumber", "_parse_idx"])
    df["Round"] = df.groupby("DrawNumber").cumcount() + 1
    if (df["Round"] > 2).any():
        raise ValueError("More than 2 rows for a single draw number - unexpected format")
    return df.drop(columns="_parse_idx").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=MERSEYWORLD_URL)
    parser.add_argument(
        "--keep-merged", action="store_true",
        help="Do not rewrite data/merged_lottery_data.csv (only write the full history file)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True, parents=True)
    df = parse_archive(fetch_archive(args.url))
    logger.info(
        "Parsed %d draws: %s -> %s",
        len(df), df["Draw Date"].iloc[0], df["Draw Date"].iloc[-1],
    )

    df.to_csv(FULL_HISTORY_FILE, index=False)
    logger.info("Wrote %s (%d draws, both eras)", FULL_HISTORY_FILE, len(df))

    era59 = df[df["Draw Date"] >= ERA_59_START.isoformat()]
    logger.info("59-ball era: %d draw rows since %s", len(era59), ERA_59_START)

    if not args.keep_merged:
        # The downstream pipeline expects one row per draw date - use Round 1
        # only (Round 2, added 2026-06-10, stays in the full history file)
        era59_r1 = era59[era59["Round"] == 1]
        pipeline_cols = [
            "Draw Date", "Number_1", "Number_2", "Number_3", "Number_4",
            "Number_5", "Number_6", "Bonus", "Ball Set", "Machine", "DrawNumber",
        ]
        era59_r1[pipeline_cols].to_csv(MERGED_FILE, index=False)
        logger.info("Wrote %s (%d draws, 59-ball era, Round 1)", MERGED_FILE, len(era59_r1))

    return 0


if __name__ == "__main__":
    sys.exit(main())
