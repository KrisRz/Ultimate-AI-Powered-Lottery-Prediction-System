"""The whole draw archive: the frozen backfill plus every draw collected since.

`data/prize_tiers_history.csv` and `data/sales_history.csv` were built once, by
backfills that stopped at draw 3195 (2026-08-05). The collector has written
every draw after that - prize tiers to `data/prize_tiers.csv`, pools to
`data/draw_pools.csv` - but the analyses kept reading only the backfills, so
the roll-down replay, the uplift calibration and the wheel backtest stopped
learning from new draws in August, three Must-Be-Wons ago.

These loaders return the backfill unchanged for the draws it holds and append
the collected draws after it, translated into the backfill's own shape:

- tiers: `category` from the tier code; a per-winner prize from prize_total
  where the collector did not record one; a jackpot row with no winner in a draw
  nobody won carries the draw's pool as its prize, which is how the backfill
  records it (checked on the six draws both files hold, 3190-3195). A tier
  nobody won is recorded inconsistently by the backfill itself - sometimes 0,
  sometimes the nominal prize - so only rows with winners are comparable.
- sales: lines sold from the pool identity, (pool - previous pool) / 8.88%,
  which agrees with the backfill to within a few lines on 3180-3195.

⛔ Deliberately NOT used by `calibrate_popularity.load_joined`. The frozen
popularity-v2 experiment depends on that loader never returning a draw after
3195 (tests/test_popularity_v2_frozen.py); the popularity weights move only
through their own gated process.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lottery.ev import exact_lines_sold

DATA_DIR = Path("data")
TIERS_HISTORY_FILE = DATA_DIR / "prize_tiers_history.csv"
COLLECTED_TIERS_FILE = DATA_DIR / "prize_tiers.csv"
POOLS_FILE = DATA_DIR / "draw_pools.csv"
SALES_HISTORY_FILE = DATA_DIR / "sales_history.csv"

CATEGORY = {1: "Match 6", 2: "Match 5 plus Bonus", 3: "Match 5",
            4: "Match 4", 5: "Match 3", 6: "Match 2"}
LINE_PRICE = 2.0     # every collected draw is two-round, £2 a line


def _pools(pools_file: Path) -> pd.DataFrame | None:
    return pd.read_csv(pools_file) if pools_file.exists() else None


def load_tier_archive(history_file: Path = TIERS_HISTORY_FILE,
                      collected_file: Path = COLLECTED_TIERS_FILE,
                      pools_file: Path = POOLS_FILE) -> pd.DataFrame:
    """Every (draw, round, tier) row, in prize_tiers_history.csv's columns."""
    history = pd.read_csv(history_file)
    if not collected_file.exists():
        return history
    collected = pd.read_csv(collected_file)
    new = collected[collected["draw_number"] > history["draw_number"].max()].copy()
    if new.empty:
        return history

    new["category"] = new["tier"].map(CATEGORY)
    # The collector filled prize_per_winner only from mid-July; before that
    # the row carries prize_total alone.
    missing = new["prize_per_winner"].isna() & (new["winners"] > 0)
    new.loc[missing, "prize_per_winner"] = (
        new.loc[missing, "prize_total"] / new.loc[missing, "winners"]).round(2)
    pools = _pools(pools_file)
    if pools is not None:
        pool_by_draw = dict(zip(pools["draw_number"].astype(int), pools["pool_gbp"]))
        jackpot = new["tier"] == 1
        won = (new[jackpot].groupby("draw_number")["winners"].sum() > 0)
        unwon = new["draw_number"].map(won).eq(False)
        fill = jackpot & unwon
        new.loc[fill, "prize_per_winner"] = (
            new.loc[fill, "draw_number"].map(pool_by_draw).round())
    return pd.concat([history, new[history.columns]], ignore_index=True)


def load_sales_archive(history_file: Path = SALES_HISTORY_FILE,
                       pools_file: Path = POOLS_FILE) -> pd.DataFrame:
    """Lines sold per draw, in sales_history.csv's columns."""
    history = pd.read_csv(history_file)
    pools = _pools(pools_file)
    if pools is None:
        return history
    last = int(history["draw_number"].max())
    dates = dict(zip(pools["draw_number"].astype(int), pools["draw_date"]))
    rows = [
        {"draw_date": dates[draw], "draw_number": draw,
         "sales_gbp": int(lines * LINE_PRICE), "lines_sold": int(lines),
         "pct_chg": float("nan")}
        for draw, lines in sorted(exact_lines_sold(pools).items())
        if draw > last and draw in dates
    ]
    if not rows:
        return history
    return pd.concat([history, pd.DataFrame(rows)[history.columns]],
                     ignore_index=True)
