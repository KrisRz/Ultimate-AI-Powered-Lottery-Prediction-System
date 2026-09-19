#!/usr/bin/env python3
"""The few things that must hold before the advisor is allowed to price a draw.

There are two different jobs that both look like "check the data", and
collapsing them caused the failure fixed in PR #40: a freshness check sat in
front of the PLAY email and a red gate skipped the one output that spends
money.

The fix then was to move every check behind the alert. That was right for
what it fixed and wrong as a principle - it means EV can price a draw whose
inputs were never validated at all. The two jobs are:

  BEFORE EV (this file)   things that make TODAY'S verdict wrong. Missing
                          pool, an unparseable jackpot, a draw number the
                          rules do not cover, sources that disagree about
                          which draw is next. Cheap, local, and fatal: a
                          confidently wrong PLAY is worse than no PLAY.

  AFTER EV (elsewhere)    things that mean the ARCHIVE is degrading.
                          Freshness, pool backfill, the test suite,
                          fairness and stats. Real problems, none of which
                          makes tonight's number wrong, so none of which
                          should be able to suppress tonight's alert.

The test for whether a check belongs here: if it fails, is the EV figure
the advisor is about to compute incorrect? Stale history is not - the
advisor prices the NEXT draw from the current pool. A pool of NaN is.

Run:  PYTHONPATH=. python scripts/monitoring/pre_ev_gate.py
Exit: 0 to proceed, 1 to stop before pricing.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from lottery.ev import (
    MINIMUM_JACKPOT,
    N_BALLS,
    N_PICK,
    TWO_ROUND_FIRST_DRAW,
    jackpot_share_of_sales,
    upcoming_draw_date,
)

POOLS_FILE = Path("data/draw_pools.csv")
TIERS_FILE = Path("data/prize_tiers.csv")

# A pool far beyond anything the game has produced is a parsing accident,
# not a jackpot. The 59-ball era record is GBP 66m (draw 2069, Jan 2016);
# an order of magnitude above it means the feed changed units.
IMPLAUSIBLE_POOL = 200_000_000.0


def _fail(problems: list, msg: str) -> None:
    problems.append(msg)


def check_pool_file(problems: list) -> pd.DataFrame | None:
    if not POOLS_FILE.exists():
        _fail(problems, f"{POOLS_FILE} missing - nothing to price the draw from")
        return None
    pools = pd.read_csv(POOLS_FILE)
    if pools.empty:
        _fail(problems, f"{POOLS_FILE} is empty")
        return None
    for col in ("draw_number", "draw_date", "pool_gbp"):
        if col not in pools.columns:
            _fail(problems, f"{POOLS_FILE}: missing column {col!r}")
            return None

    latest = pools.sort_values("draw_number").iloc[-1]
    pool = latest["pool_gbp"]
    if pool is None or (isinstance(pool, float) and not math.isfinite(pool)):
        _fail(problems, f"latest pool is not a finite number ({pool!r})")
    elif pool <= 0:
        _fail(problems, f"latest pool is not positive ({pool!r})")
    elif pool > IMPLAUSIBLE_POOL:
        _fail(problems, f"latest pool GBP {pool:,.0f} exceeds anything this "
                        "game has paid - suspect a units change in the feed")
    elif pool < MINIMUM_JACKPOT * 0.5:
        # The operator guarantees a GBP 2m minimum; well under it means the
        # number is not the event pool this model expects.
        _fail(problems, f"latest pool GBP {pool:,.0f} is far below the "
                        f"GBP {MINIMUM_JACKPOT:,.0f} guaranteed minimum")
    return pools


def check_draw_identity(pools: pd.DataFrame, problems: list) -> None:
    """The advisor prices the NEXT draw; the inputs must agree which it is."""
    latest = pools.sort_values("draw_number").iloc[-1]
    try:
        last_date = pd.to_datetime(latest["draw_date"]).date()
    except Exception as exc:
        _fail(problems, f"latest pool row has an unparseable date: {exc}")
        return

    nxt = upcoming_draw_date()
    if not isinstance(nxt, date):
        _fail(problems, "upcoming_draw_date() did not return a date")
        return
    if last_date > nxt:
        _fail(problems, f"latest collected draw ({last_date}) is later than "
                        f"the draw being priced ({nxt}) - the advisor would "
                        "price a draw that has already happened")

    draw_no = int(latest["draw_number"])
    if draw_no <= 0:
        _fail(problems, f"latest draw number is {draw_no}")
    # Versioned rules must resolve for the draw about to be priced. A number
    # outside their coverage means the EV would silently use the wrong era's
    # jackpot share.
    share = jackpot_share_of_sales(draw_no + 1)
    if not (0.0 < share < 1.0):
        _fail(problems, f"jackpot share for draw {draw_no + 1} is {share!r}")
    if draw_no + 1 >= TWO_ROUND_FIRST_DRAW and share >= 0.098:
        _fail(problems, "two-round draw resolving to the pre-June-2026 "
                        "jackpot share - versioned rules disagree with the "
                        "draw number")


def check_tier_inputs(problems: list) -> None:
    """Fixed prizes are re-derived per run; without them EV has no lower tiers."""
    if not TIERS_FILE.exists():
        _fail(problems, f"{TIERS_FILE} missing - fixed prizes cannot be derived")
        return
    tiers = pd.read_csv(TIERS_FILE)
    if tiers.empty:
        _fail(problems, f"{TIERS_FILE} is empty")
        return
    for col in ("draw_number", "tier", "winners"):
        if col not in tiers.columns:
            _fail(problems, f"{TIERS_FILE}: missing column {col!r}")
            return
    if tiers["winners"].isna().all():
        _fail(problems, f"{TIERS_FILE}: every winner count is null")


def check_model_constants(problems: list) -> None:
    """Guards against a bad edit reaching a live verdict.

    These are not data problems - they are the kind of thing a refactor
    breaks - but they are cheap, and a verdict computed with 49 balls would
    look entirely normal.
    """
    if N_BALLS != 59 or N_PICK != 6:
        _fail(problems, f"game shape is {N_PICK}/{N_BALLS}, expected 6/59")


def run() -> list:
    problems: list = []
    pools = check_pool_file(problems)
    if pools is not None:
        check_draw_identity(pools, problems)
    check_tier_inputs(problems)
    check_model_constants(problems)
    return problems


def main() -> int:
    problems = run()
    print("=" * 64)
    print("PRE-EV GATE")
    print("=" * 64)
    if not problems:
        print("  OK - the inputs tonight's verdict depends on are present")
        print("       and internally consistent. Archive-level checks run")
        print("       after the alert, where they cannot suppress it.")
        return 0
    for p in problems:
        print(f"  BLOCK  {p}")
    print()
    print("Refusing to price this draw. A confidently wrong PLAY costs more")
    print("than a missed one; these are inputs, not history.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
