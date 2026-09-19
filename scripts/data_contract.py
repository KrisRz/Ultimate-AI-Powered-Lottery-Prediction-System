#!/usr/bin/env python3
"""The contract the collected data must satisfy, and the three ways it can fail.

Written after a formatting change slipped through every existing check: the
feed dropped a space at draw 3191 and "Lotto 4" became "Lotto4", splitting
one drawing machine into two entities of ~15 draws each. Nothing was
malformed, nothing was missing, and the machine-bias analysis quietly ran on
half the data it should have had.

The lesson is that "valid" and "expected" are different questions, so this
answers them separately:

    KNOWN              the value is one we have seen and understand
    UNKNOWN_BUT_VALID  well-formed, but new - ingest it, keep the raw value,
                       and say so loudly
    INVALID            structurally wrong - refuse it

That middle state is the point. If Camelot introduces a Lotto7 machine
tomorrow, refusing it would break collection over a change that is entirely
legitimate; silently accepting it would repeat 3191. So it is accepted AND
announced.

Structural rules are separate from category rules and are not forgiving. A
missing draw number, a jackpot that stops being a number, a seventh ball in
a six-ball game - these cannot be "new values", and the fields EV depends on
get the strictest treatment, because a draw the advisor cannot price is
worse than a draw it does not have.

Run:  PYTHONPATH=. python scripts/data_contract.py
      PYTHONPATH=. python scripts/data_contract.py --strict   (unknown => exit 1)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from lottery.ev import N_BALLS, N_PICK

HISTORY_FILE = Path("data/lotto_full_history.csv")
POOLS_FILE = Path("data/draw_pools.csv")

# The 59-ball era. Before it the game drew 6 from 49, so the ball-range rule
# is era-dependent and a 53 in a 2014 draw is INVALID, not merely surprising.
ERA_FIRST_DRAW = 2066
ERA_49_MAX_BALL = 49

REQUIRED_COLUMNS = [
    "Draw Date", "Number_1", "Number_2", "Number_3", "Number_4", "Number_5",
    "Number_6", "Bonus", "DrawNumber", "Round",
]

# Categories we have seen. Machine names are compared with spaces stripped,
# which is exactly the 3191 failure encoded as a rule rather than a habit.
#
# Both eras are listed because the file holds both. Writing only the current
# era's machines here would flag thirteen legitimate 1990s drums as new on
# every run, and a report that always warns is a report nobody reads - the
# same way the collector's two spellings went unnoticed for being harmless.
KNOWN_MACHINES_59 = {"Arthur", "Guinevere", "Lancelot", "Merlin",
                     "Lotto2", "Lotto4", "Lotto5", "Lotto6"}
KNOWN_MACHINES_49 = {"Amethyst", "Galahad", "Garnet", "Moonstone", "Opal",
                     "Pearl", "Sapphire", "Topaz", "Vyvyan"}
KNOWN_MACHINES = KNOWN_MACHINES_59 | KNOWN_MACHINES_49
# Numeric sets are current; the lettered ones ("A", "B") and 12/14 belong to
# the 49-ball era and never appear after draw 2066.
KNOWN_BALL_SETS = ({str(i) for i in range(1, 15)} | {"A", "B"}) - {"13"}
KNOWN_ROUNDS = {1, 2}

KNOWN = "KNOWN"
UNKNOWN_BUT_VALID = "UNKNOWN_BUT_VALID"
INVALID = "INVALID"


def normalise_machine(name) -> str:
    """"Lotto 4" and "Lotto4" are the same machine."""
    return str(name).replace(" ", "")


class Findings:
    """Collected verdicts. Invalid is fatal; unknown is loud but survivable."""

    def __init__(self) -> None:
        self.invalid: list[str] = []
        self.unknown: list[str] = []

    def fail(self, msg: str) -> None:
        self.invalid.append(msg)

    def note_unknown(self, msg: str) -> None:
        self.unknown.append(msg)

    @property
    def status(self) -> str:
        if self.invalid:
            return INVALID
        return UNKNOWN_BUT_VALID if self.unknown else KNOWN


def check_schema(df: pd.DataFrame, f: Findings) -> None:
    """Structure. None of this can be a legitimate new value."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        f.fail(f"missing required column(s): {missing}")
        return

    if not pd.api.types.is_integer_dtype(df["DrawNumber"]):
        f.fail("DrawNumber is not integer - a type change, not a new value")
    if df["DrawNumber"].isna().any():
        f.fail("DrawNumber has nulls")

    try:
        dates = pd.to_datetime(df["Draw Date"], errors="raise")
    except Exception as exc:
        f.fail(f"Draw Date does not parse: {exc}")
        return
    if dates.isna().any():
        f.fail("Draw Date has nulls")

    for col in [f"Number_{i}" for i in range(1, N_PICK + 1)] + ["Bonus"]:
        if not pd.api.types.is_integer_dtype(df[col]):
            f.fail(f"{col} is not integer")


def check_draw_rows(df: pd.DataFrame, f: Findings) -> None:
    """Per-draw invariants: ball ranges by era, uniqueness, bonus, duplicates."""
    num_cols = [f"Number_{i}" for i in range(1, N_PICK + 1)]
    nums = df[num_cols].to_numpy()

    if (nums < 1).any():
        f.fail("ball below 1")
    era = df["DrawNumber"].to_numpy() >= ERA_FIRST_DRAW
    if nums[era].max(initial=0) > N_BALLS:
        f.fail(f"ball above {N_BALLS} in the 59-ball era")
    if len(nums[~era]) and nums[~era].max(initial=0) > ERA_49_MAX_BALL:
        f.fail(f"ball above {ERA_49_MAX_BALL} before draw {ERA_FIRST_DRAW} - "
               "era boundary violated")

    dup_rows = int(sum(len(set(row)) < N_PICK for row in nums))
    if dup_rows:
        f.fail(f"{dup_rows} row(s) repeat a ball within one draw")

    bonus_clash = int(sum(int(b) in set(row)
                          for b, row in zip(df["Bonus"], nums)))
    if bonus_clash:
        f.fail(f"{bonus_clash} row(s) have the bonus among the main six")

    dups = df.duplicated(subset=["DrawNumber", "Round"]).sum()
    if dups:
        f.fail(f"{dups} duplicate (DrawNumber, Round) row(s)")

    gaps = set(range(int(df["DrawNumber"].min()),
                     int(df["DrawNumber"].max()) + 1)) - set(df["DrawNumber"])
    if gaps:
        f.fail(f"{len(gaps)} missing draw number(s), e.g. {sorted(gaps)[:5]}")

    ordered = df.sort_values("DrawNumber")
    if (pd.to_datetime(ordered["Draw Date"]).diff().dt.days < 0).any():
        f.fail("draw numbers and dates disagree on order")


def check_categories(df: pd.DataFrame, f: Findings) -> None:
    """The middle state: new values are accepted, never silently."""
    machines = {normalise_machine(m) for m in df["Machine"].dropna().unique()} \
        if "Machine" in df.columns else set()
    for m in sorted(machines - KNOWN_MACHINES):
        f.note_unknown(f"UNKNOWN MACHINE: {m!r} - ingested, add to "
                       "KNOWN_MACHINES once confirmed")

    if "Ball Set" in df.columns:
        sets_ = {str(b) for b in df["Ball Set"].dropna().unique()}
        for b in sorted(sets_ - KNOWN_BALL_SETS):
            f.note_unknown(f"UNKNOWN BALL SET: {b!r} - ingested")

    rounds = set(int(r) for r in df["Round"].dropna().unique())
    for r in sorted(rounds - KNOWN_ROUNDS):
        # A third round would change how EV sums over rounds, so it is
        # loud rather than fatal - but it must not pass unnoticed.
        f.note_unknown(f"UNKNOWN ROUND: {r} - EV sums over rounds, so this "
                       "needs a human before the next verdict")

    raw_machines = set(df["Machine"].dropna().unique()) \
        if "Machine" in df.columns else set()
    collisions = {}
    for raw in raw_machines:
        collisions.setdefault(normalise_machine(raw), set()).add(raw)
    for norm, raws in sorted(collisions.items()):
        if len(raws) > 1:
            # Exactly the 3191 event. Not an error - the data are fine - but
            # anything grouping by the raw column is being cut in half.
            f.note_unknown(f"MACHINE SPELLING DRIFT: {sorted(raws)} all mean "
                           f"{norm!r} - group on the normalised value")


def check_ev_inputs(f: Findings) -> None:
    """The pool file EV prices draws from. Stricter, because it spends money."""
    if not POOLS_FILE.exists():
        f.fail(f"{POOLS_FILE} is missing - EV cannot price a draw without it")
        return
    pools = pd.read_csv(POOLS_FILE)
    for col in ("draw_number", "draw_date", "pool_gbp"):
        if col not in pools.columns:
            f.fail(f"{POOLS_FILE}: missing column {col!r}")
            return
    if pools["pool_gbp"].isna().any():
        f.fail(f"{POOLS_FILE}: pool_gbp has nulls")
    if (pools["pool_gbp"] <= 0).any():
        f.fail(f"{POOLS_FILE}: non-positive pool")
    if not pd.api.types.is_numeric_dtype(pools["pool_gbp"]):
        f.fail(f"{POOLS_FILE}: pool_gbp is not numeric")


def run(df: pd.DataFrame | None = None, check_pools: bool = True) -> Findings:
    f = Findings()
    if df is None:
        if not HISTORY_FILE.exists():
            f.fail(f"{HISTORY_FILE} does not exist")
            return f
        df = pd.read_csv(HISTORY_FILE)

    check_schema(df, f)
    if f.invalid:
        return f                      # structure first; the rest would lie
    check_draw_rows(df, f)
    check_categories(df, f)
    if check_pools:
        check_ev_inputs(f)
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="treat UNKNOWN_BUT_VALID as failure too")
    args = ap.parse_args()

    f = run()
    print("=" * 68)
    print("DATA CONTRACT")
    print("=" * 68)

    for msg in f.invalid:
        print(f"  INVALID  {msg}")
    for msg in f.unknown:
        print(f"  UNKNOWN  {msg}")
    if not f.invalid and not f.unknown:
        print("  KNOWN    every value recognised, every structural rule held")
    print()
    print(f"STATUS: {f.status}")

    if f.invalid:
        print("Ingest should stop: these cannot be legitimate new values.")
        return 1
    if f.unknown:
        print("Ingest continues - new values are kept as-is. Confirm them,")
        print("then add them to the KNOWN_* sets in this file.")
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
