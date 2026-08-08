#!/usr/bin/env python3
"""One-off abbreviated-wheel generator over a pool of unpopular numbers.

A wheel does not improve any line's odds - every 6-of-59 combination stays
1 in 45,057,474 per round. What it buys is a COVERAGE GUARANTEE inside a
small pool: if enough of the winning numbers land in the pool, at least one
ticket is guaranteed a minimum match, so wins arrive clumped instead of
scattered. The guarantees printed below are measured against the actual
tickets (exhaustive check), never assumed from a published wheel table.

The pool is the K least-played numbers from the calibrated popularity
weights, so the wheel keeps the one real edge this project has: lines that
share prizes with fewer co-winners.

Usage:
  PYTHONPATH=. python scripts/wheel_play.py                  # 10 lines, pool 12
  PYTHONPATH=. python scripts/wheel_play.py --lines 10 --pool-size 11
"""

import argparse
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lottery.ev import (  # noqa: E402
    N_BALLS,
    N_PICK,
    _has_consecutive_run,
    line_ev,
    number_weight,
    popularity_ratio,
)
from scripts.ev_play import next_draw_conditions  # noqa: E402

OUT_DIR = Path("outputs/predictions")


def unpopular_pool(size: int) -> list[int]:
    """The `size` least-played numbers by calibrated popularity weight.

    The calibrated weights are banded (32-49 share one weight), so the size
    cutoff usually falls inside a tie. Spread the tied band evenly instead of
    taking its lowest run: EV is identical either way, but a contiguous pool
    like 32-43 starves the no-3-consecutive filter of candidate tickets.
    """
    ranked = sorted(range(1, N_BALLS + 1), key=lambda n: (number_weight(n), n))
    cut = number_weight(ranked[size - 1])
    below = [n for n in ranked if number_weight(n) < cut]
    tied = sorted(n for n in ranked if number_weight(n) == cut)
    need = size - len(below)
    spread = [tied[i * len(tied) // need] for i in range(need)]
    return sorted(below + spread)


def build_wheel(pool: list[int], n_lines: int) -> list[list[int]]:
    """Greedy covering design aimed straight at the "3 if 4" guarantee.

    A pool quadruple counts as covered when some ticket matches at least 3
    of it - cover them all and any draw putting 4 winners in the pool pays a
    guaranteed Match 3. That is the strongest guarantee 10 tickets can chase
    on a 12-number pool (full 3-if-3 needs every one of the 220 triples on a
    ticket; 10 tickets hold at most 200). Plain triple coverage breaks ties,
    then the least popular line. Deterministic for a given pool and size.
    """
    def quads_hit(c) -> set:
        # Quadruples sharing >=3 numbers with ticket c: each triple of c
        # plus any 4th pool number.
        out = set()
        for tr in combinations(c, 3):
            for x in pool:
                if x not in tr:
                    out.add(tuple(sorted(tr + (x,))))
        return out

    candidates = [
        c for c in combinations(pool, N_PICK)
        if not _has_consecutive_run(c, 3)
    ]
    if n_lines > len(candidates):
        raise ValueError(
            f"only {len(candidates)} valid tickets exist on this pool, "
            f"cannot build {n_lines} lines"
        )
    covered3: set = set()
    covered4: set = set()
    tickets: list[list[int]] = []
    while len(tickets) < n_lines:
        best = max(
            candidates,
            key=lambda c: (
                len(quads_hit(c) - covered4),
                len(set(combinations(c, 3)) - covered3),
                -popularity_ratio(c),
            ),
        )
        tickets.append(list(best))
        covered3 |= set(combinations(best, 3))
        covered4 |= quads_hit(best)
        candidates.remove(best)
    return tickets


def measure_guarantees(pool: list[int], tickets: list[list[int]]) -> dict:
    """Exhaustive worst-case match guarantee for each pool-hit count.

    guarantee[t] = the match at least one ticket ALWAYS achieves when exactly
    t of the winning numbers fall in the pool, minimized over every possible
    t-subset. This is the honest version of a wheel table's "3 if 4" claim.
    """
    ticket_sets = [set(t) for t in tickets]
    out = {}
    for t in range(3, N_PICK + 1):
        out[t] = min(
            max(len(ts & set(sub)) for ts in ticket_sets)
            for sub in combinations(pool, t)
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=int, default=10, help="Tickets to generate")
    parser.add_argument("--pool-size", type=int, default=12,
                        help="Unpopular numbers to wheel (11-14 is sensible)")
    args = parser.parse_args()

    if args.pool_size < N_PICK:
        parser.error(f"--pool-size must be at least {N_PICK}")
    # The greedy step scores every remaining C(pool, 6) candidate each
    # iteration; past 16 the candidate set explodes and the run crawls.
    if args.pool_size > 16:
        parser.error("--pool-size above 16 makes the greedy search "
                     "impractically slow (and dilutes the unpopular-pool edge)")

    pool = unpopular_pool(args.pool_size)
    tickets = build_wheel(pool, args.lines)
    guar = measure_guarantees(pool, tickets)
    cond = next_draw_conditions()

    n_triples = len(list(combinations(pool, 3)))
    hit3 = len({tr for t in tickets for tr in combinations(sorted(t), 3)})

    print("=" * 64)
    print("WHEEL GENERATOR - abbreviated wheel on the unpopular pool")
    print("=" * 64)
    print(f"Pool ({args.pool_size} least popular): "
          + " ".join(f"{n}" for n in pool))
    print("Any line, any round:  1 in 45,057,474 - a wheel changes how wins")
    print("                      clump, never whether they come")
    print(f"Triple coverage:      {hit3}/{n_triples} "
          f"({100 * hit3 / n_triples:.0f}%) of pool triples on a ticket")
    print("Measured guarantees (if t winning numbers land in the pool,")
    print("best ticket matches at least g):")
    for t, g in guar.items():
        note = " -> guaranteed Match 3+ prize" if g >= 3 else ""
        print(f"  t={t}: g={g}{note}")
    print("-" * 64)
    total_ev = 0.0
    for i, line in enumerate(tickets, 1):
        ev = line_ev(line, cond)
        total_ev += ev
        nums = " ".join(f"{n:2d}" for n in line)
        print(f"  {i}. {nums}   EV £{ev:+.3f}   popularity x{popularity_ratio(line):.2f}")
    print("-" * 64)
    print(f"Portfolio: {len(tickets)} lines, cost £{len(tickets) * cond.ticket_price:.2f}, "
          f"total EV £{total_ev:+.2f}")
    print("=" * 64)

    # Deliberately NOT latest.json: that file is the ledger's record of the
    # EV advisor's real verdict; a wheel run is a one-off side portfolio.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"wheel_portfolio_{ts}.json"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "predictions": tickets,
        "metadata": {
            "method": "wheel_portfolio",
            "pool": pool,
            "pool_size": args.pool_size,
            "triple_coverage": f"{hit3}/{n_triples}",
            "guarantees": {str(t): g for t, g in guar.items()},
            "per_line": [{"line": line, "ev": line_ev(line, cond),
                          "popularity_ratio": popularity_ratio(line)}
                         for line in tickets],
        },
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {out_file} (latest.json untouched - wheel runs are side bets)")


if __name__ == "__main__":
    main()
