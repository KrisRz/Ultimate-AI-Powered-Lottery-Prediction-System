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
  PYTHONPATH=. python scripts/wheel_play.py                  # 6 lines, pool 12
  PYTHONPATH=. python scripts/wheel_play.py --lines 10 --pool-size 11
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lottery.ev import (  # noqa: E402
    N_BALLS,
    DrawConditions,
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


# Optimal lotto designs, by pool size, as 1-based indices into the pool.
#
# A greedy search finds the "3 if 4" guarantee on a 12-number pool at EIGHT
# tickets and never at six - it commits to a locally best first ticket and
# cannot see the symmetric arrangement. The optimum has been known for
# decades: the (12, 6, 3, 4) lotto design has 6 blocks, and 5 is proved
# impossible (Karim 2005). Taken from coveringrepository.com, which formally
# replaced the La Jolla Covering Repository in March 2026, and verified here
# with this file's own `measure_guarantees` - {4: 3, 5: 3, 6: 3}, identical to
# greedy at 8 and 10 tickets, for GBP 12 instead of GBP 20.
#
# Two of the six lines share four numbers. That is a property of the design,
# not a mistake: covering every quadruple with 6 blocks needs the overlap.
KNOWN_COVERINGS = {
    12: [[1, 2, 4, 10, 11, 12], [1, 3, 7, 8, 9, 10], [4, 5, 6, 8, 9, 11],
         [2, 5, 6, 8, 9, 12], [1, 3, 5, 6, 7, 10], [2, 3, 4, 7, 11, 12]],
}


def covering_lines(pool: list[int]) -> int | None:
    """How many lines the known optimal design needs on this pool, if any."""
    design = KNOWN_COVERINGS.get(len(pool))
    return len(design) if design else None


def build_wheel(pool: list[int], n_lines: int) -> list[list[int]]:
    """The "3 if 4" guarantee: the known optimal design, or greedy after it.

    A pool quadruple counts as covered when some ticket matches at least 3
    of it - cover them all and any draw putting 4 winners in the pool pays a
    guaranteed Match 3. That is the strongest guarantee available on a
    12-number pool at any size worth buying (full 3-if-3 needs every one of the
    220 triples on a ticket; even 10 tickets hold at most 200).

    KNOWN_COVERINGS supplies it outright where the optimum is published; the
    greedy search below is the fallback for every other pool size, and tops up
    beyond the design when more lines are asked for than the guarantee needs.
    Greedy scores each candidate by new quadruples covered, then plain triple
    coverage, then the least popular line. Deterministic for a given pool
    and size.
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

    # Start from the optimal design where one is known and there is room for
    # it; greedy then only ever tops up beyond a guarantee already complete.
    design = KNOWN_COVERINGS.get(len(pool))
    if design and n_lines >= len(design):
        for block in design:
            line = tuple(sorted(pool[i - 1] for i in block))
            tickets.append(list(line))
            covered3 |= set(combinations(line, 3))
            covered4 |= quads_hit(line)
            if line in candidates:
                candidates.remove(line)
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


MAX_POOL = 16   # the greedy step scores every C(pool, 6) candidate per line


@dataclass(frozen=True)
class Wheel:
    """A built wheel and everything measured about it; see `wheel_portfolio`."""
    pool: list
    pool_size: int
    tickets: list
    guarantees: dict
    optimal: int | None
    hit3: int
    n_triples: int
    cond: DrawConditions


def wheel_portfolio(pool_size: int = 12, lines: int | None = None,
                    cond: DrawConditions | None = None) -> Wheel:
    """Wheel the `pool_size` least-played numbers. Writes nothing.

    `lines` defaults to the published minimum for the pool's guarantee (6 on
    a pool of 12). Raises ValueError on a pool too small or too slow to wheel.
    """
    if pool_size < N_PICK:
        raise ValueError(f"--pool-size must be at least {N_PICK}")
    # The greedy step scores every remaining C(pool, 6) candidate each
    # iteration; past 16 the candidate set explodes and the run crawls.
    if pool_size > MAX_POOL:
        raise ValueError("--pool-size above 16 makes the greedy search "
                         "impractically slow (and dilutes the unpopular-pool edge)")
    pool = unpopular_pool(pool_size)
    optimal = covering_lines(pool)
    n_lines = lines if lines is not None else (optimal or 10)
    tickets = build_wheel(pool, n_lines)
    return Wheel(
        pool=pool, pool_size=pool_size, tickets=tickets,
        guarantees=measure_guarantees(pool, tickets), optimal=optimal,
        hit3=len({tr for t in tickets for tr in combinations(sorted(t), 3)}),
        n_triples=len(list(combinations(pool, 3))),
        cond=cond if cond is not None else next_draw_conditions(),
    )


def render_wheel(w: Wheel) -> str:
    """The wheel printout, exactly as `make wheel` shows it."""
    out: list[str] = []
    say = out.append
    cond, tickets, n_lines = w.cond, w.tickets, len(w.tickets)
    say("=" * 64)
    say("WHEEL GENERATOR - abbreviated wheel on the unpopular pool")
    say("=" * 64)
    say(f"Pool ({w.pool_size} least popular): " + " ".join(f"{n}" for n in w.pool))
    say("Any line, any round:  1 in 45,057,474 - a wheel changes how wins")
    say("                      clump, never whether they come")
    say(f"Triple coverage:      {w.hit3}/{w.n_triples} "
        f"({100 * w.hit3 / w.n_triples:.0f}%) of pool triples on a ticket")
    say("Measured guarantees (if t winning numbers land in the pool,")
    say("best ticket matches at least g):")
    for t, g in w.guarantees.items():
        note = " -> guaranteed Match 3+ prize" if g >= 3 else ""
        say(f"  t={t}: g={g}{note}")
    if w.optimal:
        if n_lines == w.optimal:
            say(f"Optimal design:       {w.optimal} lines is the published minimum for "
                f"this pool")
        elif n_lines > w.optimal:
            say(f"NOTE: the guarantee is complete at {w.optimal} lines "
                f"(£{(n_lines - w.optimal) * 2:.0f} of these {n_lines} buy nothing "
                f"the first {w.optimal} do not already guarantee)")
    say("-" * 64)
    total_ev = 0.0
    for i, line in enumerate(tickets, 1):
        ev = line_ev(line, cond)
        total_ev += ev
        nums = " ".join(f"{n:2d}" for n in line)
        say(f"  {i}. {nums}   EV £{ev:+.3f}   popularity x{popularity_ratio(line):.2f}")
    say("-" * 64)
    say(f"Portfolio: {len(tickets)} lines, cost £{len(tickets) * cond.ticket_price:.2f}, "
        f"total EV £{total_ev:+.2f}")
    say("=" * 64)
    return "\n".join(out)


def save_wheel(w: Wheel, out_dir: Path = OUT_DIR) -> Path:
    """Write the wheel to its own timestamped file and return its path.

    Deliberately NOT latest.json: that file is the ledger's record of the
    EV advisor's real verdict; a wheel run is a one-off side portfolio.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"wheel_portfolio_{ts}.json"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "predictions": w.tickets,
        "metadata": {
            "method": "wheel_portfolio",
            "pool": w.pool,
            "pool_size": w.pool_size,
            "triple_coverage": f"{w.hit3}/{w.n_triples}",
            "guarantees": {str(t): g for t, g in w.guarantees.items()},
            "per_line": [{"line": line, "ev": line_ev(line, w.cond),
                          "popularity_ratio": popularity_ratio(line)}
                         for line in w.tickets],
        },
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=int, default=None,
                        help="Tickets to generate (default: as many as the "
                             "guarantee needs, 6 on a pool of 12)")
    parser.add_argument("--pool-size", type=int, default=12,
                        help="Unpopular numbers to wheel (11-14 is sensible)")
    args = parser.parse_args()
    try:
        wheel = wheel_portfolio(args.pool_size, args.lines)
    except ValueError as exc:
        parser.error(str(exc))
    print(render_wheel(wheel))
    out_file = save_wheel(wheel)
    print(f"Saved to {out_file} (latest.json untouched - wheel runs are side bets)")


if __name__ == "__main__":
    main()
