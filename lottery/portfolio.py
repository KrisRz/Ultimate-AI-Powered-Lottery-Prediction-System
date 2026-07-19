"""Portfolio builder: N lines that maximize EV while staying diverse.

EV differences between lines come only from jackpot-sharing (fixed lower-tier
prizes are identical for every line), so the optimizer prefers unpopular
lines and enforces diversity constraints so one drawn set can't wipe out or
crown the whole portfolio at once.
"""

from __future__ import annotations

import random
from typing import List, Sequence

from .ev import (
    DrawConditions,
    N_BALLS,
    N_PICK,
    _has_consecutive_run,
    line_ev,
    number_weight,
    popularity_ratio,
)

MAX_PAIRWISE_OVERLAP = 2
MIN_HIGH_NUMBERS = 2          # numbers > 31 per line (anti-birthday)
SUM_BAND = (100, 260)         # loose band around the uniform mean (180)


def _passes_constraints(line: Sequence[int]) -> bool:
    return (
        len(set(line)) == N_PICK
        and all(1 <= n <= N_BALLS for n in line)
        and sum(1 for n in line if n > 31) >= MIN_HIGH_NUMBERS
        and SUM_BAND[0] <= sum(line) <= SUM_BAND[1]
        and not _has_consecutive_run(line, 3)
    )


def _sample_unpopular_line(rng: random.Random) -> List[int]:
    """Sample a line with probability tilted toward under-played numbers."""
    weights = [1.0 / number_weight(n) for n in range(1, N_BALLS + 1)]
    numbers = list(range(1, N_BALLS + 1))
    line: List[int] = []
    w = weights[:]
    for _ in range(N_PICK):
        pick = rng.choices(range(len(numbers)), weights=w, k=1)[0]
        line.append(numbers.pop(pick))
        w.pop(pick)
    return sorted(line)


def build_portfolio(
    n_lines: int,
    cond: DrawConditions,
    seed: int | None = None,
    candidates_per_line: int = 400,
) -> List[dict]:
    """Greedy selection: repeatedly pick the highest-EV candidate that
    overlaps every already-chosen line in at most MAX_PAIRWISE_OVERLAP
    numbers."""
    rng = random.Random(seed)
    chosen: List[dict] = []

    while len(chosen) < n_lines:
        pool = []
        for _ in range(candidates_per_line):
            line = _sample_unpopular_line(rng)
            if not _passes_constraints(line):
                continue
            if any(
                len(set(line) & set(c["line"])) > MAX_PAIRWISE_OVERLAP
                for c in chosen
            ):
                continue
            pool.append(line)
        if not pool:
            raise RuntimeError(
                "Could not sample a line satisfying the diversity constraints "
                f"after {candidates_per_line} tries - relax the constraints"
            )
        best = max(pool, key=lambda ln: line_ev(ln, cond))
        chosen.append({
            "line": best,
            "ev": line_ev(best, cond),
            "popularity_ratio": popularity_ratio(best),
        })
    return chosen
