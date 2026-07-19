"""Expected-value model for UK Lotto (6 of 59, two rounds per draw since 2026-06-10).

Three ingredients:

1. Exact win probabilities per round - hypergeometric for the main numbers,
   with the bonus ball drawn from the remaining 53 balls. These match the
   published odds (jackpot 1 : 45,057,474; 5+bonus 1 : 7,509,579; ...).

2. A ticket-popularity model. Prizes for 5+bonus and below are fixed amounts,
   so sharing only matters for the (pari-mutuel) jackpot and for Must-Be-Won
   roll-downs. People overwhelmingly play dates (1-31), small "lucky" numbers
   and visual patterns; a line avoiding those shares a jackpot with fewer
   people. Weights below are literature-based heuristics (Cook & Clotfelter;
   ticket-popularity studies) and are meant to be calibrated against
   data/prize_tiers.csv winner counts as they accumulate.

3. EV per line = sum over both rounds of tier probabilities x payouts
   (jackpot payout discounted by expected co-winners) minus the ticket price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb, exp
from typing import Iterable, List, Sequence

TICKET_PRICE = 2.0
N_BALLS = 59
N_PICK = 6
TOTAL_COMBOS = comb(N_BALLS, N_PICK)  # 45,057,474

# Fixed per-winner prizes by tier, observed in official 2026 draw data
# (data/prize_tiers.csv: prize_total / winners). 5+bonus had no winners in
# observed draws yet - placeholder from pre-2026 structure, update when seen.
PRIZE_MATCH_5_BONUS = 250_000.0
PRIZE_MATCH_5 = 1_000.0
PRIZE_MATCH_4 = 50.0
PRIZE_MATCH_3 = 24.0
PRIZE_MATCH_2 = 5.0

# Typical number of lines sold per draw (UK Lotto). Only affects jackpot
# sharing and roll-down splits; override from CLI when you know better.
DEFAULT_TICKETS_SOLD = 15_000_000


def match_probability(k: int) -> float:
    """P(exactly k of the player's 6 numbers are among the 6 drawn)."""
    return comb(N_PICK, k) * comb(N_BALLS - N_PICK, N_PICK - k) / TOTAL_COMBOS


# Bonus ball is drawn from the 53 balls left after the main six. With exactly
# 5 matched, the player's remaining number is the bonus with probability 1/53.
P_JACKPOT = match_probability(6)
P_MATCH_5_BONUS = match_probability(5) / 53.0
P_MATCH_5 = match_probability(5) * 52.0 / 53.0
P_MATCH_4 = match_probability(4)
P_MATCH_3 = match_probability(3)
P_MATCH_2 = match_probability(2)
P_ANY_CASH = P_JACKPOT + P_MATCH_5_BONUS + P_MATCH_5 + P_MATCH_4 + P_MATCH_3 + P_MATCH_2


# --- Popularity model -------------------------------------------------------

def number_weight(n: int) -> float:
    """Relative pick-rate of a single number vs uniform (1.0 = average).

    Heuristic, calibration-pending: dates (1-31) are over-played, especially
    1-12 (both day and month); numbers above 31 are under-played; a few
    culturally "lucky" numbers get an extra boost.
    """
    if n <= 12:
        w = 1.35
    elif n <= 31:
        w = 1.20
    else:
        w = 0.72
    lucky = {3: 1.10, 7: 1.30, 11: 1.10, 17: 1.05, 23: 1.05}
    return w * lucky.get(n, 1.0)


MEAN_WEIGHT = sum(number_weight(n) for n in range(1, N_BALLS + 1)) / N_BALLS


def _has_consecutive_run(line: Sequence[int], run: int) -> bool:
    s = sorted(line)
    streak = 1
    for a, b in zip(s, s[1:]):
        streak = streak + 1 if b == a + 1 else 1
        if streak >= run:
            return True
    return False


def _is_arithmetic(line: Sequence[int]) -> bool:
    s = sorted(line)
    diffs = {b - a for a, b in zip(s, s[1:])}
    return len(diffs) == 1


def popularity_ratio(line: Sequence[int]) -> float:
    """How much more (>1) or less (<1) likely other players are to hold this
    exact line, relative to a uniformly random pick.

    Independent weighted-pick model (product of number weights, normalized)
    times pattern multipliers for visually attractive tickets.
    """
    ratio = 1.0
    for n in line:
        ratio *= number_weight(n) / MEAN_WEIGHT

    if _is_arithmetic(line):
        ratio *= 8.0        # 1-2-3-4-5-6, 5-10-15-..., very heavily played
    elif _has_consecutive_run(line, 3):
        ratio *= 1.8
    if all(n <= 31 for n in line):
        ratio *= 1.6        # pure-birthday ticket
    return ratio


def expected_cowinner_share(line: Sequence[int], tickets_sold: int) -> float:
    """E[1 / (1 + K)] where K ~ Poisson(lambda) is the number of OTHER
    jackpot winners holding this line. lambda = tickets * P(pick this line)."""
    lam = tickets_sold * popularity_ratio(line) / TOTAL_COMBOS
    if lam < 1e-12:
        return 1.0
    return (1.0 - exp(-lam)) / lam


# --- EV ---------------------------------------------------------------------

@dataclass
class DrawConditions:
    """What we know about the next draw event."""
    jackpot: float = 2_000_000.0          # per round
    tickets_sold: int = DEFAULT_TICKETS_SOLD
    roll_down: bool = False               # Must-Be-Won draw
    rounds: int = 2                       # two rounds per event since 2026-06-10
    ticket_price: float = TICKET_PRICE


def line_ev(line: Sequence[int], cond: DrawConditions) -> float:
    """Expected value in GBP of playing `line` for one draw event (all rounds
    included, ticket price subtracted)."""
    fixed_ev = (
        P_MATCH_5_BONUS * PRIZE_MATCH_5_BONUS
        + P_MATCH_5 * PRIZE_MATCH_5
        + P_MATCH_4 * PRIZE_MATCH_4
        + P_MATCH_3 * PRIZE_MATCH_3
        + P_MATCH_2 * PRIZE_MATCH_2
    )
    jackpot_ev = P_JACKPOT * cond.jackpot * expected_cowinner_share(line, cond.tickets_sold)

    rolldown_ev = 0.0
    if cond.roll_down:
        # Must-Be-Won: if nobody hits the jackpot (near-certain), it is split
        # across all lower-tier cash winners. Uniform-share approximation:
        # this line's expected slice = J x P(line wins any cash tier)
        # / E[number of winning lines among all tickets].
        p_no_jackpot = exp(-cond.tickets_sold * P_JACKPOT)
        expected_winning_lines = max(cond.tickets_sold * P_ANY_CASH, 1.0)
        slice_per_winning_line = cond.jackpot / expected_winning_lines
        rolldown_ev = p_no_jackpot * P_ANY_CASH * slice_per_winning_line

    per_round = fixed_ev + jackpot_ev + rolldown_ev
    return cond.rounds * per_round - cond.ticket_price


def should_play(cond: DrawConditions, threshold: float = 0.0) -> dict:
    """Decide whether the draw is worth entering at all.

    Uses a maximally unpopular reference line - if even that line's EV is
    below `threshold`, skip the draw. Default threshold 0.0 = only play
    +EV draws (in practice: rare Must-Be-Won roll-downs / huge rollovers).
    """
    reference = best_unpopular_reference_line()
    ev = line_ev(reference, cond)
    return {
        "play": ev >= threshold,
        "ev_best_line": ev,
        "reference_line": list(reference),
        "threshold": threshold,
        "conditions": {
            "jackpot_per_round": cond.jackpot,
            "rounds": cond.rounds,
            "roll_down": cond.roll_down,
            "tickets_sold": cond.tickets_sold,
        },
    }


def best_unpopular_reference_line() -> List[int]:
    """A deterministic line from the least-played numbers (no patterns)."""
    candidates = sorted(range(1, N_BALLS + 1), key=number_weight)
    line: List[int] = []
    for n in candidates:
        if any(abs(n - m) == 1 for m in line):
            continue  # avoid consecutive-pair pattern appeal
        line.append(n)
        if len(line) == N_PICK:
            break
    return sorted(line)
