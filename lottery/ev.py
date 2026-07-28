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

3. EV per line: fixed tiers pay per round; the jackpot is ONE pool per draw
   event, shared across both rounds (Allwyn, June 2026: "the jackpot will be
   shared across both rounds, while all other prize tiers will continue to
   offer fixed cash prizes, paid per round"). Jackpot payout is discounted by
   expected co-winners from either round; ticket price subtracted once.
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
# (data/prize_tiers.csv: prize_total / winners). 5+bonus is the well-known fixed
# GBP 1,000,000 tier, confirmed in every collected draw and the full history
# backfill (data/prize_tiers_history.csv).
#
# Match 3 / Match 2 are 10 / 1 in the two-round game - the base values, in 12 of
# the 13 two-round draws on record. The one exception, draw 3190 (2026-07-18),
# paid 24 / 5 because it was a ROLL-DOWN draw: the source page reads "GBP 10
# Rolldown Prize: GBP 24" and "GBP 1 Rolldown Prize: GBP 5". Earlier revisions
# read 24 / 5 off that single draw and treated it as the norm, which inflated
# every line's EV by ~GBP 1.07 and put the non-MBW break-even jackpot at 4.8M
# instead of 30M. Two guards against a repeat: these defaults are the base
# prizes only, and `calibrate_fixed_prizes` re-derives them from collected data
# with a median (roll-downs are a minority, so they cannot move it). The
# roll-down boost is priced separately, in `line_ev`'s rolldown_ev term.
#
# Sanity check on the level: 2 x these prizes = GBP 0.73 of a GBP 2 line (36%),
# plus the jackpot share ~= the ~50% UK Lotto returns to players. At 24 / 5 the
# fixed tiers alone would have returned 90%, which no lottery does.
PRIZE_MATCH_5_BONUS = 1_000_000.0
PRIZE_MATCH_5 = 1_000.0
PRIZE_MATCH_4 = 50.0
PRIZE_MATCH_3 = 10.0
PRIZE_MATCH_2 = 1.0

# Tier codes as they appear in data/prize_tiers.csv (1 = jackpot).
TIER_MATCH_5_BONUS = 2
TIER_MATCH_5 = 3
TIER_MATCH_4 = 4
TIER_MATCH_3 = 5
TIER_MATCH_2 = 6

# Typical number of lines sold per draw (UK Lotto). Only a fallback: with
# collected data, `estimate_tickets_sold` measures it per draw. The 15,000,000
# this used to hold was a guess; the median implied by tier winner counts across
# the whole 59-ball era is ~8.6M and the two-round era runs ~6.5-7.5M
# (data/prize_tiers_history.csv). Affects jackpot sharing and roll-down splits.
DEFAULT_TICKETS_SOLD = 7_500_000


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
    """Relative pick-rate of a single number vs uniform (population mean 1.0).

    Calibrated 2026-07-25 against 1,126 draws of Match-3 winner counts
    (scripts/calibrate_popularity.py, data/prize_tiers_history.csv): draws with
    more birthday-range numbers (<=31) yield systematically more low-tier
    winners per ticket, so those numbers are over-played. The gap is real but
    ~half the size the earlier literature heuristic assumed, and the per-number
    "lucky" boosts it posited (7, 11, ...) sat within noise - so only the three
    calibrated bucket levels survive. Dates 1-12 (both day and month) are the
    most over-played; numbers above 31 are the safe, under-played picks.
    """
    if n <= 12:
        return 1.23
    if n <= 31:
        return 1.10
    return 0.83


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

@dataclass(frozen=True)
class FixedPrizes:
    """Per-winner payouts for the tiers that are NOT pari-mutuel.

    Base prizes: what an ordinary draw pays. A roll-down draw pays more in the
    low tiers, but that uplift is the jackpot pool being redistributed and is
    modelled in `line_ev`, so it must not be baked in here as well.
    """
    match_5_bonus: float = PRIZE_MATCH_5_BONUS
    match_5: float = PRIZE_MATCH_5
    match_4: float = PRIZE_MATCH_4
    match_3: float = PRIZE_MATCH_3
    match_2: float = PRIZE_MATCH_2
    source: str = "default"               # "default" or "observed (N draws)"

    def ev_per_round(self) -> float:
        """Expected fixed-tier payout of one line in one round, before cost."""
        return (
            P_MATCH_5_BONUS * self.match_5_bonus
            + P_MATCH_5 * self.match_5
            + P_MATCH_4 * self.match_4
            + P_MATCH_3 * self.match_3
            + P_MATCH_2 * self.match_2
        )


@dataclass
class DrawConditions:
    """What we know about the next draw event."""
    jackpot: float = 2_000_000.0          # single pool per EVENT, shared across rounds
    tickets_sold: int = DEFAULT_TICKETS_SOLD
    roll_down: bool = False               # Must-Be-Won draw
    rounds: int = 2                       # two rounds per event since 2026-06-10
    ticket_price: float = TICKET_PRICE
    prizes: FixedPrizes = field(default_factory=FixedPrizes)


def line_ev(line: Sequence[int], cond: DrawConditions) -> float:
    """Expected value in GBP of playing `line` for one draw event (all rounds
    included, ticket price subtracted).

    Fixed tiers pay per round. The jackpot is one pool for the whole event:
    a ticket enters it once per round, and co-winners can come from either
    round, so the pool sees tickets_sold x rounds competing entries.
    """
    fixed_ev = cond.prizes.ev_per_round()
    jackpot_ev = (
        cond.rounds * P_JACKPOT * cond.jackpot
        * expected_cowinner_share(line, cond.tickets_sold * cond.rounds)
    )

    rolldown_ev = 0.0
    if cond.roll_down:
        # Must-Be-Won: if no entry in any round hits the jackpot
        # (near-certain), the single pool is split across all lower-tier cash
        # winners of both rounds. Uniform-share approximation - the rounds
        # cancel: per-ticket slice = P(no jackpot) x J / tickets_sold.
        #
        # Validated against draw 3190 (2026-07-18), the first roll-down we hold
        # data for: the boost went to Match 3 (+GBP 14 x 169,438 winners) and
        # Match 2 (+GBP 4 x 1,756,390), Match 4/5 untouched, total GBP 9.40M
        # redistributed against a GBP 9.56M must-be-won pool (98%). Model term
        # J/N = GBP 1.28/line vs GBP 1.26 actually paid out.
        #
        # Note this uplift is popularity-blind: a roll-down pays fixed boosts
        # to low-tier winners, so an unpopular line gains nothing extra here.
        p_no_jackpot = exp(-cond.tickets_sold * cond.rounds * P_JACKPOT)
        rolldown_ev = p_no_jackpot * cond.jackpot / max(cond.tickets_sold, 1)

    return cond.rounds * fixed_ev + jackpot_ev + rolldown_ev - cond.ticket_price


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
        "break_even_jackpot": break_even_jackpot(cond, reference),
        "conditions": {
            "jackpot_event_pool": cond.jackpot,
            "rounds": cond.rounds,
            "roll_down": cond.roll_down,
            "tickets_sold": cond.tickets_sold,
            "prizes": {
                "match_5_bonus": cond.prizes.match_5_bonus,
                "match_5": cond.prizes.match_5,
                "match_4": cond.prizes.match_4,
                "match_3": cond.prizes.match_3,
                "match_2": cond.prizes.match_2,
                "source": cond.prizes.source,
            },
        },
    }


def estimate_tickets_sold(tiers_df, last_n_draws: int = 20) -> int | None:
    """Estimate lines sold per draw from observed winner counts.

    For fixed-prize tiers the expected winner count is N x P(tier), so each
    observation gives N ~= winners / P. Uses the high-count tiers (match 4/3/2,
    least distorted by jackpot-sharing) over recent draws, both rounds, and
    takes the median to damp popularity-of-drawn-numbers noise.

    This is the first model parameter that becomes measurable from collected
    data - it directly sharpens jackpot-sharing and roll-down EV.
    """
    if tiers_df is None or len(tiers_df) == 0:
        return None
    tier_probs = {4: P_MATCH_4, 5: P_MATCH_3, 6: P_MATCH_2}
    recent_draws = sorted(tiers_df["draw_number"].unique())[-last_n_draws:]
    sample = tiers_df[
        tiers_df["draw_number"].isin(recent_draws)
        & tiers_df["tier"].isin(tier_probs)
        & (tiers_df["winners"] > 0)
    ]
    if len(sample) == 0:
        return None
    estimates = sorted(
        row["winners"] / tier_probs[int(row["tier"])] for _, row in sample.iterrows()
    )
    return int(estimates[len(estimates) // 2])


def calibrate_fixed_prizes(tiers_df, last_n_draws: int = 30,
                           min_rows: int = 3) -> FixedPrizes:
    """Re-derive the fixed per-winner prizes from collected official data.

    Each (draw, round) row of data/prize_tiers.csv gives prize_total / winners
    for a tier - the exact amount that tier paid. The estimator is the MEDIAN
    over rows, and that choice is the whole safety mechanism: roll-down draws
    boost Match 3 / Match 2 (3190 paid 24 / 5 over a base of 10 / 1) but are a
    small minority, so they cannot move the median. Taking a mean - or reading
    a single draw, which is how the 24 / 5 bug got in - would fold the
    redistributed jackpot into the base prizes and double-count it against
    `line_ev`'s rolldown_ev term.

    Tiers with fewer than `min_rows` observations keep their module default
    (5+bonus is usually won by nobody, so it rarely clears the bar).
    """
    fallback = FixedPrizes()
    if tiers_df is None or len(tiers_df) == 0:
        return fallback

    recent_draws = sorted(tiers_df["draw_number"].unique())[-last_n_draws:]
    sample = tiers_df[
        tiers_df["draw_number"].isin(recent_draws)
        & (tiers_df["winners"] > 0)
        & (tiers_df["prize_total"] > 0)
    ]
    if len(sample) == 0:
        return fallback

    def median_for(tier: int, default: float) -> float:
        rows = sample[sample["tier"] == tier]
        if len(rows) < min_rows:
            return default
        per_winner = sorted(rows["prize_total"] / rows["winners"])
        mid = len(per_winner) // 2
        value = (per_winner[mid] if len(per_winner) % 2
                 else (per_winner[mid - 1] + per_winner[mid]) / 2)
        return round(float(value), 2)

    return FixedPrizes(
        match_5_bonus=median_for(TIER_MATCH_5_BONUS, fallback.match_5_bonus),
        match_5=median_for(TIER_MATCH_5, fallback.match_5),
        match_4=median_for(TIER_MATCH_4, fallback.match_4),
        match_3=median_for(TIER_MATCH_3, fallback.match_3),
        match_2=median_for(TIER_MATCH_2, fallback.match_2),
        source=f"observed ({len(recent_draws)} draws)",
    )


def break_even_jackpot(cond: DrawConditions,
                       line: Sequence[int] | None = None) -> float:
    """Jackpot pool at which `line` breaks even under `cond`'s other terms.

    Answers "how big does it have to get?" directly, instead of leaving the
    reader to invert the EV formula. Returns inf if no jackpot can do it.
    """
    line = list(line) if line is not None else best_unpopular_reference_line()
    shortfall = cond.ticket_price - cond.rounds * cond.prizes.ev_per_round()
    if shortfall <= 0:
        return 0.0
    per_pound = (
        cond.rounds * P_JACKPOT
        * expected_cowinner_share(line, cond.tickets_sold * cond.rounds)
    )
    if cond.roll_down:
        per_pound += exp(-cond.tickets_sold * cond.rounds * P_JACKPOT) / max(cond.tickets_sold, 1)
    return shortfall / per_pound if per_pound > 0 else float("inf")


def best_unpopular_reference_line() -> List[int]:
    """A deterministic line from the least-played numbers (no patterns)."""
    candidates = sorted(range(1, N_BALLS + 1), key=number_weight)
    line: List[int] = []
    for n in candidates:
        if any(abs(n - m) == 1 for m in line):
            continue  # avoid consecutive-pair pattern appeal
        trial = sorted(line + [n])
        if len(trial) >= 3 and len({b - a for a, b in zip(trial, trial[1:])}) == 1:
            continue  # constant-step lines carry the x8 arithmetic penalty
        line.append(n)
        if len(line) == N_PICK:
            break
    return sorted(line)
