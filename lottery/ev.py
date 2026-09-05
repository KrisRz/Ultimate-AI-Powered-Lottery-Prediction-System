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

from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from math import comb, exp
from typing import Iterable, List, Sequence
from zoneinfo import ZoneInfo

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

# The official split, and the identity it hands us
# ------------------------------------------------
# "8.88% of sales for a Lotto Draw are allocated to the Jackpot"
#   - Lotto Online Game Procedures, Edition 22 (7 June 2026), section 3.1
#   - and the Gambling Commission Section 6 licence of the same date, which
#     also sets the minimum jackpot at GBP 2m for both Wednesday and Saturday
#     (it was GBP 3.8m on Saturdays under the 2024 licence)
#
# Nobody publishes sales per draw - the Gambling Commission answered a 2026 FOI
# with "does not collect data on Lotto ticket sales by day of the week" and
# Allwyn reports quarterly totals - but that clause means sales do not have to
# be published. On any draw that inherits a rolled-over pool,
#
#     sales(d) = (pool(d) - pool(d - 1)) / share
#
# is an identity. Verified against Merseyworld's archive on every two-round
# draw with both numbers: agreement to about a tenner, Must-Be-Won draws
# included (tests/test_draw_pools.py). Which also settles what that archive is -
# the same identity computed by a third party since 2003, not a second
# independent measurement.
#
# The draw after a jackpot is won is the one draw this cannot price: the pool
# resets to the minimum, so the difference measures the reset, not the sales.
JACKPOT_SHARE_OF_SALES = 0.0888
LEGACY_JACKPOT_SHARE_OF_SALES = 0.0979   # licence in force before 7 June 2026
TWO_ROUND_FIRST_DRAW = 3179              # 2026-06-10, the first two-round draw
MINIMUM_JACKPOT = 2_000_000.0

# How many exact observations it takes before they replace the winner-count
# estimator rather than merely joining it. Three same-weekday ordinary draws
# is one rollover cycle's worth - below that the median is the noise.
MIN_EXACT_OBSERVATIONS = 3


def jackpot_share_of_sales(draw_number: int) -> float:
    """Share of sales allocated to the jackpot, by era."""
    return (JACKPOT_SHARE_OF_SALES if draw_number >= TWO_ROUND_FIRST_DRAW
            else LEGACY_JACKPOT_SHARE_OF_SALES)


def exact_lines_sold(pools_df) -> dict:
    """Lines sold per draw, read off the pool each draw carried.

    `pools_df` is data/draw_pools.csv: draw_number, draw_date, pool_gbp, where
    pool_gbp is the pool the draw ACTUALLY carried (topPrize in the official
    feed), not the estimate advertised beforehand. The two differ by up to ~4%,
    which is 8-10% once divided into sales - see scripts/backfill_draw_pools.py.

    Returns {draw_number: lines} for every draw the identity can price, which
    is every draw whose pool grew on its predecessor. A pool that did not grow
    means the previous draw paid its jackpot out and this one restarted from
    the minimum: no sales information in that difference, so no entry.

    This is what replaces N ~ winners / P(tier) wherever it can. That estimator
    is unbiased across many draws but carries about +/-15% on a single one,
    because winner counts move with how popular the drawn numbers happened to
    be: draw 3196's round-two numbers were all birthday-range, which inflated
    its measured N by 15% and its scorecard read -11% forecast error where the
    truth was +2.6%.
    """
    if pools_df is None or len(pools_df) == 0:
        return {}
    pools = {int(r["draw_number"]): float(r["pool_gbp"])
             for _, r in pools_df.iterrows()}
    exact = {}
    for draw, pool in pools.items():
        previous = pools.get(draw - 1)
        if previous is None or pool <= previous:
            continue
        sales = (pool - previous) / jackpot_share_of_sales(draw)
        exact[draw] = int(round(sales / TICKET_PRICE))
    return exact


# A Must-Be-Won draw sells MORE than an ordinary one, and that matters more than
# anything else in this model: the roll-down term is J / N, so an ordinary-draw
# sales figure applied to a Must-Be-Won draw inflates its EV - on precisely the
# ~9 draws a year the alert exists to fire on. Measured 2026-08-06 against the
# 93 roll-down draws in data/prize_tiers_history.csv (identified by Match 3
# paying above its era base), each compared with the ordinary draws immediately
# before it: median 1.379 over the 59-ball era (n=89), quartiles 1.07 / 1.69.
# Stable across periods (1.26 - 1.48 by 2-4 year block).
#
# Paired against the PRECEDING draws, not against the era average, because UK
# Lotto sales drift down over the decade and roll-downs are not spread evenly
# across it. The window is 4 draws: comparing against only the 1-2 draws before
# a roll-down understates the effect to ~1.11, because those draws sit at
# rollover 3-4 and are already selling hot. Four draws is about one rollover
# cycle, which is the mix `estimate_tickets_sold` actually measures.
#
# What this figure does NOT separate is "Must-Be-Won" from "big jackpot": a
# roll-down always carries the largest pool of its cycle. Settled 2026-08-07
# with real per-draw sales (data/sales_history.csv, PR #13): pool size does
# NOT drive the uplift (log-pool coefficient -0.016 across all 93 roll-downs)
# - the confound that actually mattered was the WEEKDAY, priced below.
MBW_SALES_UPLIFT = 1.38
MBW_SALES_UPLIFT_P25 = 1.07
MBW_SALES_UPLIFT_P75 = 1.69

# Day-aware uplift, measured 2026-08-07 on real per-draw sales
# (data/sales_history.csv) in the SAME definition the estimator uses: target
# roll-down's lines over the median of the same-weekday, non-roll-down draws
# among the 20 draws before it. Saturdays sell ~1.59x Wednesdays, so a mixed
# baseline hides that a Wednesday Must-Be-Won needs a bigger relative jump
# than a Saturday one: Sat n=62, Wed n=31. The mixed-base constant above only
# ever fit Saturdays because 62 of the 93 calibration roll-downs were
# Saturdays - on a Wednesday it overstates N by ~25-30%, i.e. EV reads too
# pessimistic on exactly half the calendar. Kept as the fallback for
# conditions with no draw date.
#
# Quartiles are far tighter than the mixed ones (which mixed two day-level
# populations): the `robust` flag in sales_sensitivity becomes decisive
# instead of spanning a third of the EV range. Recent drift is mild and
# upward (since 2022: Sat 1.30 n=37, Wed 1.59 n=10) - the p75 end carries it.
MBW_UPLIFT_BY_WEEKDAY = {
    2: (1.44, 1.38, 1.54),   # Wednesday: (median, p25, p75)
    5: (1.27, 1.19, 1.31),   # Saturday
}


def mbw_uplift(draw_date: date | None = None) -> tuple:
    """(median, p25, p75) sales uplift for a Must-Be-Won draw on `draw_date`.

    Day-specific when the weekday is known, mixed-baseline constants when it
    is not. Draws only ever fall on Wednesday/Saturday; any other weekday
    means the caller's date is wrong, and the mixed fallback is the least
    wrong answer available.
    """
    if draw_date is not None:
        trio = MBW_UPLIFT_BY_WEEKDAY.get(draw_date.weekday())
        if trio:
            return trio
    return (MBW_SALES_UPLIFT, MBW_SALES_UPLIFT_P25, MBW_SALES_UPLIFT_P75)


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


ARITHMETIC_MULT = 8.0       # 1-2-3-4-5-6, 5-10-15-..., very heavily played
CONSECUTIVE_MULT = 1.8      # any run of three or more
BIRTHDAY_MULT = 1.6         # every number <= 31


def _raw_popularity(line: Sequence[int]) -> float:
    ratio = 1.0
    for n in line:
        ratio *= number_weight(n) / MEAN_WEIGHT

    if _is_arithmetic(line):
        ratio *= ARITHMETIC_MULT
    elif _has_consecutive_run(line, 3):
        ratio *= CONSECUTIVE_MULT
    if all(n <= 31 for n in line):
        ratio *= BIRTHDAY_MULT
    return ratio


def popularity_ratio(line: Sequence[int]) -> float:
    """How much more (>1) or less (<1) likely other players are to hold this
    exact line, relative to a uniformly random pick.

    Independent weighted-pick model (product of number weights) times pattern
    multipliers for visually attractive tickets, divided by the constant that
    makes the average over all C(59,6) lines exactly 1.0 - see
    POPULARITY_NORMALIZATION.
    """
    return _raw_popularity(line) / POPULARITY_NORMALIZATION


def _weight_sums(balls: int, pick: int = N_PICK) -> tuple:
    """Exact sums of the number-weight product over every C(balls, pick) line
    drawn from 1..balls: (all lines, only lines holding a run of 3+).

    Straight enumeration would mean 45M lines. This walks the balls once,
    carrying (chosen so far, trailing run length capped at 2, run-of-3 seen)
    and accumulating the weight products - 59 steps over <60 states.
    """
    states = {(0, 0, False): 1.0}
    for n in range(1, balls + 1):
        w = number_weight(n) / MEAN_WEIGHT
        nxt: dict = {}
        for (k, run, seen), acc in states.items():
            skip = (k, 0, seen)                     # n not on the ticket: run breaks
            nxt[skip] = nxt.get(skip, 0.0) + acc
            if k < pick:
                run_now = run + 1
                take = (k + 1, min(run_now, 2), seen or run_now >= 3)
                nxt[take] = nxt.get(take, 0.0) + acc * w
        states = nxt
    total = sum(v for (k, _, _), v in states.items() if k == pick)
    with_run = sum(v for (k, _, seen), v in states.items() if k == pick and seen)
    return total, with_run


def _arithmetic_lines(balls: int = N_BALLS, pick: int = N_PICK):
    """Every constant-step line, cheapest to enumerate directly (there are 319
    of them for 6 from 59)."""
    for step in range(1, (balls - 1) // (pick - 1) + 1):
        for start in range(1, balls - step * (pick - 1) + 1):
            yield [start + i * step for i in range(pick)]


def _popularity_normalization() -> float:
    """Mean of the raw popularity score over all C(59,6) lines.

    `popularity_ratio` is a pick rate *relative to uniform*, so its average
    across every possible line has to be exactly 1.0 - otherwise
    `tickets_sold * ratio / TOTAL_COMBOS` describes more (or fewer) tickets
    than were actually sold. The raw score averages 1.046: the pattern
    multipliers push mass in and nothing takes it back out. Dividing by this
    constant restores the invariant.

    Computed exactly rather than sampled, and recomputed from whatever the
    weights and multipliers currently are, so a recalibration cannot leave a
    stale constant behind. Let w(line) be the product of number weights and
    B(line) = 1.6 when every number is <= 31:

        sum = SUM w*B  +  7 * SUM_arithmetic w*B  +  0.8 * SUM_run3-only w*B

    (7 and 0.8 are the multipliers minus the 1.0 already counted; arithmetic
    wins over the run-of-3 branch, and the only arithmetic lines holding a run
    of 3 are the step-1 ones.)
    """
    all_lines, all_run3 = _weight_sums(N_BALLS)
    low_lines, low_run3 = _weight_sums(31)      # lines that are entirely <= 31

    def with_birthday(over_all: float, over_low: float) -> float:
        return over_all + (BIRTHDAY_MULT - 1.0) * over_low

    arith = arith_low = arith_run3 = arith_run3_low = 0.0
    for line in _arithmetic_lines():
        w = 1.0
        for n in line:
            w *= number_weight(n) / MEAN_WEIGHT
        is_low = line[-1] <= 31
        arith += w
        arith_low += w if is_low else 0.0
        if line[1] - line[0] == 1:              # step 1 => also a run of 3+
            arith_run3 += w
            arith_run3_low += w if is_low else 0.0

    total = (
        with_birthday(all_lines, low_lines)
        + (ARITHMETIC_MULT - 1.0) * with_birthday(arith, arith_low)
        + (CONSECUTIVE_MULT - 1.0) * (
            with_birthday(all_run3, low_run3)
            - with_birthday(arith_run3, arith_run3_low)
        )
    )
    return total / TOTAL_COMBOS


POPULARITY_NORMALIZATION = _popularity_normalization()


def expected_cowinner_share(line: Sequence[int], tickets_sold: int,
                            rounds: int = 1) -> float:
    """E[1 / (1 + K)], K ~ Poisson(lambda) other jackpot winners to split with.

    `tickets_sold` is entries, not entries x rounds. Each entry plays every
    round, and the procedures share the jackpot "equally between all Winning
    Lotto Entries in Round one and Round two" (Game Procedures Ed. 22, 3.1) -
    but the two rounds do not carry the same risk of sharing:

        round one    the winners hold MY numbers, so their expected popularity
                     is this line's - picking an unplayed line thins them out
        other rounds the winners hold THAT round's numbers, drawn independently
                     of anything on my slip, so their expected popularity is
                     1.0 whatever I choose

    Pricing every round at the line's own popularity, as this did until
    2026-09-05, overstated what an unpopular line buys on the jackpot: at
    N = 8m the reference line's share reads 0.946 that way and 0.892 correctly,
    so its advantage over a popular line halves from +12.4% to +5.9%. Half the
    exposure to sharing does not depend on what you write on the slip, and the
    page said otherwise.

    No decision moves: an ordinary draw would need a jackpot near GBP 32m
    before the term matters, and a roll-down's EV is dominated by J/N, which
    this does not touch.
    """
    per_round = tickets_sold / TOTAL_COMBOS
    lam = per_round * popularity_ratio(line) + per_round * max(rounds - 1, 0)
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
    rollover_count: int = 0               # consecutive rollovers so far (feed)
    draw_date: date | None = None         # picks the day-specific sales uplift


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
        * expected_cowinner_share(line, cond.tickets_sold, cond.rounds)
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


def sales_sensitivity(cond: DrawConditions, threshold: float = 0.0,
                      line: Sequence[int] | None = None) -> dict | None:
    """EV across the spread of the Must-Be-Won sales uplift, or None if the
    draw is not Must-Be-Won.

    On a roll-down draw the EV is dominated by J / N, so the whole verdict turns
    on one estimated number. The uplift is a median over historical roll-downs
    with real quartile spread - not noise around a known value. Reporting only
    the central figure is what let a verdict sitting 3.2% from break-even read
    as a clean PLAY. `robust` is the honest headline: does this draw still pay
    if it sells like the busier roll-downs?

    Uses the day-specific uplift when `cond.draw_date` is set - the quartiles
    within one weekday are much tighter than the mixed ones, so the range
    reported here stops spanning day-level differences that a dated forecast
    does not actually face.
    """
    if not cond.roll_down:
        return None
    line = list(line) if line is not None else best_unpopular_reference_line()
    up_mid, up_p25, up_p75 = mbw_uplift(cond.draw_date)
    baseline = cond.tickets_sold / up_mid

    def ev_at(uplift: float) -> float:
        return line_ev(line, replace(cond, tickets_sold=max(int(baseline * uplift), 1)))

    low, high = ev_at(up_p75), ev_at(up_p25)
    return {
        "ev_low": low,            # busier draw (p75 uplift) - the pessimistic case
        "ev_high": high,          # quieter draw (p25 uplift)
        "tickets_low": int(baseline * up_p25),
        "tickets_high": int(baseline * up_p75),
        "uplift": up_mid,
        "robust": low >= threshold,
    }


def should_play(cond: DrawConditions, threshold: float = 0.0) -> dict:
    """Decide whether the draw is worth entering at all.

    Uses a maximally unpopular reference line - if even that line's EV is
    below `threshold`, skip the draw. Default threshold 0.0 = only play
    +EV draws (in practice: rare Must-Be-Won roll-downs / huge rollovers).

    `play` is the verdict at the central sales estimate. For a Must-Be-Won draw
    `sales_sensitivity` carries the same verdict across the plausible range of
    ticket sales - see there for why the central number alone is not enough.
    """
    reference = best_unpopular_reference_line()
    ev = line_ev(reference, cond)
    return {
        "play": ev >= threshold,
        "ev_best_line": ev,
        "reference_line": list(reference),
        "threshold": threshold,
        "break_even_jackpot": break_even_jackpot(cond, reference),
        "sales_sensitivity": sales_sensitivity(cond, threshold, reference),
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


def rolldown_draw_numbers(tiers_df) -> set:
    """Draw numbers in `tiers_df` that paid roll-down (boosted) low tiers.

    A roll-down redistributes the jackpot into Match 3 / Match 2, so those tiers
    pay well above their base rate (draw 3190 paid 24 / 5 against a base of
    10 / 1). Flagging them by that signature keeps this self-contained: the
    feed's `next_jackpot_roll_down` column describes the draw AFTER the row, so
    it cannot classify the earliest draw in a file - which is exactly draw 3190
    in the collected data.

    Same minority argument as `calibrate_fixed_prizes`: the median Match 3 price
    is the base rate because roll-downs are ~1 draw in 12, so anything far above
    that median is a boost, not the norm.
    """
    empty: set = set()
    if tiers_df is None or len(tiers_df) == 0:
        return empty
    # prize_total is absent from some callers' frames (and from the older
    # fixtures); without it there is no boost to detect, so flag nothing rather
    # than raising - the caller falls back to the whole window.
    if not {"tier", "winners", "prize_total", "draw_number"} <= set(tiers_df.columns):
        return empty
    rows = tiers_df[(tiers_df["tier"] == TIER_MATCH_3) & (tiers_df["winners"] > 0)
                    & (tiers_df["prize_total"] > 0)]
    if len(rows) < 3:
        return empty
    per_winner = sorted(rows["prize_total"] / rows["winners"])
    base = per_winner[len(per_winner) // 2]
    if base <= 0:
        return empty
    boosted = rows[(rows["prize_total"] / rows["winners"]) > base * 1.2]
    return set(int(d) for d in boosted["draw_number"].unique())


def must_be_won_after_cap(pools_df) -> set:
    """Draws that were Must-Be-Won because the rollover before them hit the cap.

    `pools_df` is data/draw_pools.csv, whose rollover_count comes straight from
    the feed. The count belongs to the draw it is on - 5 means "this pool has
    rolled five times" - so it identifies the NEXT draw, which the cap forces
    to pay out. Verified on 3183 -> 3184, 3189 -> 3190 and 3195 -> 3196, the
    only three Must-Be-Won draws of the two-round era.

    What it cannot see is a PROMOTIONAL Must-Be-Won: the procedures let Allwyn
    designate any draw one (special draws are the usual case), and no rollover
    count precedes that. `must_be_won_draw_numbers` catches those, from the
    forward-looking flag the collector stores. Neither has happened in the
    two-round era yet.
    """
    empty: set = set()
    if pools_df is None or len(pools_df) == 0:
        return empty
    if not {"draw_number", "rollover_count"} <= set(pools_df.columns):
        return empty
    import pandas as pd
    capped = set()
    for _, row in pools_df.iterrows():
        count = row["rollover_count"]
        if pd.notna(count) and int(count) == ROLLOVER_CAP:
            capped.add(int(row["draw_number"]))
    order = sorted(int(d) for d in pd.unique(pools_df["draw_number"]))
    return {nxt for prev, nxt in zip(order, order[1:]) if prev in capped}


def exact_sales_baseline(pools_df, draw_date: date | None = None,
                         last_n_draws: int = 20) -> int | None:
    """Ordinary-draw lines sold, measured rather than inferred.

    The same quantity `estimate_tickets_sold` builds from winner counts, over
    the same kind of window and with the same two filters - Must-Be-Won draws
    excluded, restricted to the target's weekday - but read off the pools
    instead. None when fewer than MIN_EXACT_OBSERVATIONS draws qualify, which
    is the caller's signal to fall back.

    Outliers are left in and handled by the median rather than by a rule: the
    2026-07-04 Millionaire Raffle Saturday sold 11.6m lines against 8.2-9.5m
    for ordinary Saturdays of the same cycle, and writing a special-draw
    exception would cost more than it buys while the median already ignores it.
    """
    exact = exact_lines_sold(pools_df)
    if not exact:
        return None
    must_be_won = must_be_won_after_cap(pools_df)
    dates = {int(r["draw_number"]): str(r["draw_date"])
             for _, r in pools_df.iterrows()}
    recent = sorted(int(d) for d in pools_df["draw_number"])[-last_n_draws:]
    values = []
    for draw in recent:
        if draw in must_be_won or draw not in exact:
            continue
        if draw_date is not None:
            when = date.fromisoformat(dates[draw])
            if when.weekday() != draw_date.weekday():
                continue
        values.append(exact[draw])
    if len(values) < MIN_EXACT_OBSERVATIONS:
        return None
    values.sort()
    return values[len(values) // 2]


def must_be_won_draw_numbers(tiers_df) -> set:
    """Draw numbers in `tiers_df` that were Must-Be-Won, rolled down or not.

    A Must-Be-Won draw won outright leaves no trace in its own prize table -
    draw 3196 paid ordinary Match 3 / Match 2 prizes because two tickets hit
    six - but it still SOLD like one: 9.46m lines against ~8.4m for an ordinary
    Saturday of the same cycle. `rolldown_draw_numbers` cannot see it, so until
    now such draws sat in the "ordinary" baseline and inflated it, and that
    baseline is then multiplied by the Must-Be-Won uplift. Counting the effect
    once in the baseline and again in the uplift is double-counting; it pushes
    EV pessimistic, which is the safe direction and still wrong.

    The flag is forward-looking - the rows of the PREVIOUS draw carry
    next_jackpot_roll_down for this one - so the earliest draw in a file cannot
    be classified this way. That case is `rolldown_draw_numbers`, which reads
    the prize table itself.
    """
    empty: set = set()
    if tiers_df is None or len(tiers_df) == 0:
        return empty
    if not {"draw_number", "next_jackpot_roll_down"} <= set(tiers_df.columns):
        return empty
    import pandas as pd
    flagged = set()
    for draw, group in tiers_df.groupby("draw_number"):
        values = group["next_jackpot_roll_down"].dropna()
        if any(str(v).strip().lower() in ("true", "y", "yes", "1") for v in values):
            flagged.add(int(draw))
    order = sorted(int(d) for d in pd.unique(tiers_df["draw_number"]))
    return {nxt for prev, nxt in zip(order, order[1:]) if prev in flagged}


def estimate_tickets_sold(tiers_df, last_n_draws: int = 20,
                          roll_down: bool = False,
                          draw_date: date | None = None,
                          pools_df=None) -> int | None:
    """Estimate lines sold for the UPCOMING draw from recent draws.

    For fixed-prize tiers the expected winner count is N x P(tier), so each
    observation gives N ~= winners / P. Uses the high-count tiers (match 4/3/2,
    least distorted by jackpot-sharing) over recent draws, both rounds, and
    takes the median to damp popularity-of-drawn-numbers noise.

    Given `pools_df` (data/draw_pools.csv) the baseline is not estimated at
    all - `exact_sales_baseline` reads it off the pool each draw carried, an
    identity from the 8.88% clause. Winner counts remain the fallback and
    remain what the rest of this docstring describes, because the pools reach
    back only as far as the feed's ~180-day window has been collected.

    The uplift multiplication is unchanged either way, and that is now the
    weakest link: MBW_UPLIFT_BY_WEEKDAY was calibrated on winner-count ratios
    from the one-round era, and the three live two-round Saturdays read
    1.02-1.14 against its 1.27 on exact data. Until those constants are
    re-measured this overstates N on a Must-Be-Won draw, which understates EV -
    the safe direction, and the one the scorecard now makes visible.

    Three steps, because the answer depends on WHICH draw is being forecast:

    1. Measure the ordinary-draw sales level, excluding every Must-Be-Won draw
       in the window - the ones that rolled down and the ones won outright,
       which sell the same and are invisible in their own prize tables. With a full 20-draw window one or two roll-downs barely move
       a median, but the collected file started at 3190 - a roll-down - and with
       6 draws on record that single draw was 1/6 of the sample and pulled the
       estimate up 10% (6,574,552 vs 5,913,930).
    2. Restrict the baseline to the target's own weekday when `draw_date` is
       given: Saturdays sell ~1.59x Wednesdays (data/sales_history.csv), which
       is by far the largest systematic factor in N - larger than the
       Must-Be-Won effect itself. A mixed-day median is a number for a draw
       that never happens.
    3. If the upcoming draw is Must-Be-Won, scale by the day-specific uplift
       (see MBW_UPLIFT_BY_WEEKDAY). Skipping the uplift entirely was the bug
       found 2026-08-06: the 2026-08-08 alert priced a Must-Be-Won draw at
       ordinary-draw sales and reported EV +GBP 0.038 when break-even sat only
       3.2% away in N. The one Must-Be-Won draw with full data, 3190, returned
       GBP 1.586 per GBP 2 line.
    """
    uplift = mbw_uplift(draw_date)[0] if roll_down else 1.0

    # Measured beats inferred: where data/draw_pools.csv covers enough of the
    # window, the baseline is an identity rather than an estimate.
    measured = exact_sales_baseline(pools_df, draw_date, last_n_draws)
    if measured is not None:
        return int(measured * uplift)

    if tiers_df is None or len(tiers_df) == 0:
        return None
    tier_probs = {4: P_MATCH_4, 5: P_MATCH_3, 6: P_MATCH_2}
    recent_draws = sorted(tiers_df["draw_number"].unique())[-last_n_draws:]
    window = tiers_df[
        tiers_df["draw_number"].isin(recent_draws)
        & tiers_df["tier"].isin(tier_probs)
        & (tiers_df["winners"] > 0)
    ]
    if len(window) == 0:
        return None

    # Baseline is the ORDINARY level; roll-downs are re-added via the uplift.
    # If the window is nothing but roll-downs there is no ordinary level to
    # measure, so fall back to the whole window rather than returning None.
    recent = tiers_df[tiers_df["draw_number"].isin(recent_draws)]
    boosted = rolldown_draw_numbers(recent) | must_be_won_draw_numbers(recent)
    sample = window[~window["draw_number"].isin(boosted)] if boosted else window
    if len(sample) == 0:
        sample = window

    # Same-weekday half of the window. Falls back to the mixed sample when the
    # frame carries no dates or the filter would empty it (a fresh collection
    # file may hold only a handful of draws, all on the other day).
    if draw_date is not None and "draw_date" in sample.columns:
        weekdays = _column_weekdays(sample["draw_date"])
        same_day = sample[weekdays == draw_date.weekday()]
        if len(same_day) >= 3:
            sample = same_day

    estimates = sorted(
        row["winners"] / tier_probs[int(row["tier"])] for _, row in sample.iterrows()
    )
    baseline = estimates[len(estimates) // 2]
    return int(baseline * uplift)


def _column_weekdays(dates):
    """Weekday (Mon=0) for a column of ISO date strings, NaN-safe."""
    import pandas as pd
    return pd.to_datetime(dates, errors="coerce").dt.weekday


# UK Lotto pays the jackpot out at the latest on the 6th draw of a roll:
# after ROLLOVER_CAP consecutive rollovers the next draw is Must-Be-Won and
# rolls down to the lower tiers if nobody hits six. Read off the data rather
# than the rulebook: across every draw since Nov 2018 no rollover streak
# exceeded 5, and all 70 draws that reached 5 were resolved right there - 53
# rolled down, 17 had an outright jackpot winner.
ROLLOVER_CAP = 5
DRAW_WEEKDAYS = (2, 5)          # Wednesday, Saturday

# Sales for a draw close at 19:30 UK time; the draw follows at ~20:00. Any
# question of the form "which draw am I looking at?" has to be answered against
# this clock, not against the calendar day: on Wednesday afternoon the draw you
# can still enter is TONIGHT'S, while at 22:45 the same evening it is already
# Saturday's. A date alone cannot tell those two apart.
UK_TZ = ZoneInfo("Europe/London")
SALES_CLOSE = time(19, 30)


def _london(moment: datetime | date | None) -> datetime:
    """Normalise a moment to Europe/London.

    A bare `date` is read as the START of that day - i.e. sales open - which is
    how "what does the picture look like on 2026-08-08?" naturally reads.
    """
    if moment is None:
        return datetime.now(UK_TZ)
    if isinstance(moment, datetime):        # check first: datetime subclasses date
        return (moment.astimezone(UK_TZ) if moment.tzinfo
                else moment.replace(tzinfo=UK_TZ))
    return datetime.combine(moment, time(0, 0), tzinfo=UK_TZ)


def next_draw_dates(after: date | None = None, count: int = 1) -> List[date]:
    """The next `count` draw dates strictly after `after` (default: today).

    Pure calendar arithmetic: it never returns `after` itself, even when `after`
    is a draw day. To ask which draw a ticket bought *now* would enter, use
    `upcoming_draw_date` - see the warning there.
    """
    d = (after or date.today())
    if isinstance(d, datetime):
        d = d.date()
    out: List[date] = []
    while len(out) < count:
        d += timedelta(days=1)
        if d.weekday() in DRAW_WEEKDAYS:
            out.append(d)
    return out


def upcoming_draw_date(now: datetime | date | None = None) -> date:
    """The draw a ticket bought at `now` would actually enter.

    On a draw day that is TODAY's draw while sales are open, and the following
    draw once they have closed. `next_draw_dates` cannot answer this - it always
    steps past today - and using it here was wrong in two places at once:
    the ROI ledger filed a Wednesday-afternoon ticket under Saturday's draw (so
    `settle` scored it against results it never played), and the Must-Be-Won
    forecast landed one draw into the future every Wednesday and Saturday.
    """
    moment = _london(now)
    today = moment.date()
    if today.weekday() in DRAW_WEEKDAYS and moment.time() < SALES_CLOSE:
        return today
    return next_draw_dates(today, 1)[0]


def forecast_must_be_won(rollover_count: int,
                         now: datetime | date | None = None,
                         jackpot: float | None = None,
                         pools_df=None) -> dict:
    """How many draws until the jackpot must be paid out, on what date, at what pool.

    `rollover_count` is the post-draw counter carried by the official feed
    (data/prize_tiers.csv), so the upcoming draw is the one that would make it
    `rollover_count + 1`. The Must-Be-Won draw is the one entered on a count of
    ROLLOVER_CAP.

    Counting starts at `upcoming_draw_date(now)`, so `draws_away == 1` means
    "the one you can buy into right now". Counting from `next_draw_dates`
    instead put the date one draw late whenever the question was asked on a
    draw day before 19:30 - i.e. every Wednesday and Saturday afternoon, which
    is exactly when anyone asks it.

    This is an UPPER BOUND on the wait, not a schedule: any jackpot win resets
    the counter and pushes the date out. It is for planning a budget - the
    authoritative one-draw-ahead signal stays the official feed's
    `next_jackpot_roll_down` flag, which `DrawConditions.roll_down` carries.

    Given `jackpot` (the estimate advertised for the upcoming draw) and
    `pools_df`, it also projects the POOL that Must-Be-Won draw would carry.
    The advertised estimate already contains the upcoming draw's own sales -
    3203 closed at GBP 5,255,577 and 3204 was advertised at GBP 6,809,577, a
    Saturday's worth of contribution ahead - so each FURTHER draw adds
    `share x sales`, at the exact per-weekday sales level, with the uplift on
    the Must-Be-Won draw itself because its own sales fund its own pool.

    That last point cuts both ways and is worth stating: the uplift constant
    that inflates this projection is the same one that inflates N, and EV on a
    roll-down is J / N, so the two errors partly cancel. What the projection
    does NOT model is the rollover stage - sales climb through a cycle (Walker
    & Wheeler 2018: ~GBP 1.8m more Saturday sales per GBP 1m of rollover), and
    a median over whole cycles understates the last draw of one. So read the
    projection as a floor with about a draw's worth of slack, which is enough
    to tell "nowhere near break-even" from "worth watching" - and that is all
    it is for.
    """
    draws_away = max(ROLLOVER_CAP - int(rollover_count) + 1, 1)
    first = upcoming_draw_date(now)
    dates = [first] + next_draw_dates(first, draws_away - 1)

    projected_pool = None
    if jackpot is not None and pools_df is not None:
        running = float(jackpot)
        for when in dates[1:]:
            baseline = exact_sales_baseline(pools_df, when)
            if baseline is None:
                running = None
                break
            uplift = mbw_uplift(when)[0] if when == dates[-1] else 1.0
            running += baseline * uplift * TICKET_PRICE * JACKPOT_SHARE_OF_SALES
        projected_pool = running

    return {
        "rollover_count": int(rollover_count),
        "cap": ROLLOVER_CAP,
        "draws_away": draws_away,
        "expected_date": dates[-1],
        "is_next_draw": draws_away == 1,
        "projected_pool": projected_pool,
    }


def must_be_won_outlook(cond: "DrawConditions", pools_df,
                       now: datetime | date | None = None) -> dict | None:
    """Price the Must-Be-Won draw this roll is heading for, before it arrives.

    The advisor has always been able to say WHEN the cap forces a payout, and
    never what it would be worth when it got there - so the verdict on a
    Must-Be-Won draw only existed once the draw before it had been collected.
    That is a day's notice on the one kind of draw worth planning for.

    Returns None when the draw being priced IS the Must-Be-Won one (nothing to
    forecast) or when the pools do not cover enough of the window to project a
    pool honestly. Every figure is a forecast: read it as "worth watching" or
    "nowhere near", not as a verdict.
    """
    if cond.roll_down:
        return None
    forecast = forecast_must_be_won(cond.rollover_count, now, cond.jackpot, pools_df)
    if forecast["projected_pool"] is None:
        return None
    when = forecast["expected_date"]
    baseline = exact_sales_baseline(pools_df, when)
    if baseline is None:
        return None
    tickets = max(int(baseline * mbw_uplift(when)[0]), 1)
    future = replace(cond, jackpot=forecast["projected_pool"], tickets_sold=tickets,
                     roll_down=True, draw_date=when)
    verdict = should_play(future)
    return {
        **forecast,
        "tickets_sold": tickets,
        "break_even_jackpot": verdict["break_even_jackpot"],
        "ev_best_line": verdict["ev_best_line"],
        "play": verdict["play"],
        "robust": verdict["sales_sensitivity"]["robust"],
    }


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


def default_portfolio_seed(draw_date: date | None) -> int | None:
    """One seed per draw date - the SINGLE source of truth for both the
    advisor (latest.json, which `roi_ledger add --from-latest` records) and
    the alert email. Two seeds here once meant the email proposed one set of
    lines while the ledger recorded another; nobody noticed only because the
    verdict that day was SKIP."""
    if draw_date is None:
        return None
    return int(draw_date.strftime("%Y%m%d"))


def mbw_type(roll_down: bool, rollover_count: int) -> str | None:
    """Why the upcoming draw is Must-Be-Won, or None if it is not.

    'cap-driven': the rollover count hit the cap, the ~9/year case the
    forecast counts down to. 'special-event': the operator scheduled it
    (e.g. ~GBP 15M holiday draws) without 5 rollovers - same roll-down
    mechanics, but it arrives unannounced by the rollover counter, and its
    historical pools are recorded less reliably (draw 3131's archive pool is
    plainly wrong), so it deserves its own label wherever a human reads the
    verdict."""
    if not roll_down:
        return None
    return "cap-driven" if int(rollover_count) >= ROLLOVER_CAP else "special-event"


def abrams_garibaldi_screen(cond: DrawConditions) -> dict | None:
    """Independent second opinion for ORDINARY draws, after Abrams &
    Garibaldi, "Finding good bets in the lottery, and why you shouldn't take
    them" (Amer. Math. Monthly 117(1), 2010).

    Their decision rules assume the edge lives in the pari-mutuel jackpot of
    a single-pool game - an ordinary UK draw, and exactly NOT a Must-Be-Won
    roll-down - so this returns None for roll-downs instead of misapplying
    the theorem.

    The screen is deliberately MORE conservative than our exact EV: their
    jackpot cutoff is a sufficient condition for +EV that is robust to any
    sales level, while `break_even_jackpot` is exact at the current sales
    estimate. Passing here means the edge does not depend on the sales
    estimate at all; failing while our EV is positive means the edge is real
    but rests on N. Caveat: their theorem assumes the jackpot-plus-overhead
    fraction F >= 0.8 of the ticket; UK Lotto's F is ~0.64, so the cutoffs
    are indicative, not exact - which is fine for a second opinion.
    """
    if cond.roll_down:
        return None
    entries = cond.tickets_sold * cond.rounds
    j_entries = cond.jackpot / cond.ticket_price   # jackpot in ticket units
    f_fraction = 1.0 - cond.rounds * cond.prizes.ev_per_round() / cond.ticket_price
    cutoff = 1.4 * f_fraction * TOTAL_COMBOS * cond.ticket_price
    return {
        "n_over_j": entries / j_entries if j_entries > 0 else float("inf"),
        "sales_ok": entries < j_entries / 5.0,     # their N < J/5
        "jackpot_cutoff": cutoff,                  # their J > 1.4 * F * t
        "jackpot_ok": cond.jackpot > cutoff,
        "robust_good_bet": entries < j_entries / 5.0 and cond.jackpot > cutoff,
    }


def rolldown_tier_boosts(cond: DrawConditions) -> dict:
    """Exact Must-Be-Won split: GBP 5 to every Match 2 winner first, the
    remainder shared among Match 3 winners (official game procedures, per
    lottery.co.uk/lotto/must-be-won-draws and TNL's own statements).

    EV-equivalent to the uniform J/N term in `line_ev` - ANY rule that pays
    the whole pool out to lower-tier winners gives a uniformly random ticket
    J/N in expectation, which is why the J/N approximation validated to 2% on
    draw 3190. What the split determines is the DISTRIBUTION: to see the
    boost your line must actually hit Match 2 (p ~ 1/10) or Match 3
    (p ~ 1/96), and that concentration is what sizes the Kelly stake.

    Expected-count basis: realized per-winner boosts differ when the drawn
    set's popularity moves the winner counts (3190 paid +14 on Match 3
    against ~17 expected because it drew popular numbers).
    """
    entries = max(cond.tickets_sold, 1) * cond.rounds
    e_w2 = entries * P_MATCH_2
    e_w3 = entries * P_MATCH_3
    m2_boost = max(5.0 - cond.prizes.match_2, 0.0)
    remainder = max(cond.jackpot - m2_boost * e_w2, 0.0)
    return {
        "match_2_boost": m2_boost,
        "match_3_boost": remainder / e_w3 if e_w3 > 0 else 0.0,
        "expected_match_2_winners": e_w2,
        "expected_match_3_winners": e_w3,
    }


def line_return_distribution(line: Sequence[int], cond: DrawConditions) -> List[tuple]:
    """(payout, probability) over one draw EVENT for `line`, all rounds.

    Exact enumeration: per-round outcomes (miss, match 2..4, match 5 split by
    bonus, jackpot) convolved over the independent rounds. Jackpot payout is
    the pool discounted by expected co-winners. A Must-Be-Won draw is a
    MIXTURE: with P(no entry hits 6) - about 65% at Saturday roll-down sales,
    a real branch, not a technicality - Match 3 / Match 2 carry the
    `rolldown_tier_boosts`; otherwise the tiers pay base rates. Dropping the
    mixture and pricing the boost unconditionally would overstate the
    roll-down edge by ~1/3, which then inflates the Kelly stake built on it.
    """
    prizes = cond.prizes

    def convolve(per_round) -> dict:
        dist: dict = {}
        if cond.rounds == 1:
            pairs = iter(per_round)
        else:
            pairs = ((p1 + p2, pr1 * pr2)
                     for p1, pr1 in per_round for p2, pr2 in per_round)
        for payout, prob in pairs:
            dist[payout] = dist.get(payout, 0.0) + prob
        return dist

    def per_round_outcomes(m3: float, m2: float) -> list:
        jackpot_share = cond.jackpot * expected_cowinner_share(
            line, cond.tickets_sold, cond.rounds)
        return [
            (jackpot_share, P_JACKPOT),
            (prizes.match_5_bonus, P_MATCH_5_BONUS),
            (prizes.match_5, P_MATCH_5),
            (prizes.match_4, P_MATCH_4),
            (m3, P_MATCH_3),
            (m2, P_MATCH_2),
            (0.0, 1.0 - P_ANY_CASH),
        ]

    base = convolve(per_round_outcomes(prizes.match_3, prizes.match_2))
    if not cond.roll_down:
        return sorted(base.items())

    boosts = rolldown_tier_boosts(cond)
    boosted = convolve(per_round_outcomes(
        prizes.match_3 + boosts["match_3_boost"],
        prizes.match_2 + boosts["match_2_boost"],
    ))
    p_no_jackpot = exp(-cond.tickets_sold * cond.rounds * P_JACKPOT)
    dist: dict = {}
    for payout, prob in base.items():
        dist[payout] = dist.get(payout, 0.0) + (1.0 - p_no_jackpot) * prob
    for payout, prob in boosted.items():
        dist[payout] = dist.get(payout, 0.0) + p_no_jackpot * prob
    return sorted(dist.items())


def kelly_stake(cond: DrawConditions, bankroll: float,
                line: Sequence[int] | None = None) -> dict:
    """Kelly-optimal stake for one draw, from the exact return distribution.

    EXACT Kelly, not the f* ~= E[r]/E[r^2] approximation: for lottery-shaped
    returns the quadratic shortcut is badly wrong, because the jackpot's
    positive fat tail inflates E[r^2] while the actual downside is bounded at
    the ticket price - the approximation reads a bounded-loss bet as if the
    tail could ruin you and understates f* by orders of magnitude. Instead
    the derivative of E[log(1 + f r)], which is monotone decreasing in f, is
    bisected to its root.

    Why this is worth printing at all (MacLean-Ziemba-Blazenko 1992): where
    the edge LIVES decides whether it is playable. An edge carried by the
    jackpot at p ~ 1e-8 supports a negligible fraction of any human bankroll,
    while a Must-Be-Won roll-down carries its edge in Match 3 / Match 2 at
    p ~ 1/96 and 1/10, where the growth-optimal stake is real money. This is
    the number that turns PLAY/SKIP into "how much".

    Half-Kelly is reported alongside: ~75% of the growth at ~half the
    variance, the standard practitioner's discount for parameter error (and
    N here is estimated, not known).
    """
    line = list(line) if line is not None else best_unpopular_reference_line()
    dist = line_return_distribution(line, cond)
    cost = cond.ticket_price
    rs = [(x / cost - 1.0, p) for x, p in dist]
    mean = sum(r * p for r, p in rs)

    def growth_slope(f: float) -> float:
        return sum(p * r / (1.0 + f * r) for r, p in rs)

    # A miss loses the whole stake (r = -1), so log-growth dives to -inf as
    # f -> 1: all-in is never optimal and the root always lies in (0, 1).
    cap = 1.0 - 1e-9
    if mean <= 0:
        frac = 0.0
    elif growth_slope(cap) > 0:
        frac = cap
    else:
        lo, hi = 0.0, cap
        for _ in range(60):           # bisection: slope is monotone in f
            mid = (lo + hi) / 2
            if growth_slope(mid) > 0:
                lo = mid
            else:
                hi = mid
        frac = (lo + hi) / 2
    stake = frac * bankroll
    return {
        "kelly_fraction": frac,
        "stake_full": stake,
        "lines_full": int(stake // cost),
        "lines_half": int(stake / 2 // cost),
        "edge_per_line": mean * cost,
    }


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
        * expected_cowinner_share(line, cond.tickets_sold, cond.rounds)
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
