#!/usr/bin/env python3
"""Backtest the wheel portfolio against every 6/59 draw in the history.

Three questions, answered from data instead of belief:
1. How would the fixed wheel have done (hits, cash at actual per-draw
   prizes, did the measured "3 if 4" guarantee ever break)?
2. Does the wheel differ from random portfolios in EV terms, or only in
   how wins clump? (Spoiler each run so far: only clumping.)
3. Is there any exploitable structure in the draws - ball-frequency chi2
   and a hot/cold memory test - that a "probability method" could use?

This is a validation tool, not a predictor: the honest expected outcome
is ~124 Match-3 hits per 1,147 draws and a ~21% cash return for ANY
10-line strategy played every draw. The wheel earns its keep on the
guarantee and the clumping, nothing else.

Usage:
  PYTHONPATH=. python scripts/backtest_wheel.py
  PYTHONPATH=. python scripts/backtest_wheel.py --pool-size 14 --replicates 500
"""

import argparse
import csv
import random
import statistics
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lottery.ev import N_BALLS, N_PICK  # noqa: E402
from scripts.wheel_play import build_wheel, unpopular_pool  # noqa: E402

DATA_DIR = Path("data")
ERA_START = "2015-10-10"  # first 6/59 draw; earlier history is 6/49
TOTAL_COMBOS = comb(N_BALLS, N_PICK)
TICKET_PRICE = 2.0
# Used only for draws missing from prize_tiers_history.csv; Match 2 pays a
# free lucky dip (cash value 0) and Match 5 + Bonus needs the bonus ball,
# which this backtest deliberately ignores (conservative).
FALLBACK_PRIZE = {3: 30.0, 4: 140.0, 5: 1750.0, 6: 5_000_000.0}
_CATEGORY_MATCH = {"Match 3": 3, "Match 4": 4, "Match 5": 5, "Match 6": 6}


def load_draws() -> list[tuple[int, str, frozenset]]:
    """(draw_number, date, winning numbers) for every 6/59-era draw."""
    draws = []
    with open(DATA_DIR / "lotto_full_history.csv") as f:
        for row in csv.DictReader(f):
            if row["Draw Date"] >= ERA_START:
                nums = frozenset(int(row[f"Number_{i}"]) for i in range(1, 7))
                draws.append((int(row["DrawNumber"]), row["Draw Date"], nums))
    draws.sort()
    return draws


def load_prizes() -> dict[int, dict[int, float]]:
    """draw_number -> {match count: actual GBP per winner}."""
    prizes: dict[int, dict[int, float]] = defaultdict(dict)
    with open(DATA_DIR / "prize_tiers_history.csv") as f:
        for row in csv.DictReader(f):
            m = _CATEGORY_MATCH.get(row["category"])
            if m is None:
                continue
            try:
                prizes[int(row["draw_number"])][m] = float(row["prize_per_winner"])
            except (TypeError, ValueError):
                continue
    return prizes


def run_portfolio(tickets, draws, prizes) -> dict:
    """Play the same tickets in every draw; tally at actual prizes."""
    ticket_sets = [set(t) for t in tickets]
    tiers: Counter = Counter()
    cash = 0.0
    wins_per_draw = []
    for dn, _, nums in draws:
        table = prizes.get(dn, {})
        wins = 0
        for ts in ticket_sets:
            m = len(ts & nums)
            if m >= 3:
                tiers[m] += 1
                cash += table.get(m, FALLBACK_PRIZE[m])
                wins += 1
        wins_per_draw.append(wins)
    return {
        "tiers": tiers,
        "cash": cash,
        "draws_with_win": sum(1 for w in wins_per_draw if w),
        "clump_variance": statistics.pvariance(wins_per_draw),
        "wins_per_draw": Counter(wins_per_draw),
    }


def guarantee_violations(pool, tickets, draws) -> tuple[int, Counter]:
    """Check '3 if 4' against real draws; also return the pool-hit histogram."""
    pool_set = set(pool)
    t_hist: Counter = Counter()
    violations = 0
    for _, _, nums in draws:
        t = len(nums & pool_set)
        t_hist[t] += 1
        if t >= 4 and max(len(set(tk) & nums) for tk in tickets) < 3:
            violations += 1
    return violations, t_hist


def uniformity_report(draws) -> None:
    n = len(draws)
    freq: Counter = Counter()
    for _, _, nums in draws:
        freq.update(nums)
    expected = n * N_PICK / N_BALLS
    chi2 = sum((freq[b] - expected) ** 2 / expected for b in range(1, N_BALLS + 1))
    # 95th percentile of chi2 with df=58
    verdict = "NON-uniform - investigate!" if chi2 > 76.8 else "consistent with uniform"
    print(f"Ball-frequency chi2 = {chi2:.1f} (df=58, 95% critical 76.8) -> {verdict}")

    p_uniform = N_PICK / N_BALLS
    for k in (5, 10, 20):
        hot = [0, 0]
        cold = [0, 0]
        for i in range(k, n):
            recent = set().union(*(draws[j][2] for j in range(i - k, i)))
            nxt = draws[i][2]
            for b in range(1, N_BALLS + 1):
                bucket = hot if b in recent else cold
                bucket[0] += 1
                bucket[1] += b in nxt
        print(f"  P(drawn | seen in last {k:2d}) = {hot[1] / hot[0]:.4f}   "
              f"P(drawn | not seen) = {cold[1] / cold[0]:.4f}   "
              f"(uniform: {p_uniform:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=12)
    parser.add_argument("--replicates", type=int, default=300,
                        help="Random baseline portfolios to simulate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the random baselines (wheel itself is deterministic)")
    args = parser.parse_args()

    draws = load_draws()
    prizes = load_prizes()
    n = len(draws)
    cost = n * args.lines * TICKET_PRICE
    print(f"Draws in 6/59 era: {n}  ({draws[0][1]} .. {draws[-1][1]})")
    print(f"Draws with actual prize data: {sum(1 for dn, _, _ in draws if prizes.get(dn))}/{n}")

    pool = unpopular_pool(args.pool_size)
    wheel = build_wheel(pool, args.lines)
    res = run_portfolio(wheel, draws, prizes)
    tiers = res["tiers"]

    print("\n=== WHEEL (fixed lines, pool", pool, ") ===")
    print(f"Match 3/4/5/6 hits: {tiers[3]}/{tiers[4]}/{tiers[5]}/{tiers[6]}")
    print(f"Cash won: £{res['cash']:,.0f}   cost: £{cost:,.0f}   "
          f"return: {100 * res['cash'] / cost:.1f}%")
    print(f"Draws with >=1 winning ticket: {res['draws_with_win']}/{n} "
          f"({100 * res['draws_with_win'] / n:.1f}%)")
    print("Winning tickets per draw:", dict(sorted(res["wins_per_draw"].items())))

    violations, t_hist = guarantee_violations(pool, wheel, draws)
    print(f"Pool-hit distribution t: {dict(sorted(t_hist.items()))}")
    expected_t = {t: n * comb(args.pool_size, t)
                  * comb(N_BALLS - args.pool_size, N_PICK - t) / TOTAL_COMBOS
                  for t in range(N_PICK + 1)}
    print("Expected under uniform:  ",
          {t: round(v, 1) for t, v in expected_t.items() if v >= 0.05})
    print(f"'3 if 4' guarantee violations on real draws: {violations}")

    p_line = sum(comb(N_PICK, k) * comb(N_BALLS - N_PICK, N_PICK - k)
                 for k in range(3, N_PICK + 1)) / TOTAL_COMBOS
    print(f"\nExpected Match3+ tickets for ANY {args.lines}-line strategy: "
          f"{p_line * n * args.lines:.1f}   wheel got: {sum(tiers.values())}")

    rng = random.Random(args.seed)
    R = args.replicates
    print(f"\n=== BASELINES ({R} replicate portfolios, same {n} draws) ===")
    baselines = {
        f"10 random lines from all {N_BALLS}":
            lambda: [rng.sample(range(1, N_BALLS + 1), N_PICK) for _ in range(args.lines)],
        f"10 random lines from the same {args.pool_size}-pool":
            lambda: [rng.sample(pool, N_PICK) for _ in range(args.lines)],
    }
    for name, make in baselines.items():
        runs = [run_portfolio(make(), draws, prizes) for _ in range(R)]
        med = lambda key: sorted(r[key] for r in runs)[R // 2]  # noqa: E731
        med_hits = sorted(sum(r["tiers"].values()) for r in runs)[R // 2]
        print(f"{name}: median M3+ hits {med_hits}, median cash £{med('cash'):,.0f}, "
              f"median draws-with-win {med('draws_with_win')}, "
              f"clump variance {med('clump_variance'):.4f}")
    print(f"Wheel clump variance: {res['clump_variance']:.4f} "
          "(higher than all-59 random = wins arrive in bursts, as designed)")

    print("\n=== IS THERE ANY STRUCTURE TO EXPLOIT? ===")
    uniformity_report(draws)

    print("\n=== POOL SIZE vs CHANCE THE '3 if 4' GUARANTEE FIRES ===")
    print("size  P(t>=3)   P(t>=4)   draws/yr with t>=4 (104 draws/yr)")
    for s in range(11, 17):
        p3, p4 = (sum(comb(s, t) * comb(N_BALLS - s, N_PICK - t)
                      for t in range(lo, N_PICK + 1)) / TOTAL_COMBOS
                  for lo in (3, 4))
        print(f"{s:4d}  {p3:.4f}    {p4:.5f}   {104 * p4:.2f}")


if __name__ == "__main__":
    main()
