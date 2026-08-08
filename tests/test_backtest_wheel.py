"""Tests for the wheel backtest - synthetic draws, no data files needed."""

from collections import Counter

from scripts.backtest_wheel import guarantee_violations, run_portfolio


def _draws(*number_sets):
    return [(i, f"2020-01-{i + 1:02d}", frozenset(ns))
            for i, ns in enumerate(number_sets)]


class TestRunPortfolio:
    def test_actual_prizes_beat_fallback(self):
        # Draw 0 pays Match 3 at £25 from the real table; draw 1 has no
        # table so the £30 fallback applies.
        tickets = [[1, 2, 3, 50, 51, 52]]
        draws = _draws({1, 2, 3, 40, 41, 42}, {1, 2, 3, 43, 44, 45})
        res = run_portfolio(tickets, draws, {0: {3: 25.0}})
        assert res["tiers"] == Counter({3: 2})
        assert res["cash"] == 25.0 + 30.0
        assert res["draws_with_win"] == 2

    def test_match_below_three_pays_nothing(self):
        tickets = [[1, 2, 3, 4, 5, 6]]
        draws = _draws({1, 2, 50, 51, 52, 53})
        res = run_portfolio(tickets, draws, {})
        assert res["cash"] == 0.0
        assert res["draws_with_win"] == 0

    def test_clumping_counts_tickets_not_draws(self):
        # Two tickets both hit Match 3 in one draw -> one winning draw,
        # two winning tickets.
        tickets = [[1, 2, 3, 50, 51, 52], [1, 2, 3, 53, 54, 55]]
        draws = _draws({1, 2, 3, 40, 41, 42})
        res = run_portfolio(tickets, draws, {})
        assert res["tiers"][3] == 2
        assert res["draws_with_win"] == 1
        assert res["wins_per_draw"] == Counter({2: 1})


class TestGuaranteeViolations:
    def test_detects_a_broken_wheel(self):
        # 4 pool numbers drawn, best ticket matches only 2 -> violation.
        pool = list(range(1, 13))
        tickets = [[1, 2, 20, 21, 22, 23]]
        draws = _draws({1, 2, 3, 4, 40, 41})
        violations, t_hist = guarantee_violations(pool, tickets, draws)
        assert violations == 1
        assert t_hist == Counter({4: 1})

    def test_holding_wheel_passes(self):
        pool = list(range(1, 13))
        tickets = [[1, 2, 3, 4, 5, 6]]
        draws = _draws({1, 2, 3, 4, 40, 41})
        violations, _ = guarantee_violations(pool, tickets, draws)
        assert violations == 0
