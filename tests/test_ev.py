"""Tests for the EV engine - probabilities, popularity model, EV, portfolio."""

import pytest

from lottery.ev import (
    DrawConditions,
    P_JACKPOT,
    P_MATCH_2,
    P_MATCH_3,
    P_MATCH_4,
    P_MATCH_5,
    P_MATCH_5_BONUS,
    TOTAL_COMBOS,
    best_unpopular_reference_line,
    expected_cowinner_share,
    line_ev,
    match_probability,
    popularity_ratio,
    should_play,
)
from lottery.ev import estimate_tickets_sold
from lottery.portfolio import MAX_PAIRWISE_OVERLAP, build_portfolio

BIRTHDAY_LINE = [3, 7, 11, 14, 21, 27]
UNPOPULAR_LINE = [34, 38, 41, 46, 53, 58]


class TestProbabilities:
    def test_total_combos(self):
        assert TOTAL_COMBOS == 45_057_474

    def test_official_odds(self):
        # Published UK Lotto (6/59) odds
        assert 1 / P_JACKPOT == pytest.approx(45_057_474)
        assert 1 / P_MATCH_5_BONUS == pytest.approx(7_509_579)
        assert 1 / P_MATCH_5 == pytest.approx(144_415, rel=1e-4)
        assert 1 / P_MATCH_4 == pytest.approx(2_180, rel=1e-3)
        assert 1 / P_MATCH_3 == pytest.approx(96.2, rel=1e-3)
        assert 1 / P_MATCH_2 == pytest.approx(10.26, rel=1e-3)

    def test_match_probabilities_sum_to_one(self):
        assert sum(match_probability(k) for k in range(7)) == pytest.approx(1.0)


class TestPopularity:
    def test_birthday_line_is_popular(self):
        assert popularity_ratio(BIRTHDAY_LINE) > 1.5

    def test_high_number_line_is_unpopular(self):
        assert popularity_ratio(UNPOPULAR_LINE) < 0.5

    def test_arithmetic_sequence_is_heavily_played(self):
        assert popularity_ratio([1, 2, 3, 4, 5, 6]) > popularity_ratio(BIRTHDAY_LINE)

    def test_cowinner_share_favors_unpopular(self):
        n = 15_000_000
        assert expected_cowinner_share(UNPOPULAR_LINE, n) > expected_cowinner_share(
            BIRTHDAY_LINE, n
        )
        assert 0.0 < expected_cowinner_share(BIRTHDAY_LINE, n) <= 1.0


class TestLineEV:
    def test_unpopular_line_has_higher_ev(self):
        cond = DrawConditions(jackpot=20_000_000)
        assert line_ev(UNPOPULAR_LINE, cond) > line_ev(BIRTHDAY_LINE, cond)

    def test_ev_increases_with_jackpot(self):
        small = line_ev(UNPOPULAR_LINE, DrawConditions(jackpot=2_000_000))
        big = line_ev(UNPOPULAR_LINE, DrawConditions(jackpot=50_000_000))
        assert big > small

    def test_rolldown_increases_ev(self):
        base = DrawConditions(jackpot=12_000_000, roll_down=False)
        mbw = DrawConditions(jackpot=12_000_000, roll_down=True)
        assert line_ev(UNPOPULAR_LINE, mbw) > line_ev(UNPOPULAR_LINE, base)

    def test_typical_draw_is_negative_ev(self):
        # A normal £2M draw must show a clear expected loss - anything else
        # would mean the model is lying
        ev = line_ev(UNPOPULAR_LINE, DrawConditions(jackpot=2_000_000))
        assert -2.0 < ev < 0.0

    def test_two_rounds_beat_one_round(self):
        two = line_ev(UNPOPULAR_LINE, DrawConditions(jackpot=2_000_000, rounds=2))
        one = line_ev(UNPOPULAR_LINE, DrawConditions(jackpot=2_000_000, rounds=1))
        assert two > one

    def test_rolldown_pool_is_per_event_not_per_round(self):
        # The MBW jackpot is ONE pool shared across rounds (Allwyn 2026), so
        # playing two rounds must not double the roll-down uplift
        def uplift(rounds):
            base = DrawConditions(jackpot=12_000_000, roll_down=False, rounds=rounds)
            mbw = DrawConditions(jackpot=12_000_000, roll_down=True, rounds=rounds)
            return line_ev(UNPOPULAR_LINE, mbw) - line_ev(UNPOPULAR_LINE, base)

        assert uplift(2) <= uplift(1)


class TestShouldPlay:
    def test_skips_ordinary_draw_at_zero_threshold(self):
        verdict = should_play(DrawConditions(jackpot=2_000_000), threshold=0.0)
        assert verdict["play"] is False

    def test_reference_line_is_valid(self):
        line = best_unpopular_reference_line()
        assert len(set(line)) == 6
        assert all(32 <= n <= 59 for n in line)  # unpopular = high numbers

    def test_reference_line_is_actually_unpopular(self):
        # Regression: the old builder produced 32,34,36,... - an arithmetic
        # sequence whose x8 pattern penalty erased the high-number advantage
        line = best_unpopular_reference_line()
        assert len({b - a for a, b in zip(line, line[1:])}) > 1
        assert popularity_ratio(line) < 0.3

    def test_lenient_threshold_allows_play(self):
        verdict = should_play(DrawConditions(jackpot=2_000_000), threshold=-2.0)
        assert verdict["play"] is True


class TestEstimateTicketsSold:
    def _tiers_df(self, n_true: int):
        import pandas as pd
        rows = []
        for tier, p in ((4, P_MATCH_4), (5, P_MATCH_3), (6, P_MATCH_2)):
            rows.append({"draw_number": 3190, "round": 1, "tier": tier,
                         "winners": int(n_true * p), "prize_total": 0.0})
        return pd.DataFrame(rows)

    def test_recovers_true_ticket_count(self):
        est = estimate_tickets_sold(self._tiers_df(8_000_000))
        assert est == pytest.approx(8_000_000, rel=0.02)

    def test_none_without_data(self):
        import pandas as pd
        assert estimate_tickets_sold(None) is None
        assert estimate_tickets_sold(pd.DataFrame(
            columns=["draw_number", "round", "tier", "winners"])) is None


class TestPortfolio:
    def test_builds_requested_size_with_constraints(self):
        cond = DrawConditions()
        portfolio = build_portfolio(5, cond, seed=42)
        assert len(portfolio) == 5
        lines = [p["line"] for p in portfolio]
        for line in lines:
            assert len(set(line)) == 6
            assert sum(1 for n in line if n > 31) >= 2
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                assert len(set(lines[i]) & set(lines[j])) <= MAX_PAIRWISE_OVERLAP

    def test_reproducible_with_seed(self):
        cond = DrawConditions()
        a = build_portfolio(3, cond, seed=7)
        b = build_portfolio(3, cond, seed=7)
        assert [p["line"] for p in a] == [p["line"] for p in b]

    def test_portfolio_prefers_unpopular(self):
        cond = DrawConditions()
        portfolio = build_portfolio(5, cond, seed=42)
        assert all(p["popularity_ratio"] < 1.0 for p in portfolio)
