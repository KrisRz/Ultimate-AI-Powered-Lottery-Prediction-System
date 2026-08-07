"""Exact roll-down split, per-line return distribution, Kelly staking."""

import pytest

from lottery.ev import (
    DrawConditions,
    P_MATCH_2,
    P_MATCH_3,
    best_unpopular_reference_line,
    kelly_stake,
    line_ev,
    line_return_distribution,
    rolldown_tier_boosts,
)

MBW = DrawConditions(jackpot=12_000_000, tickets_sold=9_000_000, roll_down=True)


class TestRolldownTierBoosts:
    def test_match_2_topped_up_to_five_pounds(self):
        b = rolldown_tier_boosts(MBW)
        assert b["match_2_boost"] == pytest.approx(5.0 - MBW.prizes.match_2)

    def test_remainder_goes_to_match_3(self):
        b = rolldown_tier_boosts(MBW)
        entries = MBW.tickets_sold * MBW.rounds
        remainder = MBW.jackpot - b["match_2_boost"] * entries * P_MATCH_2
        assert b["match_3_boost"] == pytest.approx(remainder / (entries * P_MATCH_3))
        assert b["match_3_boost"] > 0

    def test_split_is_ev_equivalent_to_uniform_j_over_n(self):
        """Any full redistribution pays J/N per ticket in expectation - the
        reason line_ev's J/N approximation validated to 2% on draw 3190."""
        b = rolldown_tier_boosts(MBW)
        per_line = MBW.rounds * (
            P_MATCH_2 * b["match_2_boost"] + P_MATCH_3 * b["match_3_boost"])
        assert per_line == pytest.approx(MBW.jackpot / MBW.tickets_sold, rel=1e-9)

    def test_tiny_pool_cannot_go_negative(self):
        small = DrawConditions(jackpot=100_000, tickets_sold=9_000_000, roll_down=True)
        b = rolldown_tier_boosts(small)
        assert b["match_3_boost"] == 0.0


class TestLineReturnDistribution:
    def test_probabilities_sum_to_one(self):
        line = best_unpopular_reference_line()
        for cond in (MBW, DrawConditions(jackpot=5_000_000)):
            dist = line_return_distribution(line, cond)
            assert sum(p for _, p in dist) == pytest.approx(1.0, abs=1e-12)

    def test_mean_matches_line_ev(self):
        """The distribution and line_ev price the same draw: their means must
        agree (line_ev nets off the ticket price; the roll-down terms differ
        only in the popularity-blind vs per-line jackpot share, both tiny)."""
        line = best_unpopular_reference_line()
        for cond in (MBW, DrawConditions(jackpot=5_000_000)):
            dist = line_return_distribution(line, cond)
            mean = sum(x * p for x, p in dist)
            assert mean - cond.ticket_price == pytest.approx(
                line_ev(line, cond), abs=0.02)

    def test_rolldown_branch_is_a_mixture(self):
        line = best_unpopular_reference_line()
        payouts = {x for x, _ in line_return_distribution(line, MBW)}
        boosts = rolldown_tier_boosts(MBW)
        base_m3 = MBW.prizes.match_3
        # both the base Match 3 and the boosted Match 3 must be reachable
        assert any(abs(x - base_m3) < 1e-6 for x in payouts)
        assert any(abs(x - (base_m3 + boosts["match_3_boost"])) < 1e-6 for x in payouts)


class TestKellyStake:
    def test_negative_ev_draw_stakes_zero(self):
        k = kelly_stake(DrawConditions(jackpot=5_000_000), bankroll=10_000)
        assert k["kelly_fraction"] == 0.0
        assert k["lines_full"] == 0

    def test_full_kelly_at_retail_bankroll_is_pennies(self):
        """The MacLean-Ziemba result reproduced: an 83% edge justified 65 x $1
        tickets per $10M of wealth (f ~ 6.5e-6); our +40% roll-down edge lands
        at the same order. A hobby bankroll gets a stake in pennies - the edge
        is real, the growth-theoretic backing for real money is not. If this
        test ever finds whole lines at 10k, the distribution lost its miss
        branch somewhere."""
        rich = DrawConditions(jackpot=16_000_000, tickets_sold=8_000_000,
                              roll_down=True)
        assert line_ev(best_unpopular_reference_line(), rich) > 0
        k = kelly_stake(rich, bankroll=10_000)
        assert 0 < k["kelly_fraction"] < 1e-4
        assert k["lines_full"] == 0
        assert k["stake_full"] < 1.0

    def test_rolldown_edge_beats_jackpot_only_edge_per_kelly(self):
        """MacLean-Ziemba: the same EV carried at p~1/96 (roll-down tiers)
        supports a far larger Kelly fraction than at p~1e-8 (jackpot). A
        huge-jackpot ordinary draw and a modest roll-down with comparable EV
        must produce wildly different stakes."""
        rolldown = DrawConditions(jackpot=16_000_000, tickets_sold=8_000_000,
                                  roll_down=True)
        jackpot_only = DrawConditions(jackpot=45_000_000, tickets_sold=8_000_000)
        line = best_unpopular_reference_line()
        ev_rd = line_ev(line, rolldown)
        ev_jp = line_ev(line, jackpot_only)
        assert ev_rd > 0 and ev_jp > 0        # both are +EV draws
        k_rd = kelly_stake(rolldown, bankroll=1_000_000)
        k_jp = kelly_stake(jackpot_only, bankroll=1_000_000)
        assert k_rd["kelly_fraction"] > 50 * k_jp["kelly_fraction"]
