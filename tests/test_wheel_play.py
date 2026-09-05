"""Tests for the wheel generator - guarantees must be measured, not assumed."""

from itertools import combinations

import pytest

from lottery.ev import N_PICK, _has_consecutive_run, number_weight
from scripts.wheel_play import (
    KNOWN_COVERINGS,
    build_wheel,
    covering_lines,
    measure_guarantees,
    unpopular_pool,
)


class TestUnpopularPool:
    def test_pool_is_least_popular_weight_band(self):
        pool = unpopular_pool(12)
        cut = max(number_weight(n) for n in pool)
        # Nothing outside the pool may be strictly less popular than its
        # worst member - otherwise "least popular" would be a lie.
        outside = set(range(1, 60)) - set(pool)
        assert all(number_weight(n) >= cut for n in outside)

    def test_tie_band_is_spread_not_contiguous(self):
        # The calibrated weights tie across a wide band; taking its lowest
        # run gave a contiguous 32-43 pool that starves the no-3-consecutive
        # candidate filter.
        pool = unpopular_pool(12)
        assert not _has_consecutive_run(pool, 3)

    def test_deterministic(self):
        assert unpopular_pool(12) == unpopular_pool(12)


class TestBuildWheel:
    def test_three_if_four_guarantee(self):
        # The whole point of the wheel: 4 pool hits -> a guaranteed Match 3.
        pool = unpopular_pool(12)
        tickets = build_wheel(pool, 10)
        guar = measure_guarantees(pool, tickets)
        assert all(guar[t] >= 3 for t in range(4, N_PICK + 1))

    def test_six_lines_buy_the_same_guarantee_as_ten(self):
        """The published (12, 6, 3, 4) design, not what greedy can find.

        Greedy reaches "3 if 4" at eight tickets and never at six - it commits
        to a locally best first line and cannot see the symmetric arrangement.
        Karim (2005) proves six is the minimum and five impossible, so this is
        the same promise for GBP 12 instead of GBP 20.
        """
        pool = unpopular_pool(12)
        assert covering_lines(pool) == 6
        six = measure_guarantees(pool, build_wheel(pool, 6))
        ten = measure_guarantees(pool, build_wheel(pool, 10))
        assert six == ten
        assert all(six[t] >= 3 for t in range(4, N_PICK + 1))

    def test_the_design_is_used_verbatim_and_maps_onto_the_pool(self):
        pool = unpopular_pool(12)
        expected = [sorted(pool[i - 1] for i in block)
                    for block in KNOWN_COVERINGS[12]]
        assert [sorted(t) for t in build_wheel(pool, 6)] == expected

    def test_extra_lines_extend_the_design_rather_than_replace_it(self):
        pool = unpopular_pool(12)
        six = [sorted(t) for t in build_wheel(pool, 6)]
        ten = [sorted(t) for t in build_wheel(pool, 10)]
        assert ten[:6] == six
        assert len({tuple(t) for t in ten}) == 10

    def test_pools_without_a_published_design_still_wheel(self):
        """Greedy is the fallback, and must not silently do nothing."""
        assert covering_lines(unpopular_pool(11)) is None
        tickets = build_wheel(unpopular_pool(11), 8)
        assert len({tuple(t) for t in tickets}) == 8

    def test_more_lines_than_candidates_is_a_clear_error(self):
        # Asking for more tickets than valid candidates must fail loudly,
        # not die inside max() on an empty sequence.
        pool = unpopular_pool(12)
        with pytest.raises(ValueError, match="valid tickets"):
            build_wheel(pool, 10_000)

    def test_tickets_are_valid_distinct_lines(self):
        pool = unpopular_pool(12)
        tickets = build_wheel(pool, 10)
        assert len({tuple(t) for t in tickets}) == 10
        for t in tickets:
            assert len(set(t)) == N_PICK
            assert set(t) <= set(pool)
            assert not _has_consecutive_run(t, 3)


class TestMeasureGuarantees:
    def test_full_wheel_guarantees_everything(self):
        # All C(7,6) tickets on a 7-number pool: for any t pool hits, the
        # ticket omitting a non-hit number contains every hit.
        pool = list(range(1, 8))
        tickets = [list(c) for c in combinations(pool, 6)]
        guar = measure_guarantees(pool, tickets)
        assert guar == {3: 3, 4: 4, 5: 5, 6: 6}

    def test_single_ticket_worst_case(self):
        # One ticket on a 12-pool: a triple drawn entirely outside it
        # matches 0 - the guarantee must report the worst case, not the mean.
        pool = list(range(1, 13))
        guar = measure_guarantees(pool, [list(range(1, 7))])
        assert guar[3] == 0
